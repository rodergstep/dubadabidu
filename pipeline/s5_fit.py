"""s5: fit segments into time slots. Ladder: as-is -> atempo<=max_tempo ->
pre-generated shorter variant -> flagged overflow. Decision logic in logic.py."""
from __future__ import annotations
import logging, subprocess
from pathlib import Path
import soundfile as sf
from . import manifest as M
from .logic import choose_placement
from .s3_translate import shorten as llm_shorten
from .tts_engine import synth_best_of

log = logging.getLogger("dubadabidu.s5")


def _dur(p: Path) -> float:
    i = sf.info(str(p))
    return i.frames / i.samplerate


def _atempo(src: Path, dst: Path, tempo: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-filter:a", f"atempo={tempo:.4f}", str(dst)], check=True)


def run(cfg: dict, video: str, langs: list[str]) -> None:
    f = cfg["fit"]
    man = M.load(cfg, video)
    # per-video overrides (e.g. this video's own ref picked by `preamble`);
    # synth_hash sees the merged dict, so caches stay per-ref consistent
    t = {**cfg["tts"], **man.get("tts_overrides", {})}
    wd = M.video_workdir(cfg, video)
    us = man["utterances"]

    for lang in langs:
        stats = {"ok": 0, "stretched": 0, "shortened": 0, "overflow": 0}
        for i, u in enumerate(us):
            tr = u["tr"].get(lang)
            if not tr or not tr.get("text"):
                raise SystemExit(f"{u['id']} missing {lang} translation — "
                                 f"run s3 first (only s2 is done for this video).")
            next_start = us[i + 1]["start"] if i + 1 < len(us) else man["duration"]
            slot = (u["end"] - u["start"]) + min(f["borrow_gap_s"],
                                                 max(0.0, next_start - u["end"]))
            candidates = [tr["text"]] + tr.get("variants", [])

            def seg_wav(text: str) -> Path:
                h = M.synth_hash(text, lang, t)
                wav = wd / "seg" / lang / f"{u['id']}_{h}.wav"
                if not wav.exists():
                    # same MOS gate as s4: a variant that replaces the primary
                    # in the mix must not be an ungated single take
                    synth_best_of(text, lang, wav, t)
                return wav

            soft = f.get("soft_tempo", 1.06)
            # synth the primary; only synth variants if it needs a hard stretch
            wavs = [seg_wav(candidates[0])]
            if _dur(wavs[0]) / slot > soft:
                wavs += [seg_wav(c) for c in candidates[1:]]
            durs = [_dur(w) for w in wavs]
            ci, verdict, tempo = choose_placement(durs, slot, f["max_tempo"], soft)
            if verdict == "no":
                # emergency rescue: every pre-generated variant overflows, so
                # ask the LLM for rewrites under a char budget derived from the
                # measured pace of the best candidate. Degrades to the overflow
                # flag when the endpoint is unavailable. New variants land in
                # the manifest, so re-runs place them from cache with no call.
                k = min(range(len(durs)), key=lambda j: durs[j])
                budget = int(len(candidates[k]) * slot * f["max_tempo"]
                             / durs[k] * 0.95)
                extra = [v for v in llm_shorten(cfg, lang, u["text_uk"],
                                                candidates[0], budget)
                         if v not in candidates]
                if extra:
                    candidates += extra
                    tr["variants"] = candidates[1:]
                    wavs += [seg_wav(c) for c in extra]
                    durs = [_dur(w) for w in wavs]
                    ci, verdict, tempo = choose_placement(
                        durs, slot, f["max_tempo"], soft)
                    if verdict != "no":
                        log.info("%s %s rescued by emergency shorten "
                                 "(%d chars budget)", lang, u["id"], budget)
            chosen, wav = candidates[ci], wavs[ci]
            if verdict == "as_is":
                placed = (wav, 1.0, "ok" if ci == 0 else "shortened")
            elif verdict == "stretch":
                fitted = wav.with_name(wav.stem + "_fit.wav")
                _atempo(wav, fitted, tempo)
                placed = (fitted, tempo, "stretched" if ci == 0 else "shortened")
            else:
                fitted = wav.with_name(wav.stem + "_fit.wav")
                _atempo(wav, fitted, f["max_tempo"])
                placed = (fitted, f["max_tempo"], "overflow")
                log.warning("%s %s overflow (slot %.1fs)", lang, u["id"], slot)
            tr["fitted"] = str(placed[0].relative_to(wd))
            tr["fitted_text"] = chosen          # exact text TTS spoke (may be a variant)
            tr["tempo"] = round(placed[1], 3)
            tr["fit"] = placed[2]
            stats[placed[2]] += 1
            # s6 overlays at u.start with no collision check — audio spilling
            # past the next utterance's start plays on top of it. Record it so
            # report/review surface the collision, not just the overflow flag.
            overrun = u["start"] + _dur(placed[0]) - next_start
            if overrun > 0.05:
                tr["overrun_s"] = round(overrun, 2)
                log.warning("%s %s overlaps next utterance by %.2fs",
                            lang, u["id"], overrun)
            else:
                tr.pop("overrun_s", None)
        man["stages"][f"s5_{lang}"] = "done"
        M.save(cfg, video, man)
        log.info("%s: %s", lang, stats)
