"""s5: place segments on a soft-anchored timeline. Each dub starts at its
source time or right after the previous dub (retime_step in logic.py) and may
drift forward within fit.drift_max_s; drift resets naturally at source pauses.
Within the resulting slot the old ladder applies: as-is -> atempo<=max_tempo
-> pre-generated shorter variant -> LLM emergency shorten -> flagged overflow.
Overlaps are impossible by construction; excessive drift is recorded as
drift_exceeded (report/autopilot surface it where overlaps used to be)."""
from __future__ import annotations
import logging, subprocess
from pathlib import Path
import soundfile as sf
from . import manifest as M
from .logic import choose_placement, retime_step
from .s3_translate import shorten as llm_shorten
from .tts_engine import synth_best_of

log = logging.getLogger("dubadabidu.s5")


def _dur(p: Path) -> float:
    i = sf.info(str(p))
    return i.frames / i.samplerate


_STRETCH_FILTER: str | None = None


def _stretch_filter(fit_cfg: dict) -> str:
    """rubberband (R3 engine — keeps consonant transients cleaner than atempo's
    WSOLA) when ffmpeg is built with librubberband, else atempo. Probed once.
    fit.stretcher: auto (default) | rubberband | atempo."""
    global _STRETCH_FILTER
    want = fit_cfg.get("stretcher", "auto")
    if want == "atempo":
        return "atempo"
    if _STRETCH_FILTER is None:
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
        _STRETCH_FILTER = "rubberband" if "rubberband" in filters else "atempo"
        if _STRETCH_FILTER == "atempo":
            (log.warning if want == "rubberband" else log.info)(
                "ffmpeg lacks the rubberband filter (needs a librubberband "
                "build); stretching with atempo")
    return _STRETCH_FILTER


def _atempo(src: Path, dst: Path, tempo: float, fit_cfg: dict) -> None:
    flt = _stretch_filter(fit_cfg)
    spec = (f"rubberband=tempo={tempo:.4f}" if flt == "rubberband"
            else f"atempo={tempo:.4f}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-filter:a", spec, str(dst)], check=True)


def run(cfg: dict, video: str, langs: list[str]) -> None:
    f = cfg["fit"]
    drift_max = f.get("drift_max_s", 1.5)
    min_gap = f.get("min_gap_s", 0.15)
    man = M.load(cfg, video)
    # per-video overrides (e.g. this video's own ref picked by `preamble`);
    # synth_hash sees the merged dict, so caches stay per-ref consistent
    t = {**cfg["tts"], **man.get("tts_overrides", {})}
    wd = M.video_workdir(cfg, video)
    us = man["utterances"]

    for lang in langs:
        stats = {"ok": 0, "stretched": 0, "shortened": 0, "overflow": 0}
        prev_end = 0.0
        for i, u in enumerate(us):
            tr = u["tr"].get(lang)
            if not tr or not tr.get("text"):
                raise SystemExit(f"{u['id']} missing {lang} translation — "
                                 f"run s3 first (only s2 is done for this video).")
            last = i + 1 >= len(us)
            next_start = man["duration"] if last else us[i + 1]["start"]
            placed_start, drift, slot = retime_step(
                u["start"], prev_end, next_start, drift_max,
                0.0 if last else min_gap, hard_end=man["duration"])
            candidates = [tr["text"]] + tr.get("variants", [])
            tu = t

            def seg_wav(text: str) -> Path:
                h = M.synth_hash(text, lang, tu)
                wav = wd / "seg" / lang / f"{u['id']}_{h}.wav"
                if not wav.exists():
                    # same gate/ranking as s4: a variant that replaces the
                    # primary in the mix must not be an ungated single take.
                    # target = this segment's actual slot: a variant exists
                    # to FIT, so fitting takes are preferred outright.
                    synth_best_of(text, lang, wav, tu, target_dur=slot or None)
                return wav

            soft = f.get("soft_tempo", 1.06)
            # synth the primary; only synth variants if it needs a hard stretch
            wavs = [seg_wav(candidates[0])]
            if slot <= 0 or _dur(wavs[0]) / slot > soft:
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
                _atempo(wav, fitted, tempo, f)
                placed = (fitted, tempo, "stretched" if ci == 0 else "shortened")
            else:
                fitted = wav.with_name(wav.stem + "_fit.wav")
                _atempo(wav, fitted, f["max_tempo"], f)
                placed = (fitted, f["max_tempo"], "overflow")
                log.warning("%s %s overflow (slot %.1fs)", lang, u["id"], slot)
            tr["fitted"] = str(placed[0].relative_to(wd))
            tr["fitted_text"] = chosen          # exact text TTS spoke (may be a variant)
            tr["tempo"] = round(placed[1], 3)
            tr["fit"] = placed[2]
            stats[placed[2]] += 1
            fit_dur = _dur(placed[0])
            tr["placed_start"] = round(placed_start, 3)
            tr["placed_end"] = round(placed_start + fit_dur, 3)
            tr["drift"] = round(drift, 3)
            # overflow can push the NEXT segment past its drift budget; record
            # it so report/autopilot surface it (successor of the old overlap
            # flag — actual overlap in the mix is impossible now)
            if drift > drift_max + 0.05:
                tr["drift_exceeded"] = round(drift - drift_max, 2)
                log.warning("%s %s drift %.2fs exceeds budget %.1fs",
                            lang, u["id"], drift, drift_max)
            else:
                tr.pop("drift_exceeded", None)
            tr.pop("overrun_s", None)  # legacy flag from the hard-anchor era
            prev_end = placed_start + fit_dur
        man["stages"][f"s5_{lang}"] = "done"
        M.save(cfg, video, man)
        log.info("%s: %s", lang, stats)
