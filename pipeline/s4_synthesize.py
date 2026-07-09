"""s4: per-segment TTS through pipeline.tts_engine with content-hash caching.

Auto-repair: autoregressive takes vary (measured spread ~0.74-0.80 on identical
config); a take scoring below tts.retake_mos_below is re-rolled up to
tts.best_of times and the best take wins. Cached segments are never re-judged —
the hash-named file is the accepted take."""
from __future__ import annotations
import logging
from pathlib import Path
import soundfile as sf
from . import manifest as M
from .tts_engine import synth_best_of

log = logging.getLogger("dubadabidu.s4")


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    # per-video overrides (e.g. this video's own ref picked by `preamble`)
    t = {**cfg["tts"], **man.get("tts_overrides", {})}
    wd = M.video_workdir(cfg, video)

    for lang in langs:
        seg_dir = wd / "seg" / lang
        seg_dir.mkdir(parents=True, exist_ok=True)
        n_new = 0
        total = len(man["utterances"])
        for k, u in enumerate(man["utterances"], 1):
            tr = u["tr"].get(lang)
            if not tr or not tr.get("text"):
                raise SystemExit(f"{u['id']} missing {lang} translation — run s3.")
            h = M.synth_hash(tr["text"], lang, t)
            out = seg_dir / f"{u['id']}_{h}.wav"
            fresh = not out.exists()
            if fresh:
                synth_best_of(tr["text"], lang, out, t)
                n_new += 1
            info = sf.info(str(out))
            tr["synth"] = str(out.relative_to(wd))
            tr["synth_dur"] = round(info.frames / info.samplerate, 3)
            if fresh:  # checkpoint each new synth: long best-of units are minutes each
                M.save(cfg, video, man)
                if n_new % 25 == 0:
                    log.info("%s: %d/%d ...", lang, k, total)
        man["stages"][f"s4_{lang}"] = "done"
        M.save(cfg, video, man)
        log.info("%s: %d synthesized, %d cached", lang, n_new, total - n_new)
