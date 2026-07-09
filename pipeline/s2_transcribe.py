"""s2: faster-whisper 1.2.1 (large-v3, uk) -> manifest utterances.

Merging logic lives in pipeline/logic.py (unit-tested). AFTER THIS STAGE:
hand-review text_uk in the manifest — one terminology fix here propagates
to all target languages.
"""
from __future__ import annotations
import logging, subprocess
from . import manifest as M
from .device import whisper_device
from .logic import merge_segments, split_at_pauses

log = logging.getLogger("dubadabidu.s2")


def run(cfg: dict, video: str) -> None:
    from faster_whisper import WhisperModel

    wd = M.video_workdir(cfg, video)
    old = M.manifest_path(cfg, video)
    if old.exists():
        import datetime as dt
        import shutil
        bak = old.with_name(
            f"manifest.json.bak-{dt.datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy(old, bak)
        import json
        stages = json.loads(old.read_text(encoding="utf-8")).get("stages", {})
        done = [k for k in stages if k.startswith(("s3_", "s4_"))]
        if done:
            log.warning("re-transcribing DISCARDS existing translations/synth "
                        "state (%s) — old manifest kept at %s", done, bak.name)
    a = cfg["asr"]
    dev, ctype = whisper_device(a.get("device", "auto"))
    log.info("ASR %s on %s/%s", a["model"], dev, ctype)
    model = WhisperModel(a["model"], device=dev, compute_type=ctype)
    segments, _ = model.transcribe(
        str(wd / "vocals.wav"), language=cfg["source_language"],
        vad_filter=a["vad_filter"], word_timestamps=True,
        temperature=0.0)  # no fallback sampling: segmentation must be reproducible

    raw = []
    for s in segments:
        words = [{"start": w.start, "end": w.end, "word": w.word}
                 for w in (s.words or [])]
        if words and a.get("pause_split_s"):
            raw += split_at_pauses(words, a["pause_split_s"])
        else:
            raw.append({"start": s.start, "end": s.end, "text": s.text})
    merged = merge_segments(raw, a["max_chars"], a["max_seconds"])

    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video]).strip())

    man = {"video": video, "duration": dur, "stages": {"s2": "done"}, "utterances": [
        {"id": f"u{i:04d}", "start": round(u["start"], 3), "end": round(u["end"], 3),
         "text_uk": u["text"], "tr": {}}
        for i, u in enumerate(merged, 1)]}
    M.save(cfg, video, man)
    log.info("%d utterances -> %s", len(merged), M.manifest_path(cfg, video))
    log.info(">>> Review text_uk in the manifest before translating. <<<")
