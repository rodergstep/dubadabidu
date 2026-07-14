"""s2: Whisper large-v3 (uk) -> manifest utterances.

Transcription backend (faster-whisper on CUDA/CPU, mlx-whisper on Apple
Silicon) is chosen in pipeline/asr.py; this stage only shapes the normalized
segments into the manifest. Merging logic lives in pipeline/logic.py
(unit-tested). AFTER THIS STAGE: hand-review text_uk in the manifest — one
terminology fix here propagates to all target languages.
"""
from __future__ import annotations
import csv, logging, subprocess
from pathlib import Path
from . import manifest as M
from .asr import transcribe
from .logic import merge_segments, split_at_pauses

log = logging.getLogger("dubadabidu.s2")


def _initial_prompt(a: dict) -> str | None:
    """Bias the decoder toward the course's domain terms — the exact words
    otherwise fixed by hand in the manifest review. Built from the UKRAINIAN
    side of glossary/*.csv (deduped across languages) plus any free-text
    asr.initial_prompt. Whisper reads only its last ~224 tokens, so keep it
    short; deterministic, so segmentation stays reproducible."""
    parts = []
    if a.get("initial_prompt", "").strip():
        parts.append(a["initial_prompt"].strip())
    if a.get("glossary_prompt", True):
        terms: list[str] = []
        for p in sorted(Path("glossary").glob("*.csv")):
            with p.open(encoding="utf-8") as fh:
                for r in csv.reader(fh):
                    if r and len(r) >= 2 and not r[0].startswith("#"):
                        t = r[0].strip()
                        if t and t not in terms:
                            terms.append(t)
        if terms:
            parts.append("Словник уроку: " + ", ".join(terms) + ".")
    prompt = " ".join(parts)[:600]
    return prompt or None


def run(cfg: dict, video: str) -> None:
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
    prompt = _initial_prompt(a)
    if prompt:
        log.info("initial_prompt (%d chars): %s", len(prompt), prompt[:80])
    segments = transcribe(a, wd / "vocals.wav", cfg["source_language"], prompt)

    raw = []
    for s in segments:
        words = s.get("words") or []
        if words and a.get("pause_split_s"):
            raw += split_at_pauses(words, a["pause_split_s"])
        else:
            raw.append({"start": s["start"], "end": s["end"], "text": s["text"]})
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
