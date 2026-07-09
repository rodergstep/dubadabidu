"""prep: per-video preamble — extract reference-clip candidates from the
video's OWN clean vocals (the echo-free ref source validated on sketch60).

Requires s1 (demucs stem) + s2 (utterance timings). Picks the best-sized
utterance spans, cuts them from the 44.1 kHz demucs vocals stem into ref/,
and writes work/<video>/refs.json mapping each ref file to its transcript
(CosyVoice zero-shot cloning needs the transcript; tune reads it too).

After prep: `dubadabidu tune <video> --langs en` picks the winning ref (R1),
then skim text_uk / terms_*.json and run the pipeline.
"""
from __future__ import annotations
import json
import logging
import subprocess
from pathlib import Path
from . import manifest as M

log = logging.getLogger("dubadabidu.prep")

MIN_S, MAX_S, N_REFS = 12.0, 20.0, 3


def run(cfg: dict, video: str) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    stem = Path(video).stem

    src = wd / "demucs" / cfg["separation"]["demucs_model"] / "audio_full" / "vocals.wav"
    if not src.exists():
        src = wd / "audio_full.wav"
        log.warning("no demucs vocals stem — cutting refs from %s "
                    "(fine only if the source has no music/noise)", src.name)

    spans = [u for u in man["utterances"] if MIN_S <= u["end"] - u["start"] <= MAX_S]
    spans.sort(key=lambda u: u["end"] - u["start"], reverse=True)
    if not spans:  # fall back to the longest available utterances
        spans = sorted(man["utterances"],
                       key=lambda u: u["end"] - u["start"], reverse=True)
        log.warning("no %d-%ds utterances; using the longest available",
                    int(MIN_S), int(MAX_S))

    refs = {}
    for i, u in enumerate(spans[:N_REFS], 1):
        out = Path("ref") / f"{stem}_ref_{i:02d}.wav"
        dur = min(u["end"] - u["start"], MAX_S)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ss", str(u["start"]), "-t", str(dur),
                        str(out)], check=True)
        refs[out.name] = {"text_uk": u["text_uk"],
                          "start": u["start"], "end": round(u["start"] + dur, 2)}
        print(f"[prep] {out}  ({dur:.1f}s)  {u['text_uk'][:60]}")

    rp = wd / "refs.json"
    rp.write_text(json.dumps(refs, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"[prep] transcripts -> {rp}\n[prep] next: dubadabidu tune {video} "
          f"--langs en   (tune.refs_glob may need ref/{stem}_ref_*.wav)")
