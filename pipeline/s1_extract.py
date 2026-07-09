"""s1: ffmpeg audio extraction + Demucs (htdemucs) vocal/background separation."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from .manifest import video_workdir


def sh(*args: str) -> None:
    subprocess.run(args, check=True)


def run(cfg: dict, video: str) -> None:
    wd = video_workdir(cfg, video)
    full = wd / "audio_full.wav"
    if not full.exists():
        sh("ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "44100", str(full))

    vocals = wd / "vocals.wav"
    bg = wd / "background.wav"
    if vocals.exists() and bg.exists():
        print(f"[s1] cached: {wd}")
        return

    if cfg["separation"]["enabled"]:
        model = cfg["separation"]["demucs_model"]
        sh(sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", model,
           "-o", str(wd / "demucs"), str(full))
        stem_dir = wd / "demucs" / model / full.stem
        sh("ffmpeg", "-y", "-i", str(stem_dir / "vocals.wav"),
           "-ac", "1", "-ar", "16000", str(vocals))          # 16k mono for ASR
        sh("ffmpeg", "-y", "-i", str(stem_dir / "no_vocals.wav"), str(bg))
    else:
        # no music in source: light denoise instead of separation
        sh("ffmpeg", "-y", "-i", str(full), "-af", "afftdn=nf=-25",
           "-ac", "1", "-ar", "16000", str(vocals))
        sh("ffmpeg", "-y", "-f", "lavfi", "-t", "1", "-i",
           "anullsrc=r=44100:cl=mono", str(bg))               # silent bg placeholder
    print(f"[s1] done: {vocals}, {bg}")
