"""s8: one MP4 = original video (stream copy) + original UA audio + N dubbed
audio tracks + N+1 mov_text subtitle tracks, all language-tagged.

Note: per-language .m4a files stay in work/<video>/ — keep them for YouTube's
multi-language audio track upload, which wants separate audio files.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from . import manifest as M


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    tags = cfg["mux"]["lang_tags"]
    src = cfg["source_language"]
    out = Path(cfg["output_dir"]) / f"{Path(video).stem}_multi.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", video]                       # 0: video + ua audio
    for lang in langs:
        cmd += ["-i", str(wd / f"dub_{lang}.m4a")]            # 1..N audio
    sub_langs = [src] + langs
    for lang in sub_langs:
        cmd += ["-i", str(wd / f"subs_{lang}.srt")]           # N+1.. subs

    cmd += ["-map", "0:v", "-map", "0:a"]                     # original UA track first
    for i in range(len(langs)):
        cmd += ["-map", f"{1 + i}:a"]
    for i in range(len(sub_langs)):
        cmd += ["-map", f"{1 + len(langs) + i}:s"]

    cmd += ["-c:v", "copy", "-c:a", "copy",
            "-c:s", cfg["mux"]["subtitle_codec"],
            "-metadata:s:a:0", f"language={tags[src]}",
            "-disposition:a:0", "default"]
    for i, lang in enumerate(langs, start=1):
        cmd += [f"-metadata:s:a:{i}", f"language={tags[lang]}"]
    for i, lang in enumerate(sub_langs):
        cmd += [f"-metadata:s:s:{i}", f"language={tags[lang]}"]

    cmd += [str(out)]
    subprocess.run(cmd, check=True)
    print(f"[s8] {out}")
