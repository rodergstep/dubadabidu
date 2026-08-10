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
    # edge (generic-voice) output validates plumbing only. Name it so it can
    # never be mistaken for a judgeable dub — and so an edge fixture run can
    # never overwrite a real cloned _multi.mp4.
    edge = M.edge_langs(man, langs)
    suffix = "_multi_EDGE-PLUMBING-ONLY.mp4" if edge else "_multi.mp4"
    out = Path(cfg["output_dir"]) / f"{Path(video).stem}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if edge:
        print(f"[s8] !! {', '.join(edge)} synthesized with the EDGE fallback "
              f"(generic voice, no cloning) — do NOT judge voice quality on "
              f"this file: {out.name}")

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
        # `-disposition:a:i 0` is NOT redundant with setting a:0 default above.
        # ffmpeg COPIES each input's disposition, and every dub_<lang>.m4a has
        # default set on its only audio stream — so all six tracks arrived
        # flagged default (verified with ffprobe on the first production mux,
        # 2026-08-02). A file with several "default" audio tracks lets the
        # player choose, which is exactly what explicit language tagging is
        # meant to prevent.
        cmd += [f"-metadata:s:a:{i}", f"language={tags[lang]}",
                f"-disposition:a:{i}", "0"]
    for i, lang in enumerate(sub_langs):
        # Same for subtitles: only the source-language track should be default,
        # otherwise players can burn in a translation nobody asked for.
        cmd += [f"-metadata:s:s:{i}", f"language={tags[lang]}",
                f"-disposition:s:{i}", "default" if i == 0 else "0"]

    cmd += [str(out)]
    subprocess.run(cmd, check=True)
    print(f"[s8] {out}")

    if cfg["mux"].get("per_language"):
        _mux_per_language(cfg, video, langs, man, wd, tags, src, edge)


def _mux_per_language(cfg: dict, video: str, langs: list[str], man: dict,
                      wd: Path, tags: dict, src: str, edge: list) -> None:
    """One MP4 per language: video + THAT dub only + its subtitles.

    The multi-track file is the better artifact — one upload, YouTube's
    multi-language audio, no duplicated video. But that feature is per-channel
    and its ingest is fussier than an ordinary upload, so this emits the
    conventional thing too: five self-contained files anyone can upload
    anywhere. Video is stream-copied in both, so the only cost is disk.

    Deliberately NOT the original UA audio: a viewer picking the Spanish file
    wants Spanish, and a second track invites the same ambiguous-default
    problem the multi file just had."""
    out_dir = Path(cfg["output_dir"])
    stem = Path(video).stem
    for lang in langs:
        suffix = ("_EDGE-PLUMBING-ONLY" if lang in edge else "")
        out = out_dir / f"{stem}_{lang}{suffix}.mp4"
        cmd = ["ffmpeg", "-y", "-i", video,
               "-i", str(wd / f"dub_{lang}.m4a"),
               "-i", str(wd / f"subs_{lang}.srt"),
               # 0:v only — drop the source UA audio, take the dub as track 0
               "-map", "0:v", "-map", "1:a", "-map", "2:s",
               "-c:v", "copy", "-c:a", "copy",
               "-c:s", cfg["mux"]["subtitle_codec"],
               "-metadata:s:a:0", f"language={tags[lang]}",
               "-disposition:a:0", "default",
               "-metadata:s:s:0", f"language={tags[lang]}",
               # NO -disposition on the subtitle track: with a SINGLE subtitle
               # stream the mp4/mov muxer marks it default regardless, and both
               # `-disposition:s:0 0` and `... none` were verified to leave
               # default:1 (2026-08-03). The multi-track file can select one
               # because it has six streams to choose between; here there is
               # nothing to choose. So a player may auto-show same-language
               # subtitles over the dub — harmless, and not worth burning the
               # subtitle track to prevent.
               str(out)]
        subprocess.run(cmd, check=True)
        print(f"[s8] {out}")
