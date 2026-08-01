"""Dub a whole course: many videos, ONE pod.

    python -m pipeline.course input/complex_oil_still_life_on_canvas
    python -m pipeline.course <dir-or-videos...> [--langs en,ru] [--dry-run]

WHY THIS AND NOT `run` IN A LOOP. Only s4 needs a GPU. `remote run` per video
provisions, bootstraps, installs qwen and downloads weights EACH TIME — roughly
5 minutes and a fresh set of downloads before the first take of every video. For
a 20-video course that is ~100 minutes of billed setup to do ~20 minutes of
work. This runs the local stages for every video first, then attaches ONE pod
for all the s4 work, then finishes locally.

    phase A  s1 separate, s2 transcribe, s3 translate   local, free
    phase B  s4 synthesize                              ONE pod, --reuse
    phase C  s5 fit, s6 mix, s7 subs, s8 mux            local, free

Phase A is also where the slow parts live and none of them are GPU work:
separation is CPU-bound and transcription is not much better, while translation
is pure API latency (measured: 32 min elapsed for 3.5 s of CPU). Doing it on a
pod would bill for waiting.

RESUMABLE by construction: every stage records completion in the manifest, so a
re-run skips what is done. A video that fails does not stop the course — the
whole point is that one bad lesson must not cost the other nineteen their pod.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v"}


def _videos(args: list[str]) -> list[str]:
    out: list[str] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            out += [str(f) for f in sorted(p.rglob("*"))
                    if f.suffix.lower() in VIDEO_EXT]
        elif p.suffix.lower() in VIDEO_EXT:
            out.append(str(p))
    return out


def _stage_done(video: str, langs: list[str], stage: str, work: str) -> bool:
    """Has `stage` finished for every language of this video?"""
    man = Path(work) / Path(video).stem / "manifest.json"
    if not man.exists():
        return False
    try:
        stages = json.loads(man.read_text(encoding="utf-8")).get("stages", {})
    except json.JSONDecodeError:
        return False
    if stage in ("s1_extract", "s2_transcribe"):     # language-independent
        return bool(stages)
    return all(stages.get(f"{stage.split('_')[0]}_{lg}") == "done"
               for lg in langs)


def _run(cmd: list[str], dry: bool) -> int:
    print("   $", " ".join(cmd[1:]))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="video files or a directory")
    ap.add_argument("--langs", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-local", action="store_true",
                    help="phase A already done elsewhere")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    langs = (a.langs.split(",") if a.langs else cfg["languages"])
    work = cfg["work_dir"]
    videos = _videos(a.targets)
    if not videos:
        print("no videos found in", a.targets)
        return 1

    py = sys.executable
    base = [py, "dub.py"]
    gpu = ["--overlay", "config.gpu.yaml", "--overlay", "config.deepseek.yaml"]
    print(f"\n{len(videos)} video(s), langs={','.join(langs)}\n")
    status: dict[str, dict[str, str]] = {v: {} for v in videos}
    t0 = time.time()

    # ---- phase A: everything that does not need a GPU ------------------
    if not a.skip_local:
        for v in videos:
            print(f"\n[A] {Path(v).name}")
            for stage, extra in (("s2_transcribe", []),
                                 ("s3_translate", ["--overlay",
                                                   "config.deepseek.yaml"])):
                if _stage_done(v, langs, stage, work):
                    print(f"   {stage}: cached")
                    status[v][stage] = "cached"
                    continue
                frm = "s1_extract" if stage == "s2_transcribe" else stage
                rc = _run(base + ["run", v, "--langs", ",".join(langs),
                                  "--from", frm, "--to", stage] + extra, a.dry_run)
                status[v][stage] = "ok" if rc == 0 else f"FAILED rc={rc}"
                if rc != 0:
                    break     # no point translating what was never transcribed

    # ---- phase B: ONE pod for every video's s4 -------------------------
    need_gpu = [v for v in videos
                if not _stage_done(v, langs, "s4_synthesize", work)
                and not str(status[v].get("s3_translate", "")).startswith("FAILED")]
    if not need_gpu:
        print("\n[B] every video already synthesized — no pod needed")
    else:
        print(f"\n[B] {len(need_gpu)} video(s) need the GPU — one pod, --reuse")
        try:
            for i, v in enumerate(need_gpu):
                print(f"\n[B {i+1}/{len(need_gpu)}] {Path(v).name}")
                rc = _run(base + ["remote", "run", v, "--langs", ",".join(langs),
                                  "--from", "s4_synthesize", "--to",
                                  "s4_synthesize", "--reuse", "--keep-alive"]
                          + gpu, a.dry_run)
                status[v]["s4_synthesize"] = "ok" if rc == 0 else f"FAILED rc={rc}"
        finally:
            # ALWAYS, on any exit path. --keep-alive left it up on purpose so
            # the next video could attach; nothing else will take it down.
            if not a.dry_run:
                print("\n[B] terminating the pod")
                if _run(base + ["remote", "kill", "--overlay",
                                "config.gpu.yaml"], False) != 0:
                    print("   !! remote kill FAILED — check for a leaked pod: "
                          "dubadabidu remote status")

    # ---- phase C: fit, mix, subtitles, mux — all local -----------------
    for v in videos:
        if str(status[v].get("s4_synthesize", "ok")).startswith("FAILED"):
            print(f"\n[C] {Path(v).name}: skipped (s4 failed)")
            continue
        print(f"\n[C] {Path(v).name}")
        rc = _run(base + ["run", v, "--langs", ",".join(langs),
                          "--from", "s5_fit"] + gpu, a.dry_run)
        status[v]["s5_s8"] = "ok" if rc == 0 else f"FAILED rc={rc}"

    # ---- report --------------------------------------------------------
    print(f"\n{'='*72}\ncourse summary ({(time.time()-t0)/60:.1f} min)")
    cols = ["s2_transcribe", "s3_translate", "s4_synthesize", "s5_s8"]
    print(f"{'video':40s} " + " ".join(f"{c.split('_')[0]:>7s}" for c in cols))
    bad = 0
    for v in videos:
        row = " ".join(f"{str(status[v].get(c,'-'))[:7]:>7s}" for c in cols)
        print(f"{Path(v).name[:40]:40s} {row}")
        bad += any(str(status[v].get(c, "")).startswith("FAILED") for c in cols)
    out = Path(cfg["output_dir"])
    made = [p.name for p in out.glob("*_multi.mp4")] if out.exists() else []
    print(f"\nmuxed outputs in {out}: {len(made)}")
    if bad:
        print(f"{bad} video(s) had a failing phase — see above")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
