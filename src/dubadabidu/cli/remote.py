# SPDX-License-Identifier: AGPL-3.0-or-later
"""How `remote run` splits a stage range between this machine and the pod."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from dubadabidu.cli.stages import ORDER, STAGES
from dubadabidu.core import manifest as M


POD_FIRST, POD_LAST = "s4_synthesize", "s5_fit"


def _remote_plan(a) -> tuple[list[str], list[str], list[str]]:
    """Split `remote run`'s --from/--to range into (local before, POD, local
    after). Only s4 needs a GPU (s5 rides along: it is cheap and its input is
    on the pod), so the pod gets at most s4..s5 and the laptop does the rest:

      before: s3 translation — API-latency-bound, so on the pod it billed GPU
              time while waiting on DeepSeek. Untouched --from starts HERE,
              not at s1: s2 has no cache and always re-transcribes, which
              discards every translation and synth state (2026-11-24: the
              pod re-ran s2 on a cached video and crashed in faster-whisper).
              An explicit --from s1/s2 still runs them, locally, where the
              source video lives.
      after:  s6 mix (its mastering chain needs the [master] extra no pod
              installs — a full run died there 2026-08-09), s7 subtitles and
              s8 mux (needs the source video, which never uploads).
    """
    frm = getattr(a, "from_stage", "s1_extract")
    to = getattr(a, "to_stage", "s8_mux")
    if frm == "s1_extract":
        frm = "s3_translate"
    i, j = ORDER.index(frm), ORDER.index(to)
    rng = ORDER[i:j + 1]
    p0, p1 = ORDER.index(POD_FIRST), ORDER.index(POD_LAST)
    before = [s for s in rng if ORDER.index(s) < p0]
    pod = [s for s in rng if p0 <= ORDER.index(s) <= p1]
    after = [s for s in rng if ORDER.index(s) > p1]
    return before, pod, after


def _remote_stages(a) -> str:
    """--from/--to for the POD-side command: the pod slice of _remote_plan.
    Without this the pod ran the whole pipeline every time."""
    # dest names are from_stage / to_stage — NOT "from"/"to". Reading the wrong
    # attribute returned None for both, so the pod silently fell back to the
    # whole pipeline, ran s1/s2, and died on the input video that is
    # deliberately never uploaded. It cost a provisioned pod to find out.
    _before, pod, _after = _remote_plan(a)
    if not pod:
        return ""
    return f"--from {pod[0]} --to {pod[-1]}"


def _pod_stages_done(cfg: dict, video: str, langs: list[str],
                     stages: list[str]) -> bool:
    """Did every pod stage finish for every language? Read from the manifest
    the pod synced back (stage keys are s4_<lang>, s5_<lang>)."""
    p = M.manifest_path(cfg, video)
    if not p.exists():
        return False
    import json
    done = json.loads(p.read_text(encoding="utf-8")).get("stages", {})
    return all(done.get(f"{st.split('_')[0]}_{lg}") == "done"
               for st in stages for lg in langs)


def _remote_run_batch(cfg: dict, vids: list[str], langs: list[str], a,
                      rpi: ModuleType) -> bool:
    """`remote run` over one or many videos: ONE pod, ONE pod-side process.

      1. local `before` stages (translation) for EVERY video first, so the pod
         never bills while DeepSeek answers;
      2. one remote_run with all videos: setup is paid once (4.5 min vs 1.5
         min of synthesis per 1-min lesson, 2026-11-24) and the models load
         once (~30 s per process). The pod runs `run --keep-going`, so one bad
         video does not stop the others;
      3. local `after` stages (mix, subtitles, mux) for every video whose pod
         stages are recorded done in its synced-back manifest.
    """
    before, pod, after = _remote_plan(a)
    for v in vids:
        _run_local(cfg, v, langs, before)
    if pod:
        rpi.remote_run(cfg, vids, langs, "run", a.budget,
                       keep_alive=bool(a.keep_alive), reuse=bool(a.reuse),
                       overlays=a.overlay or [],
                       stages=f"--from {pod[0]} --to {pod[-1]}")
        done = [v for v in vids if _pod_stages_done(cfg, v, langs, pod)]
        for v in vids:
            if v not in done:
                logging.error("%s: GPU stages did not finish — skipping its "
                              "local finish", Path(v).name)
    else:
        done = list(vids)
    for v in done:
        _run_local(cfg, v, langs, after)
    return len(done) == len(vids)


def _run_local(cfg: dict, video: str, langs: list[str],
               stages: list[str]) -> None:
    for name in stages:
        logging.info("[local] %s", name)
        try:
            STAGES[name](cfg, video, langs)
        except subprocess.CalledProcessError as e:
            sys.exit(f"[{name}] external command failed: {e}")
