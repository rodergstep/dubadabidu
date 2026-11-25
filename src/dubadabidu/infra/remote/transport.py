# SPDX-License-Identifier: AGPL-3.0-or-later
"""ssh and rsync to a rented box (provider-neutral)."""
from __future__ import annotations
import logging
import os
import subprocess
import time

from dubadabidu.core.config.schema import Runpod
from dubadabidu.infra.remote import bootstrap

log = logging.getLogger("dubadabidu.runpod")


def _ssh_base(rp: Runpod, host: str, port: int) -> list[str]:
    key = os.path.expanduser(rp.ssh_key)
    return ["ssh", "-i", key, "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", f"{rp.ssh_user}@{host}"]


def ssh_exec(rp: Runpod, host: str, port: int, cmd: str, timeout: int = 7200) -> int:
    full = _ssh_base(rp, host, port) + [cmd]
    return subprocess.run(full, timeout=timeout).returncode


def wait_ssh(rp: Runpod, host: str, port: int, deadline: float) -> bool:
    while time.time() < deadline:
        r = subprocess.run(_ssh_base(rp, host, port) + ["true"],
                           capture_output=True, timeout=30)
        if r.returncode == 0:
            return True
        time.sleep(8)
    return False


# Model caches that must NOT cross between this Mac and the pod. Patterns carry
# a slash so they match on both transfer roots this function is called with (the
# project dir when pushing up, work/ when pulling results back).
#
#   .models/ecapa — speechbrain populates its savedir with SYMLINKS into the
#     local HF cache (/Users/.../.cache/huggingface/...). rsync -a preserves
#     symlinks as symlinks, so these land on a Linux pod dangling, and ECAPA is
#     needed there for the bake-off anchor and every sim score. Excluded so the
#     pod fetches its own copy cleanly instead of inheriting broken links —
#     which would fail AFTER the expensive engine installs.
#   .models/audio-separator — 610 MB, and dead weight on the pod: separation is
#     s1, which runs locally. The pod works from the synced work/ stems and
#     never separates anything. The pod is billed while rsync runs.
#
# floor_*.wav (a few hundred KB, directly in .models/) is deliberately NOT
# excluded: it pins the per-language calibration band, so regenerating it on the
# pod would make sim_cal subtly incomparable between local and pod runs.
_MODEL_CACHE_EXCLUDES = (".models/ecapa", ".models/audio-separator")


def _verify_ready(rp: Runpod, host: str, port: int, remote: str,
                  engines: list[str]) -> bool:
    """Is this pod ALREADY set up for these engines? Proof, not assumption.

    Checks, in the venv that now holds everything:
      - .venv exists
      - each engine's module imports
      - torch.cuda.is_available() — a CPU-only torch is exactly the failure that
        once cost a week of 126 s/take runs, and it is silent without this
    Any failure, any non-zero rc, any doubt -> False, and setup runs normally.
    Skipping setup wrongly costs a whole experiment; running it needlessly costs
    ~50 s, so the asymmetry says be conservative.

    `edge` needs no install, so it neither contributes an import check nor
    disqualifies the pod — but a list containing ONLY engines with nothing to
    verify still returns False, because then this proves nothing about the venv
    beyond its existence. Unknown engine names stay disqualifying."""
    if not engines:
        return False
    py = f"{remote}/.venv/bin/python"
    checks = [f"test -x {py}"]
    for e in engines:
        if e == "edge":        # network service; no venv install to verify
            continue
        mod = bootstrap.ENGINE_MODULE.get(e)
        if not mod:            # unknown engine — cannot prove anything
            return False
        checks.append(
            f"{py} -c 'import {mod}, torch; assert torch.cuda.is_available()'")
    if len(checks) == 1:       # nothing but `test -x` — not proof enough
        return False
    rc = ssh_exec(rp, host, port, " && ".join(checks), timeout=180)
    return rc == 0


def ssh_capture(rp: Runpod, host: str, port: int, cmd: str,
                timeout: int = 60) -> str:
    """Run a remote command and return its stdout (empty on failure)."""
    try:
        r = subprocess.run(_ssh_base(rp, host, port) + [cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def free_engine(rp: Runpod, host: str, port: int, remote: str,
                engine: str, threshold_pct: int = 70) -> None:
    """Delete an engine's venv and model cache once its results are banked.

    ONLY when the disk is actually tight (>`threshold_pct` used). Per-engine
    isolation means each challenger drags in its own CUDA torch plus weights
    (7-15 GB), so unbounded accumulation would recreate the four-venv exhaustion
    that broke seven runs — but freeing EAGERLY is just as wrong when the pod is
    being reused: 60 GB holds the base, the main venv and two engines at ~39 GB,
    and re-fetching a deleted engine costs real time (IndexTTS-2's checkpoints
    alone are 5.6 GB, ~12 min). So keep engines around until space demands
    otherwise. Best-effort — a failure here costs disk, not correctness."""
    used = ssh_capture(rp, host, port,
                       "df --output=pcent / | tail -1 | tr -dc '0-9'")
    try:
        pct = int(used.strip())
    except (ValueError, AttributeError):
        pct = 100      # unknown -> assume tight and reclaim, the safe direction
    if pct < threshold_pct:
        log.info("disk %d%% used (<%d%%) — nothing reclaimed, %s stays warm "
                 "for a --reuse run", pct, threshold_pct, engine)
        return
    # Reclaim the BULK — model weights — and keep the venv and clone, which are
    # small and expensive to rebuild. The original order was backwards: it deleted
    # third_party/ (~5 GB) while keeping checkpoints/ and
    # pretrained_models/ (~16 GB), so it fired right after a big download and
    # destroyed the environment that download was for. Measured 2026-07-30: a
    # 5.4 GB CosyVoice2 fetch pushed disk to 79%, the sweep deleted the freshly
    # validated venv + clone, and the weights it could have freed stayed put.
    cmd = (f"rm -rf {remote}/checkpoints {remote}/pretrained_models "
           f"/root/.cache/huggingface /root/.cache/pip 2>/dev/null; "
           f"df -h / | tail -1")
    if ssh_exec(rp, host, port, cmd, timeout=300) == 0:
        log.info("disk %d%% used — freed model weights/caches "
                 "(venv + clone kept: small and costly to rebuild)", pct)
    else:
        log.warning("could not free model weights — disk may fill on the "
                    "next engine")


def _rsync_paths(rp: Runpod, port: int, paths: list[str], dst: str) -> None:
    """Send an explicit list of project-relative paths, preserving their layout
    (-R). Used by the bake-off to ship the 3.5 MB it actually reads instead of
    the whole work/ tree."""
    if not paths:
        return
    key = os.path.expanduser(rp.ssh_key)
    args = (["rsync", "-azR"]
            + ["-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=no "
                     f"-o UserKnownHostsFile=/dev/null"]
            + paths + [dst])
    subprocess.run(args, check=True)


def rsync(rp: Runpod, port: int, src: str, dst: str, check: bool = True,
          extra_excludes: tuple[str, ...] = ()) -> None:
    """rsync over SSH on the pod's MAPPED port (RunPod exposes 22 on a random
    public port — omitting -p hits default 22 and fails auth)."""
    key = os.path.expanduser(rp.ssh_key)
    args = ["rsync", "-az"]
    for e in (".venv", "venvs", "__pycache__", ".env", ".git",
              *_MODEL_CACHE_EXCLUDES, *extra_excludes):
        args += ["--exclude", e]
    args += ["-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=no "
                   f"-o UserKnownHostsFile=/dev/null", src, dst]
    subprocess.run(args, check=check)
