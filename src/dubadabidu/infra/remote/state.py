# SPDX-License-Identifier: AGPL-3.0-or-later
"""The state-file protocol: what must be killed is what the state file
says, not a return value (see orchestrator's SAFETY notes)."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from dubadabidu.infra.remote import provider

log = logging.getLogger("dubadabidu.runpod")


STATE_FILE = Path("work/.runpod_active.json")


def terminate_pod(pid: str, retries: int = 4) -> bool:
    """DELETE the pod and CONFIRM it's gone. Returns True only when termination
    is verified (or the pod is already absent). This is the one call that must
    not fail silently — a dropped DELETE leaks a billing pod — so it retries and
    then GETs the pod to check it actually left the RUNNING state."""
    for attempt in range(retries):
        try:
            provider.active().destroy(pid)
        except RuntimeError as e:
            if "HTTP 404" in str(e):            # already gone
                log.info("pod %s already absent", pid)
                return True
            log.warning("terminate %s attempt %d/%d failed: %s",
                        pid, attempt + 1, retries, e)
            time.sleep(3)
            continue
        # DELETE returned OK — verify the pod really left RUNNING
        try:
            st = provider.active().status(provider.active().get(pid))
        except RuntimeError as e:
            if "HTTP 404" in str(e):            # confirmed gone
                log.info("terminated pod %s (confirmed)", pid)
                return True
            st = "?"
        if st in ("TERMINATED", "EXITED", "", "?"):
            log.info("terminated pod %s", pid)
            return True
        log.warning("pod %s still %s after DELETE; retrying", pid, st)
        time.sleep(3)
    return False


# ---------------- state file (leak protection) ----------------

def _save_state(d: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _held_pod() -> str | None:
    """Pod id the state file marks as held for a hotfix, else None."""
    try:
        d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("pod_id") if d.get("held_for_hotfix") else None


def _clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def _terminate_tracked() -> None:
    """THE single cleanup path: terminate + clear whatever pod is recorded in
    the state file. Idempotent and crash-safe — because the pod id is persisted
    on create, this cleans up even when provisioning dies mid-flight (the local
    pid variable is useless then). Every finally block and orphan-sweep calls
    this, so termination never depends on a return value."""
    if not STATE_FILE.exists():
        return
    try:
        pid = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("pod_id")
    except (OSError, json.JSONDecodeError):
        _clear_state(); return
    if not pid:
        _clear_state(); return
    # Clear the state file ONLY when termination is confirmed. If it failed,
    # KEEP the id so the next `remote` call (or `remote kill`) retries — clearing
    # it here would erase the only record of a still-running billing pod.
    if terminate_pod(pid):
        _clear_state()
    else:
        log.error("could NOT confirm termination of %s — state file KEPT for "
                  "retry. Run `dubadabidu remote kill` or delete it in the "
                  "RunPod console.", pid)


def sweep_orphans() -> None:
    """Clean up a pod left in the state file by a prior run. Called at the start
    of every `remote` command and exposed as `remote kill`."""
    if STATE_FILE.exists():
        log.warning("state file present — terminating tracked pod before start")
    _terminate_tracked()


def _live_pod() -> tuple[str, str, int] | None:
    """(pod_id, host, port) for the pod in the state file when it is still up and
    reachable, else None. Used by --reuse to attach to a pod a previous
    --keep-alive run left running, instead of paying another bootstrap.

    Deliberately conservative: any doubt (no state file, API error, not RUNNING,
    no SSH mapping yet) returns None and the caller provisions fresh. Reusing a
    half-dead pod would be worse than paying for a new one."""
    if not STATE_FILE.exists():
        return None
    try:
        pid = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("pod_id")
        if not pid:
            return None
        p = provider.active().get(pid)
        status = provider.active().status(p)
        if status and status != "RUNNING":
            log.info("state-file pod %s is %s, not RUNNING", pid, status)
            return None
        target = provider.active().ssh_target(p)
        if not target:
            return None
        return pid, target[0], target[1]
    except (RuntimeError, json.JSONDecodeError, OSError) as e:
        log.info("could not reuse the state-file pod (%s)", e)
        return None
