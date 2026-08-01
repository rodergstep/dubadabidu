"""Run a queue of experiments against ONE pod, then terminate it.

    python -m pipeline.experiments [--dry-run] [--only NAME ...] [--file F]

Every question used to cost its own pod: provision, apt, bootstrap, engine
install, weights — 5-10 min and ~$0.10 before a single measurement, paid again
for the next question. That did not just waste money, it decided which
questions got asked: anything that looked small was not "worth a pod", so it
went unmeasured and got answered by inference instead.

Each experiment runs `remote bakeoff` with --reuse --keep-alive, so the first
one provisions and the rest attach to that same pod (remote_run falls back to
provisioning when --reuse finds nothing alive, so the first is not special).
The pod is terminated in a finally block whatever happens, and the termination
is VERIFIED rather than assumed — a leaked pod bills until the watchdog fires.

A failing experiment does not stop the queue: the whole point is to get every
answer from one setup, and one bad overlay must not cost the others.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_OVERLAY = "config.gpu.yaml"


def _cmd(video: str, langs: list[str], overlays: list[str]) -> list[str]:
    cmd = [sys.executable, "dub.py", "remote", "bakeoff", video,
           "--langs", ",".join(langs), "--reuse", "--keep-alive",
           "--overlay", BASE_OVERLAY]
    for ov in overlays:
        cmd += ["--overlay", ov]
    return cmd


def _kill() -> None:
    """Terminate and VERIFY. `remote kill` already retries and confirms, but a
    non-zero rc here is the difference between a $0.7/h pod stopping and it
    running until the watchdog deadline, so it is surfaced loudly."""
    rc = subprocess.run(
        [sys.executable, "dub.py", "remote", "kill", "--overlay", BASE_OVERLAY],
        cwd=ROOT).returncode
    if rc == 0:
        print("\n[experiments] pod terminated")
    else:
        print(f"\n[experiments] !! remote kill exited {rc} — CHECK FOR A LEAKED "
              f"POD: dubadabidu remote kill --overlay {BASE_OVERLAY}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="experiments.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these experiment names")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = yaml.safe_load((ROOT / a.file).read_text(encoding="utf-8"))
    video = spec["video"]
    budget_min = float(spec.get("max_total_minutes", 60))

    queue = []
    for e in spec["experiments"]:
        if a.only:
            if e["name"] not in a.only:
                continue
        elif not e.get("enabled"):
            if e.get("blocked_by"):
                print(f"[skip] {e['name']}: {' '.join(e['blocked_by'].split())}")
            continue
        if e.get("blocked_by") and not a.only:
            continue
        queue.append(e)

    if not queue:
        print("nothing to run")
        return 0

    print(f"\n{len(queue)} experiment(s), one pod, {budget_min:.0f} min cap:")
    for e in queue:
        print(f"  - {e['name']:16s} langs={','.join(e['langs'])} "
              f"overlays={e.get('overlays') or ['(base only)']}")
        print(f"      asks: {' '.join(e['asks'].split())}")
    print()

    if a.dry_run:
        for e in queue:
            print(" ".join(_cmd(video, e["langs"], e.get("overlays") or [])))
        return 0

    started = time.time()
    results = []
    try:
        for e in queue:
            elapsed_min = (time.time() - started) / 60
            if elapsed_min > budget_min:
                print(f"[experiments] {elapsed_min:.0f} min elapsed > "
                      f"{budget_min:.0f} min cap — not starting {e['name']}")
                results.append((e["name"], "SKIPPED (time cap)", 0.0))
                continue
            print(f"\n{'='*70}\n[experiments] {e['name']}  "
                  f"({elapsed_min:.0f}/{budget_min:.0f} min used)\n{'='*70}")
            t0 = time.time()
            rc = subprocess.run(_cmd(video, e["langs"],
                                     e.get("overlays") or []), cwd=ROOT).returncode
            dt = (time.time() - t0) / 60
            # keep going on failure — one bad overlay must not cost the others
            results.append((e["name"], "ok" if rc == 0 else f"FAILED rc={rc}", dt))
    finally:
        _kill()

    print(f"\n{'='*70}\nsummary ({(time.time()-started)/60:.1f} min total)")
    for name, status, dt in results:
        print(f"  {name:16s} {status:16s} {dt:5.1f} min")
    for e in queue:
        if e.get("reads"):
            print(f"\n[{e['name']}] read: {' '.join(e['reads'].split())}")
    return 0 if all(s == "ok" for _, s, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
