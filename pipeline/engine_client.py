"""Client side of per-engine venv isolation (Phase C, THIRD_PARTY.md).

Why: the bake-off challengers are git-clone installs whose resolvers fight
over torch — in the old single-venv scheme a challenger could silently move
the chatterbox pin off 2.6.0 and invalidate every verdict (best-effort guards
only *detected* the break). Isolation makes the collision structurally
impossible: each challenger lives in its OWN venv with whatever torch its
resolver wants, and synthesis runs in a persistent worker subprocess using
that venv's python (pipeline/engine_worker.py). The model loads once per
worker and stays resident across takes/segments, so the per-call cost is one
pipe round-trip, not a model reload.

Discovery — an engine is isolated when either:
  - venvs/<engine>/bin/python exists under the project root (the convention
    runpod_infra._install_engines creates on the pod; create one locally by
    hand to isolate an engine on your own machine), or
  - tts.engine_venvs maps the engine to a venv dir explicitly (an override
    for venvs living elsewhere). Configured-but-missing raises the actionable
    FileNotFoundError that marks the engine unavailable, same as a missing
    package.
No entry, no venv -> None: the engine runs in-process exactly as before
(the local Mac path is untouched).

No per-call timeout: a worker's first call may legitimately spend minutes
downloading weights, and in-process synthesis has no timeout either. A hung
pod run is bounded by the budget deadline + pod-side watchdog. A worker that
DIES (OOM-kill, crash) surfaces as RuntimeError -> synthesize()'s retry calls
again and the worker is respawned with a fresh model load.
"""
from __future__ import annotations
import atexit
import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger("dubadabidu.engine_client")
ROOT = Path(__file__).resolve().parents[1]
_workers: dict[str, "_Worker"] = {}


def isolated_python(engine: str, t: dict, root: Path = ROOT) -> Path | None:
    """The engine's isolated-venv python, or None for in-process (see module
    doc for the discovery rules)."""
    venvs = t.get("engine_venvs") or {}
    if engine in venvs:
        py = Path(venvs[engine]) / "bin" / "python"
        if not py.exists():
            raise FileNotFoundError(
                f"tts.engine_venvs routes {engine!r} to {venvs[engine]} but "
                f"{py} does not exist — create the venv and install the "
                f"engine there (THIRD_PARTY.md), or drop the mapping.")
        return py
    py = root / "venvs" / engine / "bin" / "python"
    return py if py.exists() else None


class _Worker:
    def __init__(self, engine: str, python: Path):
        self.engine = engine
        self.python = python
        self._spawn()

    def _spawn(self) -> None:
        log.info("starting %s worker: %s", self.engine, self.python)
        # cwd=ROOT so `-m pipeline.engine_worker` resolves without installing
        # the project into the engine venv; stderr stays on the console so
        # engine logs/tracebacks are visible in the pod log.
        self.proc = subprocess.Popen(
            [str(self.python), "-m", "pipeline.engine_worker", self.engine],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            cwd=ROOT, text=True, bufsize=1)

    def call(self, req: dict) -> None:
        if self.proc.poll() is not None:   # died since last call — fresh start
            log.warning("%s worker gone (rc=%s); respawning",
                        self.engine, self.proc.returncode)
            self._spawn()
        try:
            self.proc.stdin.write(json.dumps(req, default=str) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"{self.engine} worker pipe broke: {e}")
        if not line:                       # EOF: worker died mid-request
            raise RuntimeError(
                f"{self.engine} worker exited (rc={self.proc.poll()}) "
                f"mid-request — see its stderr above; a retry respawns it")
        resp = json.loads(line)
        if resp.get("ok"):
            return
        if resp.get("kind") == "notfound":   # engine-unavailable contract
            raise FileNotFoundError(resp.get("error", ""))
        raise RuntimeError(f"{self.engine} worker: "
                           f"{resp.get('error', 'unknown error')}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()      # EOF -> worker's stdin loop ends
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()


def synth(engine: str, python: Path, text: str, lang: str, out: Path,
          t: dict) -> None:
    """One synthesis through the engine's worker (spawned on first use)."""
    w = _workers.get(engine)
    if w is None:
        w = _workers[engine] = _Worker(engine, python)
    w.call({"text": text, "lang": lang, "out": str(out), "t": t})


def shutdown(engine: str) -> None:
    """Stop one engine's worker, releasing its VRAM/RAM. The bake-off calls
    this between engines so challengers don't accumulate on the GPU."""
    w = _workers.pop(engine, None)
    if w:
        w.stop()


def shutdown_all() -> None:
    for engine in list(_workers):
        shutdown(engine)


atexit.register(shutdown_all)   # never leave orphan model processes behind
