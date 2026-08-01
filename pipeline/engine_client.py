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
import queue
import threading
import logging
import subprocess
from pathlib import Path

log = logging.getLogger("dubadabidu.engine_client")
ROOT = Path(__file__).resolve().parents[1]
_workers: dict[str, "_Pool"] = {}


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


class _Pool:
    """N workers for one engine, checked out one thread at a time.

    Takes are INDEPENDENT — best_of rolls the same text N times sharing no
    state — but they were generated strictly one after another, so a 1.7B model
    decoding a single sequence left the GPU at ~7% utilisation while synthesis
    still accounted for 60% of wall clock. Each worker is its own process with
    its own pipes, so concurrency needs nothing more than making sure two
    threads never share one worker; the queue is that guarantee.

    Workers spawn LAZILY. Each one loads its own copy of the model (~4 GB for
    qwen 1.7B, plus its own CUDA-graph capture), so a pool sized past the card
    turns a speedup into an OOM mid-run. Spawning on demand means a pool of 3
    that only ever sees serial calls costs one model, not three."""

    def __init__(self, engine: str, python: Path, size: int):
        self.engine, self.python, self.size = engine, python, max(1, size)
        self._idle: queue.Queue = queue.Queue()
        self._spawned = 0
        self._lock = threading.Lock()

    def _checkout(self) -> _Worker:
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._spawned < self.size:
                self._spawned += 1
                return _Worker(self.engine, self.python)
        return self._idle.get()          # all spawned and busy — wait for one

    def call(self, req: dict) -> None:
        w = self._checkout()
        try:
            w.call(req)
        finally:
            # returned even on failure: _Worker.call respawns a dead process on
            # its next use, so a crashed worker must stay in the pool or the
            # pool silently shrinks to nothing over a long run.
            self._idle.put(w)

    def stop(self) -> None:
        while True:
            try:
                self._idle.get_nowait().stop()
            except queue.Empty:
                return


def synth(engine: str, python: Path, text: str, lang: str, out: Path,
          t: dict, workers: int = 1) -> None:
    """One synthesis through the engine's worker pool (spawned on first use)."""
    pool = _workers.get(engine)
    if pool is None:
        pool = _workers[engine] = _Pool(engine, python, workers)
    pool.call({"text": text, "lang": lang, "out": str(out), "t": t})


def shutdown(engine: str) -> None:
    """Stop one engine's workers, releasing their VRAM/RAM. The bake-off calls
    this between engines so challengers don't accumulate on the GPU."""
    pool = _workers.pop(engine, None)
    if pool:
        pool.stop()


def shutdown_all() -> None:
    for engine in list(_workers):
        shutdown(engine)


atexit.register(shutdown_all)   # never leave orphan model processes behind
