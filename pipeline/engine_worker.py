"""Worker side of per-engine venv isolation (Phase C, THIRD_PARTY.md).

Executed as `venvs/<engine>/bin/python -m pipeline.engine_worker <engine>`
with cwd = project root: the `-m` + cwd combination puts `pipeline` on
sys.path, so the engine venv needs NOTHING of the project installed — only
the engine's own dependencies (tts_engine's module-level imports are
stdlib-only by design; torch/soundfile load lazily inside the adapters).

Protocol: one JSON request per stdin line -> one JSON reply per stdout line.
  request  {"text": str, "lang": str, "out": str, "t": {tts cfg}}
  reply    {"ok": true} | {"ok": false, "kind": "notfound"|"error", "error": str}
`kind` preserves the engine-unavailable contract across the process boundary:
the client re-raises "notfound" as FileNotFoundError (synthesize() passes it
through un-retried; the bake-off marks the engine unavailable) and "error" as
RuntimeError (retried like any in-process synthesis glitch).

stdout is RESERVED for protocol replies: engine libraries love printing
progress to stdout, so fd 1 is duplicated for the protocol and then pointed
at stderr BEFORE any engine code can load. Tracebacks/logs stream to stderr,
which the parent leaves attached to the console — pod debugging stays easy.
"""
from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path


def handle(req: dict, fn) -> dict:
    """One request -> one reply dict. Split out for testability; the error
    mapping here IS the cross-process synthesize() contract (see module doc)."""
    try:
        fn(req["text"], req["lang"], Path(req["out"]), req["t"])
        return {"ok": True}
    except FileNotFoundError as e:          # engine unavailable / missing ref
        return {"ok": False, "kind": "notfound", "error": str(e)}
    except Exception as e:                  # a glitch — the client retries
        traceback.print_exc()               # full trace to stderr for the log
        return {"ok": False, "kind": "error",
                "error": f"{type(e).__name__}: {e}"}


def serve(engine: str) -> None:
    proto = os.fdopen(os.dup(1), "w", buffering=1)   # protocol replies only
    os.dup2(2, 1)                # fd-level prints (C extensions) -> stderr
    sys.stdout = sys.stderr      # python-level prints -> stderr
    from pipeline import tts_engine as T
    fn = {"qwen": T._synth_qwen, "edge": T._synth_edge}[engine]
    for line in sys.stdin:
        if not line.strip():
            continue
        proto.write(json.dumps(handle(json.loads(line), fn)) + "\n")


if __name__ == "__main__":
    serve(sys.argv[1])
