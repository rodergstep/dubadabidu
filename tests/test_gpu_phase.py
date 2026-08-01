"""Pure-logic tests: config overlay merge, per-lang engine routing, cache-safe
hashing for the GPU phase."""
import time
from pathlib import Path

import pytest

from pipeline.logic import deep_merge
from pipeline.manifest import resolve_engine, synth_hash

TTS = {"engine": "chatterbox", "reference_wav": "ref/r.wav",
       "cfg_weight": 0.0, "exaggeration": 0.55}


def test_deep_merge_nested_and_replace():
    base = {"tts": {"engine": "chatterbox", "best_of": 3}, "languages": ["en", "ru"]}
    over = {"tts": {"best_of": 5}, "languages": ["fr"]}
    out = deep_merge(base, over)
    assert out["tts"] == {"engine": "chatterbox", "best_of": 5}
    assert out["languages"] == ["fr"]          # lists replace, not merge
    assert base["tts"]["best_of"] == 3         # base untouched


def test_resolve_engine_default_and_override():
    assert resolve_engine(TTS, "en") == "chatterbox"
    t = dict(TTS, engine_by_lang={"fr": "qwen"})
    assert resolve_engine(t, "fr") == "qwen"
    assert resolve_engine(t, "en") == "chatterbox"


def test_synth_hash_stable_without_new_features():
    # adding the empty engine_by_lang / reference_text keys must NOT change
    # hashes — existing seg caches stay valid
    old_style = synth_hash("hello", "en", TTS)
    new_style = synth_hash("hello", "en",
                           dict(TTS, engine_by_lang={}, reference_text=""))
    assert old_style == new_style


def test_synth_hash_changes_with_engine_override():
    base = synth_hash("hello", "fr", TTS)
    routed = synth_hash("hello", "fr", dict(TTS, engine_by_lang={"fr": "qwen"}))
    assert base != routed
    # reference_text reaches no surviving engine's key (the UA ref cannot be
    # tokenized by any of them), so it must not perturb a hash
    assert synth_hash("hello", "en", dict(TTS, reference_text="x")) == \
        synth_hash("hello", "en", TTS)


# --- per-engine venv isolation: discovery + worker error contract ---

def test_isolated_python_convention_and_override(tmp_path):
    from pipeline.engine_client import isolated_python
    # no venv anywhere -> in-process (the local Mac path, unchanged)
    assert isolated_python("qwen", {}, root=tmp_path) is None
    # convention: venvs/<engine>/bin/python under the root -> isolated
    py = tmp_path / "venvs" / "qwen" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    assert isolated_python("qwen", {}, root=tmp_path) == py
    # explicit tts.engine_venvs override wins over the convention
    over = tmp_path / "elsewhere"
    (over / "bin").mkdir(parents=True)
    (over / "bin" / "python").touch()
    t = {"engine_venvs": {"qwen": str(over)}}
    assert isolated_python("qwen", t, root=tmp_path) == \
        over / "bin" / "python"


def test_isolated_python_configured_but_missing_raises(tmp_path):
    # an explicit mapping to a missing venv must raise the actionable
    # FileNotFoundError (-> engine marked unavailable), not fall through
    import pytest
    from pipeline.engine_client import isolated_python
    t = {"engine_venvs": {"qwen": str(tmp_path / "nope")}}
    with pytest.raises(FileNotFoundError):
        isolated_python("qwen", t, root=tmp_path)


def test_worker_handle_error_contract(tmp_path):
    # handle() maps exceptions onto the cross-process synthesize() contract:
    # FileNotFoundError -> "notfound" (unavailable, no retry), else "error"
    from pipeline.engine_worker import handle
    req = {"text": "hi", "lang": "en", "out": str(tmp_path / "o.wav"), "t": {}}

    def ok(text, lang, out, t):
        pass

    def notfound(text, lang, out, t):
        raise FileNotFoundError("engine not importable")

    def boom(text, lang, out, t):
        raise ValueError("cuda hiccup")

    assert handle(req, ok) == {"ok": True}
    r = handle(req, notfound)
    assert r["ok"] is False and r["kind"] == "notfound"
    assert "not importable" in r["error"]
    r = handle(req, boom)
    assert r["ok"] is False and r["kind"] == "error"
    assert "ValueError" in r["error"]


# --- synth worker pool: takes are independent, so they can run concurrently ---

def test_pool_never_hands_one_worker_to_two_threads(monkeypatch):
    """The whole safety argument: each worker is a process with its own pipes,
    so concurrency is safe ONLY if a worker is checked out by one thread at a
    time. If that breaks, two requests interleave on one stdin/stdout pair and
    the responses cross."""
    import threading
    import pipeline.engine_client as C

    holders, clash = [], []
    lock = threading.Lock()

    class FakeWorker:
        def __init__(self, engine, python):
            self.id = len(holders)
            holders.append(self.id)
            self.busy = False

        def call(self, req):
            with lock:
                if self.busy:
                    clash.append(self.id)
                self.busy = True
            time.sleep(0.01)
            self.busy = False

        def stop(self):
            pass

    monkeypatch.setattr(C, "_Worker", FakeWorker)
    pool = C._Pool("qwen", Path("py"), size=3)
    threads = [threading.Thread(target=pool.call, args=({"i": i},))
               for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not clash, f"worker(s) {clash} used by two threads at once"
    assert len(holders) <= 3, "pool spawned more workers than its size"


def test_pool_spawns_lazily_not_upfront(monkeypatch):
    """Each worker loads its own model copy (~4 GB VRAM for qwen 1.7B plus its
    own CUDA-graph capture). A pool of 3 that only ever sees serial calls must
    cost ONE model, not three."""
    import pipeline.engine_client as C
    spawned = []

    class FakeWorker:
        def __init__(self, engine, python):
            spawned.append(1)

        def call(self, req):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(C, "_Worker", FakeWorker)
    pool = C._Pool("qwen", Path("py"), size=4)
    for _ in range(5):
        pool.call({})            # strictly serial usage
    assert len(spawned) == 1


def test_failed_worker_is_returned_to_the_pool(monkeypatch):
    """_Worker.call respawns a dead process on its next use, so a crashed
    worker must stay in the pool — otherwise the pool silently shrinks to
    nothing over a long run and the last failure deadlocks it."""
    import pipeline.engine_client as C

    class Boom:
        def __init__(self, *a):
            pass

        def call(self, req):
            raise RuntimeError("worker died")

        def stop(self):
            pass

    monkeypatch.setattr(C, "_Worker", Boom)
    pool = C._Pool("qwen", Path("py"), size=1)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            pool.call({})
    assert pool._idle.qsize() == 1
