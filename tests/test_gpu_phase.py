"""Pure-logic tests: config overlay merge, per-lang engine routing, cache-safe
hashing for the GPU phase."""
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
    t = dict(TTS, engine_by_lang={"fr": "cosyvoice"})
    assert resolve_engine(t, "fr") == "cosyvoice"
    assert resolve_engine(t, "en") == "chatterbox"


def test_synth_hash_stable_without_new_features():
    # adding the empty engine_by_lang / reference_text keys must NOT change
    # hashes — existing seg caches stay valid
    old_style = synth_hash("hello", "en", TTS)
    new_style = synth_hash("hello", "en",
                           dict(TTS, engine_by_lang={}, reference_text=""))
    assert old_style == new_style


def test_synth_hash_changes_with_override_and_ref_text():
    base = synth_hash("hello", "fr", TTS)
    routed = synth_hash("hello", "fr", dict(TTS, engine_by_lang={"fr": "cosyvoice"},
                                            reference_text="привіт"))
    assert base != routed
    # for cosyvoice, a different reference transcript is a different output
    other_rt = synth_hash("hello", "fr", dict(TTS, engine_by_lang={"fr": "cosyvoice"},
                                              reference_text="інший текст"))
    assert routed != other_rt
    # but reference_text must NOT affect non-cosyvoice engines
    assert synth_hash("hello", "en", dict(TTS, reference_text="x")) == \
        synth_hash("hello", "en", TTS)


# --- per-engine venv isolation: discovery + worker error contract ---

def test_isolated_python_convention_and_override(tmp_path):
    from pipeline.engine_client import isolated_python
    # no venv anywhere -> in-process (the local Mac path, unchanged)
    assert isolated_python("cosyvoice", {}, root=tmp_path) is None
    # convention: venvs/<engine>/bin/python under the root -> isolated
    py = tmp_path / "venvs" / "cosyvoice" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    assert isolated_python("cosyvoice", {}, root=tmp_path) == py
    # explicit tts.engine_venvs override wins over the convention
    over = tmp_path / "elsewhere"
    (over / "bin").mkdir(parents=True)
    (over / "bin" / "python").touch()
    t = {"engine_venvs": {"cosyvoice": str(over)}}
    assert isolated_python("cosyvoice", t, root=tmp_path) == \
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
