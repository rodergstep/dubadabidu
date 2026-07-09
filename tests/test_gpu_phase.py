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
