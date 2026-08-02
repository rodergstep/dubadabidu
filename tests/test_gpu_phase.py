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









def test_gpu_profile_actually_routes_production_to_qwen():
    """bakeoff.engines drives only the BAKE-OFF's loop. A real `run` resolves
    through tts.engine, which config.gpu.yaml never set — so it inherited
    chatterbox from config.yaml and the first production run was minutes from
    synthesizing five languages with the wrong engine. Silently, since
    chatterbox installs in the base venv: wrong output, not a crash."""
    import yaml
    from pathlib import Path
    from pipeline.logic import deep_merge
    root = Path(__file__).resolve().parents[1]
    cfg = deep_merge(yaml.safe_load((root / "config.yaml").read_text()),
                     yaml.safe_load((root / "config.gpu.yaml").read_text()))
    t = cfg["tts"]
    for lang in cfg["languages"]:
        assert resolve_engine(t, lang) == "qwen", (
            f"{lang} resolves to {resolve_engine(t, lang)}, not the engine the "
            f"bake-off selected — tts.engine and bakeoff.engines disagree")


def test_no_isolation_machinery_survives():
    """Per-engine venvs existed because four engines had colliding pins. Three
    are gone and chatterbox took its torch pin with it, so the isolation cost a
    SECOND ~2.5 GB torch download per pod and bought nothing. Removed
    2026-08-02 — this fails if it creeps back without the engines that
    justified it."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert not (root / "pipeline" / "engine_client.py").exists()
    assert not (root / "pipeline" / "engine_worker.py").exists()
    src = (root / "pipeline" / "tts_engine.py").read_text()
    assert "isolated_python" not in src and "engine_client" not in src
