"""PLAN_v0.5 P4.5: per-language qwen checkpoint, the model-keyed loader, and
the guard against stress marks on a checkpoint not trained on them."""
import pytest

from pipeline import manifest as M
from pipeline import tts_engine as T

FT = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"


def _cfg(**kw: object) -> dict:
    return {"engine": "qwen", "engine_by_lang": {}, "reference_wav": "r.wav", **kw}


def test_per_language_checkpoint_resolves_and_falls_back() -> None:
    t = _cfg(qwen_model_dir_by_lang={"ru": FT})
    assert M.resolve_qwen_model(t, "ru") == FT
    assert M.resolve_qwen_model(t, "en") == M.DEFAULT_QWEN_MODEL


def test_synth_hash_keys_on_the_resolved_per_language_checkpoint() -> None:
    plain, routed = _cfg(), _cfg(qwen_model_dir_by_lang={"ru": FT})
    assert M.synth_hash("т", "ru", plain) != M.synth_hash("т", "ru", routed)
    # en is untouched by a ru-only route: its cache stays valid
    assert M.synth_hash("t", "en", plain) == M.synth_hash("t", "en", routed)


def test_loader_reloads_when_the_checkpoint_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []

    class Fake:
        @classmethod
        def from_pretrained(cls, model_id: str, **_: object) -> "Fake":
            loaded.append(model_id)
            return cls()

    import sys
    import types
    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=Fake))
    # CI installs no torch (by design); the loader only needs the dtype names
    fake_torch = types.SimpleNamespace(bfloat16="bf16", float32="fp32",
                                       cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(T, "require_gpu", lambda *_: "cpu")
    T.release_models()
    t = _cfg(qwen_model_dir_by_lang={"ru": FT}, qwen_fast=False)
    T._load_qwen(t, "en")
    T._load_qwen(t, "en")
    T._load_qwen(t, "ru")
    assert loaded == [M.DEFAULT_QWEN_MODEL, FT], "en loads once, ru triggers a reload"
    T.release_models()


def test_marks_on_the_base_model_are_refused() -> None:
    with pytest.raises(RuntimeError, match="not trained"):
        T._refuse_marks_on_untrained_model(_cfg(ru_stress=True), "ru", "qwen")


def test_marks_on_a_stress_trained_checkpoint_or_forced_are_allowed() -> None:
    T._refuse_marks_on_untrained_model(_cfg(ru_stress=True, qwen_model_dir=FT), "ru", "qwen")
    T._refuse_marks_on_untrained_model(_cfg(ru_stress=True, ru_stress_force=True), "ru", "qwen")
    T._refuse_marks_on_untrained_model(_cfg(ru_stress=False), "ru", "qwen")


def test_sample_rate_guard_rejects_a_mismatch(tmp_path):
    """faster-qwen3-tts assumes 24000 Hz; a checkpoint declaring anything else
    must fail the take, not ship it at the wrong pitch."""
    import json
    import pytest
    from pipeline import tts_engine as T
    ck = tmp_path / "ckpt"
    (ck / "speech_tokenizer").mkdir(parents=True)
    (ck / "speech_tokenizer" / "config.json").write_text(
        json.dumps({"output_sample_rate": 22050}), encoding="utf-8")
    T._declared_sr.pop(str(ck), None)
    assert T.declared_output_sr(str(ck)) == 22050
    with pytest.raises(RuntimeError, match="wrong pitch"):
        T.check_output_sr(str(ck), 24000)
    T.check_output_sr(str(ck), 22050)
    # unreadable -> guard skipped, never a false failure
    T._declared_sr.pop(str(tmp_path / "nope"), None)
    T._declared_sr[str(tmp_path / "nope")] = None
    T.check_output_sr(str(tmp_path / "nope"), 24000)


def test_title_case_is_lowercased_before_ruaccent_but_acronyms_are_not():
    """Model-card recommendation: RUAccent marks capitalized words worse."""
    from pipeline.text_norm import NORM_VERSIONS, lower_title_ru
    assert lower_title_ru("Какая сегодня погода. Потом мы пошли") == \
        "какая сегодня погода. потом мы пошли"
    assert lower_title_ru("Краски ООО и РФ") == "краски ООО и РФ"
    assert lower_title_ru("Paint Red") == "Paint Red"      # not Cyrillic
    assert NORM_VERSIONS["ru"] >= 2, "the marked text changed: salt the cache"
