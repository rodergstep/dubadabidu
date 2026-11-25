# SPDX-License-Identifier: AGPL-3.0-or-later
"""The source language is configurable (uk | ru): nothing downstream of s2 may
assume Ukrainian, and data written before the switch must still load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dubadabidu.core import manifest as M
from dubadabidu.stages import s3_translate as S3
from dubadabidu.core.config.schema import ConfigError, validate
from tests.paths import REPO, config
from dubadabidu.core.config.schema import Config
from tests.cfgs import settings, toy_tts

ROOT = REPO


def _cfg(tmp_path: Path, src: str = "ru") -> Config:
    return settings({"work_dir": str(tmp_path), "source_language": src})


def test_legacy_manifest_text_uk_is_read_as_text_src(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    p = M.manifest_path(cfg, "input/v.mp4")
    p.write_text(json.dumps({"utterances": [
        {"id": "u0001", "text_uk": "джерело", "tr": {}}]}), encoding="utf-8")
    u = M.load(cfg, "input/v.mp4")["utterances"][0]
    assert u["text_src"] == "джерело" and "text_uk" not in u


def test_legacy_refs_json_text_uk_is_read_as_text_src(tmp_path: Path) -> None:
    (tmp_path / "refs.json").write_text(json.dumps(
        {"a_ref_01.wav": {"text_uk": "t", "start": 0.0}}), encoding="utf-8")
    assert M.load_refs(tmp_path)["a_ref_01.wav"]["text_src"] == "t"
    assert M.load_refs(tmp_path / "missing") == {}


def test_legacy_terms_cache_uk_key_is_read_as_src(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "uk")
    wd = M.video_workdir(cfg, "input/v.mp4")
    (wd / "terms_en.json").write_text(json.dumps(
        {"terms": [{"uk": "лесирування", "tr": "glazing"}]}), encoding="utf-8")
    terms = S3.load_terms(cfg, "input/v.mp4", "en")
    assert terms == [{"src": "лесирування", "tr": "glazing"}]
    assert S3.terms_block(cfg, terms).startswith(
        "Video-specific terminology (Ukrainian -> target")


@pytest.mark.parametrize("name", ["translate_system", "translate_reflect",
                                  "translate_adapt"])
def test_prompts_name_the_configured_source(monkeypatch: pytest.MonkeyPatch,
                                            name: str) -> None:
    monkeypatch.chdir(ROOT)
    text = S3._prompt("ru", "uk", 0.1, 2, name)
    assert "Russian" in text and "Ukrainian" in text      # ru -> uk
    assert "{SRC}" not in text and "{LANG}" not in text
    assert "Ukrainian" not in S3._prompt("ru", "en", 0.1, 2, name)


def test_glossary_is_scoped_to_the_source_language(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    assert S3._glossary("uk", "en").startswith(
        "Mandatory terminology (Ukrainian -> target)")
    assert S3._glossary("ru", "en") == ""     # no ru glossary: nothing injected


def test_uk_is_a_target_language() -> None:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    from dubadabidu.core.languages import lang_name
    assert lang_name("uk") == "Ukrainian"
    assert cfg["tts"]["edge_voices"]["uk"].startswith("uk-UA-")
    assert cfg["mux"]["lang_tags"]["uk"] == "ukr"


def test_source_language_is_uk_or_ru() -> None:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    validate(dict(cfg, source_language="ru"))
    with pytest.raises(ConfigError):
        validate(dict(cfg, source_language="pl"))


def test_uk_is_voiced_by_edge() -> None:
    """FINDINGS §2.4: qwen's "Auto" Ukrainian was unusable by ear."""
    import yaml
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    gpu = yaml.safe_load(config("config.gpu.yaml").read_text(encoding="utf-8"))
    from dubadabidu.core.logic import deep_merge
    for cfg in (base, deep_merge(base, gpu)):
        t = validate(cfg).tts
        assert M.resolve_engine(t, "uk") == "edge"
        assert M.resolve_engine(t, "en") == "qwen"


def test_edge_routed_on_purpose_is_not_a_plumbing_run() -> None:
    man = {"utterances": [{"id": "u0001", "tr": {
        "uk": {"synth_engine": "edge"}, "en": {"synth_engine": "edge"}}}]}
    tts = toy_tts(engine="qwen", engine_by_lang={"uk": "edge"})
    assert M.edge_langs(man, ["en", "uk"]) == ["en", "uk"]   # verdicts refuse both
    assert M.edge_fallback_langs(man, ["en", "uk"], tts) == ["en"]
