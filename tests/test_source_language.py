# SPDX-License-Identifier: AGPL-3.0-or-later
"""The source language is configurable (uk | ru): nothing downstream of s2 may
assume Ukrainian, and data written before the switch must still load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import manifest as M
from pipeline import s3_translate as S3
from pipeline.config_schema import ConfigError, validate

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path: Path, src: str = "ru") -> dict:
    return {"work_dir": str(tmp_path), "source_language": src}


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
    assert S3.LANG_NAMES["uk"] == "Ukrainian"
    assert cfg["tts"]["edge_voices"]["uk"].startswith("uk-UA-")
    assert cfg["mux"]["lang_tags"]["uk"] == "ukr"


def test_source_language_is_uk_or_ru() -> None:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    validate(dict(cfg, source_language="ru"))
    with pytest.raises(ConfigError):
        validate(dict(cfg, source_language="pl"))
