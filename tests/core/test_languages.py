# SPDX-License-Identifier: AGPL-3.0-or-later
"""The language registry is the only per-language table, and it reproduces
every table it replaced (s3 LANG_NAMES, tts QWEN_LANGS, s2 _GLOSSARY_LEAD,
stress_words STRESS_LANGS) and the ones config.yaml still carries."""
from __future__ import annotations

import typing

import yaml

from dubadabidu.core import languages as L
from dubadabidu.core.config.schema import Config
from tests.paths import REPO


def test_names_qwen_leads_and_stress_rows_are_what_the_code_used() -> None:
    assert {c: L.lang_name(c) for c in L.LANGUAGES} == {
        "en": "English", "fr": "French", "de": "German", "es": "Spanish",
        "ru": "Russian", "pl": "Polish", "uk": "Ukrainian"}
    assert {c: L.qwen_language(c) for c in ("en", "fr", "de", "es", "ru", "uk", "pl", "xx")} == {
        "en": "English", "fr": "French", "de": "German", "es": "Spanish",
        "ru": "Russian", "uk": "Auto", "pl": "Auto", "xx": "Auto"}
    assert (L.glossary_lead("uk"), L.glossary_lead("ru"), L.glossary_lead("en")) == (
        "Словник уроку: ", "Словарь урока: ", "")
    assert L.stress_marking() == ("ru", "uk")


def test_registry_matches_the_tables_config_yaml_ships() -> None:
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["tts"]["edge_voices"] == L.edge_voices()
    assert cfg["mux"]["lang_tags"] == L.sub_tags()
    forced = {c: l.forced_engine for c, l in L.LANGUAGES.items() if l.forced_engine}
    assert cfg["tts"]["engine_by_lang"] == forced


def test_schema_sources_are_the_registry_sources() -> None:
    literal = typing.get_args(Config.model_fields["source_language"].annotation)
    assert set(literal) == set(L.sources())
