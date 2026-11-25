# SPDX-License-Identifier: AGPL-3.0-or-later
"""The language registry: one row per language, the only per-language table.

Adding a language is one row here, plus glossary/<source>/<target>.csv if
wanted; the s3 prompts are already parameterised ({SRC}, {LANG}).

  name           English name, used in every s3 prompt and f-string
  whisper        Whisper language code; None = it cannot be a source
  qwen           Qwen3-TTS language name; None = qwen does not list it, and the
                 adapter falls back to "Auto" (as it always has)
  edge_voice     edge-tts voice (config tts.edge_voices may override it)
  sub_tag        ISO 639-2 tag for the mux (config mux.lang_tags may override it)
  roles          source and/or target
  forced_engine  the engine this target must use whatever tts.engine says:
                 uk -> edge, because qwen has no Ukrainian and its "Auto" uk was
                 unusable by ear (FINDINGS §2.4)
  glossary_lead  s2's initial_prompt lead-in ("Lesson vocabulary:") in the
                 source language
  stress_marking the review page offers per-word stress marks (lexical stress
                 is a defect class there; FINDINGS §2.1k)
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Role = Literal["source", "target"]
Engine = Literal["qwen", "edge"]


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    whisper: str | None
    qwen: str | None
    edge_voice: str
    sub_tag: str
    roles: frozenset[Role]
    forced_engine: Engine | None = None
    glossary_lead: str | None = None
    stress_marking: bool = False


_BOTH: frozenset[Role] = frozenset({"source", "target"})
_TARGET: frozenset[Role] = frozenset({"target"})

LANGUAGES: Mapping[str, Language] = {l.code: l for l in (
    Language("uk", "Ukrainian", "uk", None, "uk-UA-OstapNeural", "ukr", _BOTH,
             forced_engine="edge", glossary_lead="Словник уроку: ", stress_marking=True),
    Language("ru", "Russian", "ru", "Russian", "ru-RU-DmitryNeural", "rus", _BOTH,
             glossary_lead="Словарь урока: ", stress_marking=True),
    Language("en", "English", None, "English", "en-US-AndrewNeural", "eng", _TARGET),
    Language("fr", "French", None, "French", "fr-FR-HenriNeural", "fra", _TARGET),
    Language("de", "German", None, "German", "de-DE-ConradNeural", "deu", _TARGET),
    Language("es", "Spanish", None, "Spanish", "es-ES-AlvaroNeural", "spa", _TARGET),
    Language("pl", "Polish", None, None, "pl-PL-MarekNeural", "pol", _TARGET),
)}


def lang_name(code: str) -> str:
    """The prompt name of a language (KeyError for an unknown code, as the
    old LANG_NAMES[code] lookup raised)."""
    return LANGUAGES[code].name


def qwen_language(code: str) -> str:
    """Qwen3-TTS's name for a language, "Auto" where qwen has none."""
    lang = LANGUAGES.get(code)
    return (lang.qwen if lang else None) or "Auto"


def glossary_lead(code: str) -> str:
    lang = LANGUAGES.get(code)
    return (lang.glossary_lead if lang else None) or ""


def sources() -> tuple[str, ...]:
    return tuple(c for c, l in LANGUAGES.items() if "source" in l.roles)


def targets() -> tuple[str, ...]:
    return tuple(c for c, l in LANGUAGES.items() if "target" in l.roles)


def stress_marking() -> tuple[str, ...]:
    return tuple(sorted(c for c, l in LANGUAGES.items() if l.stress_marking))


def edge_voices() -> dict[str, str]:
    return {c: l.edge_voice for c, l in LANGUAGES.items()}


def sub_tags() -> dict[str, str]:
    return {c: l.sub_tag for c, l in LANGUAGES.items()}
