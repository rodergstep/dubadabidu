"""Pre-TTS text normalization, applied ONLY at the synthesis boundary.

Russian: Chatterbox was trained with combining-acute stress marks (its own
optional russian_text_stresser dep is unmaintained and conflicts with
chatterbox-tts — resemble-ai/chatterbox#304/#340). We mark stress with
RUAccent (Apache-2.0, context-aware homographs) and convert its plus
notation to combining acutes: "дер+евья" -> "дере́вья". A/B-verified
2026-07-08: acute-marked input fixes stress; plus notation breaks generation.

Everything downstream of synthesis (subtitles, backcheck WER, manifest,
synth_hash) keeps the CLEAN text — whisper never emits accent marks, and
hashing clean text keeps the cache stable. Consequence: if this
normalization changes, bump/delete the affected seg/<lang> caches manually.
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("dubadabidu.text_norm")

# Per-language normalization version, salted into synth_hash so cache
# invalidation is automatic, not a manual chore. Bump a language's number
# whenever its normalize_for_tts output changes for the same input; when
# ADDING normalization for a new language, start it at 2 (its existing
# caches were built unnormalized). Version 1 adds no hash key, so all
# pre-existing caches stay valid.
NORM_VERSIONS = {"ru": 1}
_ruaccent = None
_PLUS = re.compile(r"\+(.)")
ACUTE = "́"


def plus_to_acute(text: str) -> str:
    """RUAccent plus notation -> combining acute after the stressed vowel."""
    return _PLUS.sub("\\1" + ACUTE, text)


def _accent_ru(text: str) -> str:
    global _ruaccent
    try:
        if _ruaccent is None:
            from ruaccent import RUAccent
            log.info("loading RUAccent (turbo3.1) ...")
            _ruaccent = RUAccent()
            _ruaccent.load(omograph_model_size="turbo3.1", use_dictionary=True)
        return plus_to_acute(_ruaccent.process_all(text))
    except Exception as e:  # never let normalization kill a synthesis run
        log.warning("RUAccent failed (%s); synthesizing unaccented", e)
        return text


def normalize_for_tts(text: str, lang: str) -> str:
    if lang == "ru":
        return _accent_ru(text)
    return text
