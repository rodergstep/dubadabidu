"""Pre-TTS text normalization, applied ONLY at the synthesis boundary.

Two layers:

1. Number/symbol localization (`localize_numbers`) — ENGINE-AGNOSTIC, applied
   to every engine. Digits are read in the WRONG language or with the wrong
   decimal separator otherwise ("0.5" -> "null Komma fünf" in German, not
   "zero point five"). A measurement-heavy course needs this. Uses num2words
   (already a dep) with a graceful fallback to the raw token when a language
   isn't supported.

2. Russian stress (`normalize_for_tts`) — CHATTERBOX-ONLY. Chatterbox was
   trained with combining-acute stress marks (its own optional
   russian_text_stresser dep is unmaintained and conflicts with chatterbox-tts
   — resemble-ai/chatterbox#304/#340). We mark stress with RUAccent (Apache-2.0,
   context-aware homographs) and convert its plus notation to combining acutes:
   "дер+евья" -> "дере́вья". A/B-verified 2026-07-08: acute-marked input fixes
   stress; plus notation breaks generation.

Everything downstream of synthesis (subtitles, backcheck WER, manifest) keeps
the CLEAN text — whisper never emits accent marks or word-numbers, and the
manifest stays human-editable. Cache stability is handled by the version salts
below (synth_hash keys on them); if either layer's output changes for the same
input, bump the version.
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("dubadabidu.text_norm")

# --- layer 2: chatterbox RU stress ---
# Per-language stress-normalization version, salted into synth_hash. Version 1
# adds no hash key, so pre-existing caches stay valid.
NORM_VERSIONS = {"ru": 1}

# --- layer 1: number/symbol localization (all engines) ---
# Per-language number-localization version, salted into synth_hash for EVERY
# engine. A lang listed here (>1) gets its digits expanded; the >1 also means
# existing caches (built before expansion) invalidate exactly once. A lang NOT
# listed is left untouched (no expansion, no key).
NUM_VERSIONS = {"en": 2, "fr": 2, "de": 2, "es": 2, "ru": 2}

_ruaccent = None
_PLUS = re.compile(r"\+(.)")
ACUTE = "́"

# a run of digits with an optional single decimal separator (either . or ,)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
# symbols TTS routinely mangles, expanded per language after the numbers
_SYMBOLS = {
    "°C": {"en": "degrees Celsius", "fr": "degrés Celsius",
           "de": "Grad Celsius", "es": "grados Celsius",
           "ru": "градусов Цельсия"},
    "%": {"en": "percent", "fr": "pour cent", "de": "Prozent",
          "es": "por ciento", "ru": "процентов"},
    "°": {"en": "degrees", "fr": "degrés", "de": "Grad", "es": "grados",
          "ru": "градусов"},
}


def plus_to_acute(text: str) -> str:
    """RUAccent plus notation -> combining acute after the stressed vowel."""
    return _PLUS.sub("\\1" + ACUTE, text)


def localize_numbers(text: str, lang: str) -> str:
    """Expand digits (and %, ° symbols) into target-language words. Applied to
    ALL engines at the synth boundary; the manifest/subs/WER keep raw digits.
    No-op for languages not in NUM_VERSIONS or if num2words is unavailable."""
    if lang not in NUM_VERSIONS:
        return text
    try:
        from num2words import num2words
    except ImportError:
        return text

    def _repl(m: "re.Match[str]") -> str:
        tok = m.group()
        try:
            if re.search(r"\d[.,]\d", tok):        # decimal
                return num2words(float(tok.replace(",", ".")), lang=lang)
            return num2words(int(tok), lang=lang)  # integer
        except (NotImplementedError, ValueError, OverflowError):
            return tok                             # unsupported lang/number: leave as-is

    text = _NUM.sub(_repl, text)
    for sym, words in _SYMBOLS.items():   # "°C" before "°" (dict order preserved)
        if sym in text and lang in words:
            text = text.replace(sym, " " + words[lang])
    return text


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
    """Chatterbox-only stress marking (layer 2). Number localization (layer 1)
    is applied separately in tts_engine.synthesize for every engine."""
    if lang == "ru":
        return _accent_ru(text)
    return text
