"""Pre-TTS text normalization, applied ONLY at the synthesis boundary.

Two layers:

1. Number/symbol localization (`localize_numbers`) — ENGINE-AGNOSTIC, applied
   to every engine. Digits are read in the WRONG language or with the wrong
   decimal separator otherwise ("0.5" -> "null Komma fünf" in German, not
   "zero point five"). A measurement-heavy course needs this. Uses num2words
   (already a dep) with a graceful fallback to the raw token when a language
   isn't supported.

2. Russian stress — REMOVED 2026-08-02 with chatterbox. It was CHATTERBOX-ONLY
   (acute marks are a quirk of its training) and was never validated on any
   other engine, so it left with the engine rather than being pointed at qwen
   on the assumption that marks help. OPEN QUESTION: whether qwen mis-stresses
   Russian is UNTESTED — if the ru track sounds wrong, that is an A/B to run,
   and git history has the RUAccent implementation. Historically: chatterbox was
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

# --- layer 1: number/symbol localization (all engines) ---
# Per-language number-localization version, salted into synth_hash for EVERY
# engine. A lang listed here (>1) gets its digits expanded; the >1 also means
# existing caches (built before expansion) invalidate exactly once. A lang NOT
# listed is left untouched (no expansion, no key).
NUM_VERSIONS = {"en": 2, "fr": 2, "de": 2, "es": 2, "ru": 2}

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



