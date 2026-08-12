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
_WARNED_NO_NUM2WORDS = False

# --- layer 2: chatterbox RU stress ---
# Per-language stress-normalization version, salted into synth_hash. Version 1
# adds no hash key, so pre-existing caches stay valid.

# --- layer 1: number/symbol localization (all engines) ---
# Per-language number-localization version, salted into synth_hash for EVERY
# engine. A lang listed here (>1) gets its digits expanded; the >1 also means
# existing caches (built before expansion) invalidate exactly once. A lang NOT
# listed is left untouched (no expansion, no key).
# v3 (2026-08-11): locale-aware grouping/decimal separators. v2 read every
# "," and "." as a decimal point, so "1,000 ml" was SPOKEN as "one ml" — the
# expansion changed for the same input, which is precisely what this salt is
# for. Segments containing a grouped number re-synthesize once; nothing else
# moves (no manifest on disk contains a digit, so in practice this is a no-op
# that keeps the invariant honest).
NUM_VERSIONS = {"en": 3, "fr": 3, "de": 3, "es": 3, "ru": 3}

# A full numeric run, INCLUDING every separator inside it. Matching the whole
# run (rather than `\d+(?:[.,]\d+)?`) is what lets _expand_token see that
# "1.234,56" carries two separators and is therefore ambiguous. A trailing
# sentence period is not consumed: the group requires digits AFTER the
# separator, so "Step 1." still matches just "1".
_NUM = re.compile(r"\d+(?:[.,]\d+)*")

# (decimal, grouping) per language. Both are needed because the SAME string
# means different numbers in different languages: "1,500" is 1500 in English
# and 1.5 in German. A separator groups ONLY when it is that language's
# grouping character and exactly three digits follow — the minimal rule that
# covers real course text without becoming a number parser.
#
# KNOWN GAP, deliberately not covered: fr/ru group with a (thin) space rather
# than a dot, so "1 000" still reads as two numbers. Recognising that means
# deciding whether "wait 10 15 minutes" is one number or two, which is a guess.
# Zero field exposure — no manifest on disk contains a grouped number at all.
_SEPS = {"en": (".", ","), "es": (",", "."), "de": (",", "."),
         "fr": (",", None), "ru": (",", None)}
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



def _expand_token(tok: str, lang: str, num2words) -> str | None:
    """One numeric run -> words, or None when the token is AMBIGUOUS.

    Returning None (leave the digits alone) is a feature, not a shortfall. A
    wrong expansion is spoken confidently AND invisible to QC, because backcheck
    normalises the reference through this same function and so makes the same
    mistake on both sides — that is exactly how "Mix 1,000 ml" became "Mix one
    ml" with a clean WER. An unexpanded token is at worst read in the wrong
    language, which the ear catches on the first listen.
    """
    dec, grp = _SEPS.get(lang, (".", None))
    seps = re.findall(r"[.,]", tok)
    parts = re.split(r"[.,]", tok)
    try:
        if not seps:
            return num2words(int(tok), lang=lang)
        if len(seps) == 1:
            if grp and seps[0] == grp and len(parts[1]) == 3:
                return num2words(int(parts[0] + parts[1]), lang=lang)
            if seps[0] == dec:
                return num2words(float(parts[0] + "." + parts[1]), lang=lang)
        return None          # 2+ separators, or a separator this locale doesn't use
    except (NotImplementedError, ValueError, OverflowError):
        return None          # unsupported language/number: leave as-is


def localize_numbers(text: str, lang: str) -> str:
    """Expand digits (and %, ° symbols) into target-language words. Applied to
    ALL engines at the synth boundary; the manifest/subs keep raw digits.
    No-op for languages not in NUM_VERSIONS or if num2words is unavailable.

    qc.backcheck._norm runs the SAME function over the reference text before
    computing WER, so the two sides always share a domain. They did not until
    2026-08-11, and every segment containing % or ° scored WER 1.0 as a result.
    """
    if lang not in NUM_VERSIONS:
        return text
    try:
        from num2words import num2words
    except ImportError:
        # SAY SO. This used to return the text unchanged and silently: digits
        # would then reach the engine raw and be read in the wrong language,
        # which is the defect this function exists to prevent, on a paid pod,
        # with nothing in the log. num2words is a core dependency, so reaching
        # here means a broken install rather than a supported configuration.
        global _WARNED_NO_NUM2WORDS
        if not _WARNED_NO_NUM2WORDS:
            _WARNED_NO_NUM2WORDS = True
            log.warning(
                "num2words is not importable — digits and %%/° are being sent "
                "to the engine RAW and will be read in the wrong language. "
                "It is a core dependency: pip install -e '.[dev]'")
        return text

    def _repl(m: "re.Match[str]") -> str:
        return _expand_token(m.group(), lang, num2words) or m.group()

    text = _NUM.sub(_repl, text)
    for sym, words in _SYMBOLS.items():   # "°C" before "°" (dict order preserved)
        if sym in text and lang in words:
            text = text.replace(sym, " " + words[lang])
    return text

# --- layer 2: Russian lexical stress (RESTORED 2026-08-03) -------------------
# Removed with chatterbox on 2026-08-02 because the call site was gated on
# engine == "chatterbox". That commit recorded the risk verbatim: "Whether qwen
# mis-stresses Russian is UNTESTED - and a ru track is about to ship. If it
# sounds wrong, that is an A/B to run and git history has the RUAccent
# implementation." It shipped, and the listener reported wrong stress on the
# Russian track. So: back, pointed at whichever engine is configured.
#
# Chatterbox was TRAINED on combining acutes; qwen was not, so this is a
# hypothesis, not a restoration of known-good behaviour — gate it on
# tts.ru_stress and let an A/B decide. The 2026-07-08 finding that plus notation
# BREAKS generation still stands: convert to acutes, never pass "дер+евья".
NORM_VERSIONS = {"ru": 1}
_ruaccent = None
_PLUS = re.compile(r"\+(.)")
ACUTE = "\u0301"


def plus_to_acute(text: str) -> str:
    """RUAccent plus notation -> combining acute after the stressed vowel."""
    return _PLUS.sub("\\1" + ACUTE, text)


def accent_ru(text: str) -> str:
    """Mark Russian lexical stress. Never raises: an unaccented synthesis is a
    quality regression, a crashed one is a lost pod."""
    global _ruaccent
    try:
        if _ruaccent is None:
            from ruaccent import RUAccent
            log.info("loading RUAccent (turbo3.1) ...")
            _ruaccent = RUAccent()
            _ruaccent.load(omograph_model_size="turbo3.1", use_dictionary=True)
        return plus_to_acute(_ruaccent.process_all(text))
    except Exception as e:
        log.warning("RUAccent unavailable (%s); synthesizing unaccented", e)
        return text


def normalize_for_tts(text: str, lang: str, tts_cfg: dict | None = None) -> str:
    """Stress-mark Russian when tts.ru_stress is on. Engine-agnostic now: the
    old version hardcoded chatterbox, which is why it became unreachable."""
    if lang == "ru" and (tts_cfg or {}).get("ru_stress"):
        return accent_ru(text)
    return text
