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


def marked_words(lang: str) -> set[str]:
    """Surface forms a native listener flagged as mis-stressed, from
    stress_lexicon_<lang>.json. Empty when the table does not exist yet, which
    makes respelling a no-op rather than a guess."""
    import json
    from pathlib import Path
    p = Path(f"stress_lexicon_{lang}.json")
    if not p.exists():
        return set()
    try:
        lex = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {f for e in lex.values() for f in (e.get("forms") or [])}


def normalize_for_tts(text: str, lang: str, tts_cfg: dict | None = None) -> str:
    """Stress-mark Russian when tts.ru_stress is on. Engine-agnostic now: the
    old version hardcoded chatterbox, which is why it became unreachable."""
    cfg = tts_cfg or {}
    if lang == "ru" and cfg.get("ru_stress"):
        return accent_ru(text)
    return text


def marked_words(lang: str) -> set[str]:
    """Surface forms a native listener flagged as mis-stressed, from
    stress_lexicon_<lang>.json. Empty when the table does not exist yet, which
    makes respelling a no-op rather than a guess."""
    import json
    from pathlib import Path
    p = Path(f"stress_lexicon_{lang}.json")
    if not p.exists():
        return set()
    try:
        lex = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {f for e in lex.values() for f in (e.get("forms") or [])}


def normalize_for_tts(text: str, lang: str, tts_cfg: dict | None = None) -> str:
    """Stress-mark Russian when tts.ru_stress is on. Engine-agnostic now: the
    old version hardcoded chatterbox, which is why it became unreachable."""
    cfg = tts_cfg or {}
    if lang == "ru" and cfg.get("ru_stress"):
        return accent_ru(text)
    return text


# --- layer 3: Russian REDUCTION RESPELLING (2026-08-13, UNTESTED) ------------
# The last untried remediation for wrong Russian stress. FINDINGS 2.1b/2.1bb
# refuted feeding qwen combining acutes (U+0301): it was not trained on them and
# reads them as content to pronounce — 97% of clips unusable by ear. That
# refutation is about a DIACRITIC, and says nothing about respelling a word in
# ordinary Cyrillic, which qwen certainly saw in training.
#
# The mechanism is akanye/ikanye. Russian reduces unstressed vowels: /o/ is
# realised [a] and unstressed /e/ raises toward [i], while the stressed vowel
# keeps full quality. So writing the reduction explicitly leaves the stressed
# vowel as the only unreduced one, and the stress position is encoded in
# ordinary letters rather than in a mark. qc/stress_detect measured that this
# contrast is real in the audio: stressed-о words produce [o] 89% of the time
# against 26% for unstressed ones (FINDINGS 2.1g.2).
#
# TARGETED, NOT GLOBAL, and that is the whole design. Respelling every word
# would hand the engine a page of misspelled Russian to test one hypothesis.
# Only words in stress_lexicon_<lang>.json are touched — the finite set a native
# listener has actually marked wrong (FINDINGS 2.1k) — so a failure costs those
# words and nothing else.
#
# THE ORACLE IS RUAccent, and it is imperfect: FINDINGS 2.1h caught it marking
# `перед` on the wrong slot. Here that matters less than it did for a detector,
# because the word list is human-confirmed-wrong to begin with and the A/B is
# judged by ear. But a respelling is only as right as the stress it encodes.
#
# THE REDUCTION RULES ARE A SIMPLIFICATION. Real Russian reduction is
# position-dependent (first pretonic differs from the rest) and conditioned by
# the preceding consonant's palatalisation. This does о->а and unstressed е->и
# and no more. It is a gate, not a phonology: if the simple version moves
# nothing, the elaborate one is not worth building.
RESPELL_VERSIONS = {"ru": 1}


def _stressed_index(word: str) -> int | None:
    """Index of the stressed character in `word`, via RUAccent. None if unknown."""
    marked = accent_ru_plus(word)
    if marked is None:
        return None
    i = marked.find("+")
    return i if i >= 0 else None      # '+' precedes the vowel; removing it lands here


def accent_ru_plus(text: str) -> str | None:
    """RUAccent's raw PLUS notation ('дер+евья'), or None if unavailable.

    Kept separate from accent_ru(), which converts to combining acutes for the
    refuted marking path. Respelling needs the POSITION, not the mark.
    """
    global _ruaccent
    try:
        if _ruaccent is None:
            from ruaccent import RUAccent
            log.info("loading RUAccent (turbo3.1) ...")
            _ruaccent = RUAccent()
            _ruaccent.load(omograph_model_size="turbo3.1", use_dictionary=True)
        return _ruaccent.process_all(text)
    except Exception as e:
        log.warning("RUAccent unavailable (%s); cannot respell", e)
        return None


def respell_word_ru(word: str) -> str:
    """One word rewritten so its unstressed vowels are spelled as reduced.

    Returns the word unchanged when the stress position is unknown — silently
    guessing would encode a WRONG stress, which is worse than encoding none.
    """
    idx = _stressed_index(word)
    if idx is None or not (0 <= idx < len(word)):
        return word
    out = []
    for i, ch in enumerate(word):
        if i == idx:
            out.append(ch)                    # the stressed vowel keeps quality
        elif ch == "о":
            out.append("а")                   # akanye
        elif ch == "е":
            out.append("и")                   # ikanye
        elif ch == "О":
            out.append("А")
        elif ch == "Е":
            out.append("И")
        else:
            out.append(ch)
    return "".join(out)


def respell_ru(text: str, words: set[str]) -> str:
    """Respell only the members of `words` (case-insensitively) found in `text`."""
    if not words:
        return text
    low = {w.casefold() for w in words}

    def _one(m: "re.Match[str]") -> str:
        w = m.group()
        return respell_word_ru(w) if w.casefold() in low else w

    return re.sub(r"[^\W\d_]+", _one, text, flags=re.UNICODE)
