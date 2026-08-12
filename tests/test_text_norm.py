"""The digits corpus — written BEFORE the locale logic, deliberately.

`localize_numbers` has never processed a number in production: 0 of 238
translated segments across every manifest on disk contains a digit, `%` or `°`
(measured 2026-08-11). Both bugs it carried were found by READING a path nobody
had run, which is exactly the situation where "fix the two you found" is the
wrong move — there was no reason to think two was all of them.

So this file is the specification, and it covers the shapes a painting course
actually produces: grouping separators, decimals, percentages, degrees, units,
ordinals and ranges. Everything here is checked in BOTH directions, because the
layer has two consumers that must agree:

  synthesis   pipeline.text_norm.localize_numbers  -> what the voice says
  QC          qc.backcheck._norm                   -> what WER compares against

They drifted apart once already (`%` expanded at synthesis, stripped as
punctuation at QC, so every percentage segment scored WER 1.0 and the autopilot
re-rolled a correct take until it gave up). The round-trip tests below are what
makes that impossible to reintroduce quietly.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.text_norm import localize_numbers  # noqa: E402
from qc.backcheck import _norm  # noqa: E402


# ---------------------------------------------------------------- grouping ---
# The bug: `_NUM` treated any , or . as a decimal point, so "1,000 ml" was
# SPOKEN as "one ml". A content error the WER check could not see, because the
# reference normalised the same wrong way.
#
# Grouping is locale-determined AND positional: a separator only groups when it
# is that language's grouping character and exactly three digits follow.

@pytest.mark.parametrize("lang,text,must_contain", [
    ("en", "Mix 1,000 ml",     "thousand"),
    ("en", "1,500 grams",      "thousand"),
    ("es", "Mezcla 1.000 ml",  "mil"),
    ("de", "1.000 ml",         "tausend"),
])
def test_grouping_separator_is_not_a_decimal_point(lang, text, must_contain):
    assert must_contain in localize_numbers(text, lang).lower()


@pytest.mark.parametrize("lang,text,unit", [
    ("en", "0.5 l", "l"),     # zero point five
    ("de", "0,5 l", "l"),     # null Komma fünf
    ("fr", "0,5 l", "l"),     # zéro virgule cinq
    ("ru", "0,5 л", "л"),     # ноль целых пять десятых — not a "запятая" reading
    ("es", "0,5 l", "l"),     # cero punto cinco — num2words uses "punto" for es
])
def test_decimal_separator_still_reads_as_a_decimal(lang, text, unit):
    """Asserted against num2words itself rather than a hand-guessed phrase: how
    a language SPELLS a fraction is that library's business (ru reads 0.5 as
    'ноль целых пять десятых', with no separator word at all). What this file
    owns is that the token is routed to the FLOAT path, not the int path."""
    from num2words import num2words
    assert localize_numbers(text, lang) == f"{num2words(0.5, lang=lang)} {unit}"


def test_grouping_and_decimal_are_opposite_in_en_and_de():
    """The same string means different numbers in different languages. This is
    the whole reason the rule cannot be locale-blind."""
    assert "thousand" in localize_numbers("1,500", "en").lower()   # 1500
    assert "komma" in localize_numbers("1,500", "de").lower()      # 1.5


@pytest.mark.parametrize("lang,text", [
    ("en", "1,00 ml"),      # 2 digits after a grouping sep — not grouping
    ("en", "1,0000 ml"),    # 4 digits — not grouping
    ("en", "1.234,56"),     # two separators: ambiguous, refuse to guess
    ("de", "1.234,56"),
])
def test_ambiguous_numerics_are_left_alone_rather_than_guessed(lang, text):
    """REFUSING is the requirement, not a limitation. A wrong expansion is
    spoken confidently and is invisible to WER (both sides normalise the same
    wrong way); an unexpanded token is at worst read in the wrong language,
    which the ear catches immediately."""
    token = text.split()[0]          # the numeric run, e.g. "1,00" / "1.234,56"
    assert token in localize_numbers(text, lang), (
        f"{lang}: {token!r} was expanded despite being ambiguous — a confident "
        f"wrong number is the one failure mode WER cannot see")


# ----------------------------------------------------------------- symbols ---
# The bug: synthesis expanded % and ° to words, QC stripped them as punctuation.
# ref "fifty" vs hyp "fifty percent" = WER 1.00 on a threshold of 0.15.

@pytest.mark.parametrize("lang,text", [
    ("en", "50%"), ("de", "50%"), ("fr", "50%"), ("es", "50%"), ("ru", "50%"),
    ("en", "heat to 20°C"), ("de", "20°C"), ("ru", "20°C"),
    ("en", "a 45° angle"), ("fr", "45°"),
])
def test_symbols_expand_the_same_way_on_both_sides(lang, text):
    """_norm must put the raw manifest text into the SAME domain as the audio.
    Whisper may write back either form, so both must normalise identically."""
    spoken = localize_numbers(text, lang)
    assert _norm(text, lang) == _norm(spoken, lang), (
        f"{lang}: QC normalises {text!r} to {_norm(text, lang)!r} but the voice "
        f"says {spoken!r} -> {_norm(spoken, lang)!r}")


def test_degree_celsius_wins_over_bare_degree():
    """'°C' must match before '°', or '20°C' becomes 'twenty degrees C'."""
    out = localize_numbers("20°C", "en").lower()
    assert "celsius" in out and not out.rstrip().endswith(" c")


# ------------------------------------------------------------------- units ---
# Deliberately NOT expanded (IMPROVEMENT_PLAN: language-correct inflection,
# especially ru case, is error-prone). The requirement is that they SURVIVE
# unchanged and that both sides agree about them.

@pytest.mark.parametrize("lang,text", [
    ("en", "250 ml of medium"), ("de", "250 ml"), ("ru", "250 мл"),
    ("en", "a 5 cm brush"), ("fr", "5 cm"),
])
def test_letter_units_pass_through_and_still_round_trip(lang, text):
    spoken = localize_numbers(text, lang)
    assert not any(c.isdigit() for c in spoken), f"digits left in {spoken!r}"
    assert _norm(text, lang) == _norm(spoken, lang)


# --------------------------------------------------- ordinals and ranges -----

@pytest.mark.parametrize("lang,text", [
    ("en", "step 3"), ("en", "Step 1. Prime the canvas."),
    ("de", "Schritt 3"), ("ru", "шаг 3"),
])
def test_a_trailing_period_is_sentence_punctuation_not_a_decimal(lang, text):
    """'Step 1.' must not become 'Step one point'."""
    assert "point" not in localize_numbers(text, "en").lower() or "en" != lang
    assert _norm(text, lang) == _norm(localize_numbers(text, lang), lang)


@pytest.mark.parametrize("lang,text", [
    ("en", "3-5 layers"), ("de", "3-5 Schichten"), ("ru", "3-5 слоёв"),
    ("en", "wait 10-15 minutes"),
])
def test_ranges_expand_both_endpoints(lang, text):
    out = localize_numbers(text, lang)
    assert not any(c.isdigit() for c in out), f"digits left in {out!r}"
    assert _norm(text, lang) == _norm(out, lang)


# ------------------------------------------------------- the WER round trip --
# The property that actually protects production: for every shape above, the
# text QC scores against and the text the voice speaks must normalise to the
# same string. If they do not, a CORRECT take is flagged and re-rolled.

CORPUS = [
    "Mix 1,000 ml of medium", "Use 0.5 l of solvent", "50% white",
    "heat to 20°C", "a 45° angle", "3-5 thin layers", "step 3",
    "250 ml", "5 cm from the edge", "wait 10-15 minutes",
]


@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("lang", ["en", "fr", "de", "es", "ru"])
def test_qc_and_synthesis_agree_on_every_corpus_shape(text, lang):
    spoken = localize_numbers(text, lang)
    assert _norm(text, lang) == _norm(spoken, lang), (
        f"{lang}: WER would compare {_norm(text, lang)!r} against audio saying "
        f"{_norm(spoken, lang)!r} — a correct take scores as a hallucination")


def test_whisper_writing_digits_back_still_matches():
    """The other direction: the voice says 'fifty percent', Whisper writes
    '50%'. Both must land in the same domain or the segment is flagged."""
    for lang in ("en", "de", "fr", "es", "ru"):
        asked = "50%"
        heard_as_words = localize_numbers("50%", lang)
        assert _norm(asked, lang) == _norm(heard_as_words, lang)


# ------------------------------------------------------- ё vs е (Russian) ----
# Russian is conventionally written WITHOUT ё and Whisper transcribes it that
# way, so `жёлтая` came back as `желтая` and scored a substitution. Measured on
# the first real ru backcheck: 7 segments over the 0.15 threshold, ё accounting
# for the entire difference in four of them.

@pytest.mark.parametrize("asked,heard", [
    ("Дальше у меня — жёлтая краска.", "Дальше у меня желтая краска."),
    ("И даёт очень интересные кремовые оттенки.",
     "И дает очень интересные кремовые оттенки."),
    ("Я добавляю в неё немного разбавителя.",
     "Я добавляю в нее немного разбавителя."),
    ("Жёлтые цвета — стронциановая жёлтая, потом охра.",
     "Желтые цвета стронциановая желтая, потом охра."),
    ("Ещё у меня есть краска.", "Еще у меня есть краска."),
])
def test_yo_spelling_is_not_a_transcription_error(asked, heard):
    from jiwer import wer
    assert wer(_norm(asked, "ru"), _norm(heard, "ru")) == 0.0, (
        "ё/е spelling scored as a substitution — a five-word line needs only "
        "one of these to cross the 0.15 flag threshold")


def test_uppercase_yo_folds_too():
    """_norm lowercases only after this step, so Ё has to be handled itself."""
    assert _norm("Ёлка", "ru") == _norm("Елка", "ru")


def test_yo_folding_is_russian_only():
    """No other target language has ё; folding elsewhere would be a silent
    no-op today and a trap if a Cyrillic language is ever added."""
    src = (ROOT / "qc" / "backcheck.py").read_text(encoding="utf-8")
    assert 'if lang == "ru":' in src


def test_wer_folding_does_not_contradict_the_stress_lexicon():
    """The stress table keeps ё and е APART on purpose — ё is always stressed,
    so the distinction is a remediation lever there. Here it is orthographic
    noise between two spellings of one sound. Different jobs; a future tidy-up
    must not unify them."""
    from qc.stress_words import lexicon_key
    assert lexicon_key("тёмным") != lexicon_key("темным")
    assert _norm("тёмным", "ru") == _norm("темным", "ru")
