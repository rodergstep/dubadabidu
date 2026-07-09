"""Pure-logic tests: RUAccent plus-notation -> combining acute conversion."""
from pipeline.text_norm import plus_to_acute, ACUTE


def test_single_word():
    assert plus_to_acute("дер+евья") == "дере" + ACUTE + "вья"


def test_multiple_words_and_untouched_text():
    src = "М+елкие ф+ормы, а также тех"
    out = plus_to_acute(src)
    assert "+" not in out
    assert out.count(ACUTE) == 2
    assert out.endswith("а также тех")


def test_no_marks_passthrough():
    assert plus_to_acute("уже размечено") == "уже размечено"
