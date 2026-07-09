"""Pure-logic tests: pause-aware segment splitting (s2)."""
from pipeline.logic import split_at_pauses


def w(start, end, word):
    return {"start": start, "end": end, "word": " " + word}


def test_no_pauses_single_part():
    words = [w(0.0, 0.4, "hello"), w(0.5, 0.9, "world")]
    assert split_at_pauses(words, 0.6) == [
        {"start": 0.0, "end": 0.9, "text": "hello world"}]


def test_split_at_long_gap():
    words = [w(0.0, 0.4, "first"), w(0.5, 0.9, "part"),
             w(2.0, 2.4, "second"), w(2.5, 2.9, "part")]
    parts = split_at_pauses(words, 0.6)
    assert [p["text"] for p in parts] == ["first part", "second part"]
    assert parts[0]["end"] == 0.9 and parts[1]["start"] == 2.0


def test_teaching_cadence_many_pauses():
    # 22s narration with three real pauses -> four parts, each ~in its slot
    words = ([w(i * 0.5, i * 0.5 + 0.4, f"a{i}") for i in range(4)]
             + [w(5.0 + i * 0.5, 5.0 + i * 0.5 + 0.4, f"b{i}") for i in range(4)]
             + [w(12.0 + i * 0.5, 12.0 + i * 0.5 + 0.4, f"c{i}") for i in range(4)]
             + [w(18.0 + i * 0.5, 18.0 + i * 0.5 + 0.4, f"d{i}") for i in range(4)])
    parts = split_at_pauses(words, 0.6)
    assert len(parts) == 4
    assert parts[-1]["start"] == 18.0


def test_short_fragment_glued_to_previous():
    # a lone hesitation word after a pause must not become its own segment
    words = [w(0.0, 0.4, "real"), w(0.5, 0.9, "sentence"), w(3.0, 3.2, "uh")]
    parts = split_at_pauses(words, 0.6, min_words=2)
    assert len(parts) == 1
    assert parts[0]["text"] == "real sentence uh"


def test_leading_short_fragment_kept():
    words = [w(0.0, 0.3, "so"), w(2.0, 2.4, "let's"), w(2.5, 2.9, "begin")]
    parts = split_at_pauses(words, 0.6, min_words=2)
    assert [p["text"] for p in parts] == ["so", "let's begin"]


def test_empty_words():
    assert split_at_pauses([], 0.6) == []


def test_sentence_boundary_splits_at_lower_gap():
    # 0.4s pause: below max_gap, but after a period -> split
    words = [w(0.0, 0.4, "one"), w(0.5, 0.9, "sentence."),
             w(1.3, 1.7, "another"), w(1.8, 2.2, "one")]
    parts = split_at_pauses(words, 0.6, sent_gap=0.25)
    assert [p["text"] for p in parts] == ["one sentence.", "another one"]


def test_mid_sentence_small_gap_not_split():
    # same 0.4s pause, no sentence end -> stays together
    words = [w(0.0, 0.4, "no"), w(0.5, 0.9, "period"),
             w(1.3, 1.7, "keeps"), w(1.8, 2.2, "going")]
    assert len(split_at_pauses(words, 0.6, sent_gap=0.25)) == 1
