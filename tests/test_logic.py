from pipeline.logic import merge_segments, decide_fit


def test_merge_respects_max_chars():
    segs = [{"start": 0, "end": 2, "text": "a" * 100 + "."},
            {"start": 2.5, "end": 4, "text": "b" * 100 + "."},
            {"start": 4.5, "end": 6, "text": "c" * 100 + "."}]
    out = merge_segments(segs, max_chars=150, max_seconds=30)
    assert len(out) == 3


def test_merge_joins_short_clauses():
    segs = [{"start": 0, "end": 1, "text": "Беремо пензель,"},
            {"start": 1.1, "end": 2, "text": "змочуємо його."}]
    out = merge_segments(segs, max_chars=220, max_seconds=12)
    assert len(out) == 1 and out[0]["text"].endswith(".")


def test_merge_splits_on_sentence_plus_gap():
    segs = [{"start": 0, "end": 1, "text": "Готово."},
            {"start": 2.0, "end": 3, "text": "Далі."}]
    out = merge_segments(segs, max_chars=220, max_seconds=12)
    assert len(out) == 2


def test_merge_respects_max_seconds():
    segs = [{"start": 0, "end": 8, "text": "перша частина"},
            {"start": 8, "end": 15, "text": "друга частина"}]
    out = merge_segments(segs, max_chars=500, max_seconds=12)
    assert len(out) == 2


def test_fit_as_is():
    assert decide_fit(3.0, 4.0, 1.12) == ("as_is", 1.0)


def test_fit_stretch():
    v, t = decide_fit(4.4, 4.0, 1.12)
    assert v == "stretch" and abs(t - 1.1) < 1e-9


def test_fit_no():
    v, t = decide_fit(6.0, 4.0, 1.12)
    assert v == "no" and t == 1.5


def test_fit_zero_slot():
    assert decide_fit(1.0, 0.0, 1.12)[0] == "no"
