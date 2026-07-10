from pipeline.logic import retime_step
from pipeline.tts_engine import _take_rank


def test_take_rank_prefers_source_pace():
    # identical quality metrics; only pacing differs (target 10s)
    fast = {"mos_min": 3.0, "f0st": 2.5, "sim": 0.75, "dur": 8.0}   # -20%
    close = {"mos_min": 3.0, "f0st": 2.5, "sim": 0.75, "dur": 9.8}  # -2%
    assert _take_rank(close, 10.0) > _take_rank(fast, 10.0)


def test_take_rank_pace_cannot_beat_much_better_mos():
    good = {"mos_min": 4.5, "f0st": 2.5, "sim": 0.75, "dur": 8.5}
    bad = {"mos_min": 1.5, "f0st": 2.5, "sim": 0.75, "dur": 10.0}
    assert _take_rank(good, 10.0) > _take_rank(bad, 10.0)


def test_take_rank_without_target_ignores_duration():
    a = {"mos_min": 3.0, "f0st": 2.5, "sim": 0.75, "dur": 5.0}
    b = {"mos_min": 3.0, "f0st": 2.5, "sim": 0.75, "dur": 15.0}
    assert _take_rank(a) == _take_rank(b)


def test_no_drift_when_source_has_room():
    placed, drift, slot = retime_step(
        src_start=10.0, prev_end=8.0, next_src_start=15.0,
        drift_max=1.5, min_gap=0.15)
    assert placed == 10.0 and drift == 0.0
    assert abs(slot - (15.0 + 1.5 - 10.0 - 0.15)) < 1e-9


def test_drift_accumulates_after_long_dub():
    # previous dub ran until 10.8 though this segment's source start is 10.0
    placed, drift, slot = retime_step(
        src_start=10.0, prev_end=10.8, next_src_start=15.0,
        drift_max=1.5, min_gap=0.15)
    assert abs(placed - 10.95) < 1e-9 and abs(drift - 0.95) < 1e-9


def test_drift_resets_at_source_pause():
    # long source gap: prev dub ended well before this source start
    placed, drift, _ = retime_step(
        src_start=20.0, prev_end=17.2, next_src_start=25.0,
        drift_max=1.5, min_gap=0.15)
    assert placed == 20.0 and drift == 0.0


def test_first_segment_gets_no_min_gap():
    placed, drift, _ = retime_step(
        src_start=0.0, prev_end=0.0, next_src_start=4.0,
        drift_max=1.5, min_gap=0.15)
    assert placed == 0.0 and drift == 0.0


def test_hard_end_caps_last_segment():
    _, _, slot = retime_step(
        src_start=58.0, prev_end=57.0, next_src_start=60.0,
        drift_max=1.5, min_gap=0.0, hard_end=60.0)
    assert abs(slot - 2.0) < 1e-9  # not 60 + drift_max


def test_slot_never_negative():
    _, _, slot = retime_step(
        src_start=10.0, prev_end=18.0, next_src_start=11.0,
        drift_max=1.5, min_gap=0.15)
    assert slot == 0.0
