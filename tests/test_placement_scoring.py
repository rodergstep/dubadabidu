"""Pure-logic tests: choose_placement ladder and qc scoring helpers."""
from pipeline.logic import choose_placement
from qc.metrics import calibrate_sim, tempo_penalty, composite_score

MAX, SOFT = 1.12, 1.06


def test_primary_fits_as_is():
    assert choose_placement([4.0], 5.0, MAX, SOFT) == (0, "as_is", 1.0)


def test_primary_mild_stretch_preferred_over_variants():
    # 5.2/5.0 = 1.04 <= soft: keep the primary even though a variant fits as-is
    idx, verdict, tempo = choose_placement([5.2, 4.0], 5.0, MAX, SOFT)
    assert (idx, verdict) == (0, "stretch") and abs(tempo - 1.04) < 1e-9


def test_variant_as_is_beats_hard_stretch():
    # old first-fit stretched the primary to 1.10; new ladder takes the variant
    idx, verdict, tempo = choose_placement([5.5, 4.8], 5.0, MAX, SOFT)
    assert (idx, verdict, tempo) == (1, "as_is", 1.0)


def test_least_stretch_candidate_when_none_fit():
    idx, verdict, tempo = choose_placement([6.0, 5.5], 5.0, MAX, SOFT)
    assert idx == 1 and verdict == "stretch" and abs(tempo - 1.1) < 1e-9


def test_overflow_uses_last_candidate_at_max():
    idx, verdict, tempo = choose_placement([7.0, 6.5], 5.0, MAX, SOFT)
    assert (idx, verdict, tempo) == (1, "no", MAX)


def test_zero_slot_is_overflow():
    assert choose_placement([1.0], 0.0, MAX, SOFT) == (0, "no", MAX)


def test_calibrate_sim_band():
    assert calibrate_sim(0.65, 0.4, 0.9) == 0.5
    assert calibrate_sim(0.3, 0.4, 0.9) == 0.0    # below floor clips
    assert calibrate_sim(0.95, 0.4, 0.9) == 1.0   # above ceiling clips
    assert calibrate_sim(0.5, 0.6, 0.6) == 0.0    # degenerate band


def test_tempo_penalty():
    assert tempo_penalty(1.0, MAX) == 0.0
    assert abs(tempo_penalty(1.06, MAX) - 0.5) < 1e-9
    assert tempo_penalty(1.12, MAX) == 1.0
    assert tempo_penalty(2.0, MAX) == 1.0  # clipped


def test_composite_score_bounds_and_order():
    w = {"sim": 0.5, "mos": 0.35, "tempo": 0.15}
    perfect = composite_score(1.0, 5.0, 0.0, w)
    worst = composite_score(0.0, 1.0, 1.0, w)
    assert abs(perfect - 0.85) < 1e-9 and abs(worst - (-0.15)) < 1e-9
    better_sim = composite_score(0.8, 4.0, 0.0, w)
    worse_sim = composite_score(0.5, 4.0, 0.0, w)
    assert better_sim > worse_sim


def test_composite_score_f0_term():
    w = {"sim": 0.25, "mos": 0.40, "f0": 0.20, "tempo": 0.15}
    lively = composite_score(0.7, 4.0, 0.0, w, f0st=4.0)
    monotone = composite_score(0.7, 4.0, 0.0, w, f0st=1.0)
    assert abs((lively - monotone) - 0.20 * 0.75) < 1e-9
    assert composite_score(0.7, 4.0, 0.0, w, f0st=8.0) == lively  # f0 clipped at 4
    # weights without f0 key keep the old two-term behavior
    legacy = {"sim": 0.5, "mos": 0.35, "tempo": 0.15}
    assert composite_score(0.7, 4.0, 0.0, legacy, f0st=4.0) == \
        composite_score(0.7, 4.0, 0.0, legacy, f0st=0.0)
