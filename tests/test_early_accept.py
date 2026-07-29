"""Early-accept gate (tts_engine.early_accept_ok).

Regression guard: accepting early leaves a ONE-take pool, and every downstream
filter in synth_best_of degrades to a no-op on a one-element pool
(`pool = filtered or pool`). So any floor the winner must clear has to be
checked HERE too. f0 was missing, which meant best_of_early_accept silently
bypassed tts.min_f0st — hardest on config.gpu.yaml (best_of: 5).
"""
from pipeline.tts_engine import early_accept_ok

GOOD = {"mos_min": 4.0, "dur": 5.0, "sim": 0.8, "f0st": 3.0, "wer": 0.0}


def _take(**over):
    return {**GOOD, **over}


def test_excellent_take_accepts():
    assert early_accept_ok(_take(), mos_floor=3.6, sim_floor=0.50)


def test_low_mos_rejects():
    assert not early_accept_ok(_take(mos_min=3.0), mos_floor=3.6, sim_floor=0.50)


def test_off_voice_rejects():
    assert not early_accept_ok(_take(sim=0.3), mos_floor=3.6, sim_floor=0.50)


def test_overflowing_take_rejects():
    # dur 5.0 vs target 4.0 -> 1.25x, past FIT_SLACK (1.10)
    assert not early_accept_ok(_take(), mos_floor=3.6, sim_floor=0.50,
                               target_dur=4.0)
    assert early_accept_ok(_take(), mos_floor=3.6, sim_floor=0.50,
                           target_dur=5.0)


def test_hallucinated_take_rejects():
    assert not early_accept_ok(_take(wer=0.5), mos_floor=3.6, sim_floor=0.50,
                               wer_max=0.15)


def test_flat_take_rejects_when_monotony_floor_is_set():
    """THE fix: a take that is excellent on every other axis but flatter than
    min_f0st must not end the roll — it would become a one-take pool and the
    monotony floor would never be applied."""
    flat = _take(f0st=1.2)
    assert not early_accept_ok(flat, mos_floor=3.6, sim_floor=0.50,
                               min_f0st=2.2)
    # ...and is accepted once the floor is off, so this only bites when the
    # user actually asked for a monotony floor
    assert early_accept_ok(flat, mos_floor=3.6, sim_floor=0.50, min_f0st=0.0)


def test_f0_floor_boundary_is_inclusive():
    at_floor = _take(f0st=2.2)
    assert early_accept_ok(at_floor, mos_floor=3.6, sim_floor=0.50,
                           min_f0st=2.2)


def test_missing_f0_treated_as_flat():
    """Unranked takes carry no f0st; with a floor set, they must not early-accept
    on absent evidence."""
    m = {"mos_min": 4.0, "dur": 5.0}
    assert not early_accept_ok(m, mos_floor=3.6, sim_floor=0.50, min_f0st=2.2)
    assert early_accept_ok(m, mos_floor=3.6, sim_floor=0.50)
