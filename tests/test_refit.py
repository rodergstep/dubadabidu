"""qc-weight re-fit (AUTOPILOT.md M4).

The re-fit moves the objective every downstream decision is made against, from
a handful of human ratings. The tests that matter are therefore the ones that
prove it REFUSES to move on evidence that isn't there.
"""
import json
import random

import pytest

from qc import refit as R
from qc import metrics as X

CUR = {"sim": 0.25, "mos": 0.40, "f0": 0.20, "tempo": 0.15}
TRUE = {"sim": 0.05, "mos": 0.60, "f0": 0.30, "tempo": 0.05}
MAX_TEMPO = 1.12


def _rows(n, seed, weights=None, noise=0.25):
    """Synthetic ratings. weights=None -> ratings are pure noise."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        r = {"video": "v", "id": f"u{i:04d}",
             "qc_sim_cal": rnd.uniform(0.2, 1.0),
             "qc_mos": rnd.uniform(2.5, 4.9),
             "qc_f0st": rnd.uniform(0.5, 4.0),
             "tempo": rnd.uniform(1.0, 1.12)}
        r["rating"] = (rnd.randint(1, 5) if weights is None else
                       max(1, min(5, round(1 + 4 * R.predict(r, weights,
                                                             MAX_TEMPO)
                                           + rnd.gauss(0, noise)))))
        out.append(r)
    return R.usable(out)


# --- rank correlation ---

def test_spearman_perfect_and_inverse():
    assert R.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert R.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_spearman_is_rank_based_not_linear():
    """Monotone but wildly non-linear must still be a perfect 1.0."""
    assert R.spearman([1, 2, 3, 4], [1, 2, 3, 4000]) == 1.0


def test_spearman_constant_side_is_zero_not_nan():
    assert R.spearman([1, 2, 3, 4], [3, 3, 3, 3]) == 0.0


def test_spearman_too_few_points():
    assert R.spearman([1, 2], [1, 2]) == 0.0


def test_ties_share_the_average_rank():
    """1..5 star ratings are mostly ties; ordinal ranks would invent an
    ordering inside each tied group."""
    assert R._ranks([5, 5, 5, 9]) == [2.0, 2.0, 2.0, 4.0]
    assert R._ranks([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0]


# --- search space ---

def test_simplex_weights_sum_to_one():
    """score_flag (0.55) and mean_score_min (0.60) are cut points on this
    scale — weights summing to anything else silently move every threshold."""
    pts = R.simplex(0.1)
    assert pts and all(abs(sum(w.values()) - 1.0) < 1e-9 for w in pts)


def test_simplex_covers_every_key():
    assert all(set(w) == set(R.KEYS) for w in R.simplex(0.25))


def test_simplex_is_deterministic():
    assert R.simplex(0.25) == R.simplex(0.25)


# --- row selection ---

def test_usable_drops_unrated_and_incomplete_rows():
    rows = [
        {"video": "v", "id": "a", "rating": 4, "qc_sim_cal": .8,
         "qc_mos": 4.0, "qc_f0st": 2.0},                    # keep
        {"video": "v", "id": "b", "verdict": "accept", "qc_sim_cal": .8,
         "qc_mos": 4.0, "qc_f0st": 2.0},                    # no rating
        {"video": "v", "id": "c", "rating": 3, "qc_mos": 4.0},   # no sim/f0
    ]
    assert [r["id"] for r in R.usable(rows)] == ["a"]


def test_usable_order_is_deterministic():
    """Folds are assigned by index, so a stable order makes CV reproducible."""
    rows = [{"video": v, "id": i, "rating": 3, "qc_sim_cal": .5,
             "qc_mos": 4.0, "qc_f0st": 2.0}
            for v, i in [("b", "u2"), ("a", "u9"), ("a", "u1")]]
    assert [(r["video"], r["id"]) for r in R.usable(rows)] == [
        ("a", "u1"), ("a", "u9"), ("b", "u2")]


def test_predict_goes_through_the_production_composite():
    """A proposal must mean in production exactly what it means here."""
    row = {"qc_sim_cal": 0.8, "qc_mos": 4.2, "qc_f0st": 2.5, "tempo": 1.05}
    assert R.predict(row, CUR, MAX_TEMPO) == X.composite_score(
        0.8, 4.2, X.tempo_penalty(1.05, MAX_TEMPO), CUR, 2.5)


# --- fitting ---

def test_fit_recovers_known_weights():
    best, rho = R.fit(_rows(150, 7, TRUE), MAX_TEMPO)
    assert rho > 0.7
    assert best["mos"] > best["sim"]      # the rater cared about mos, not sim
    assert best["f0"] > best["sim"]
    # weights come off a 0.05 lattice as v*step, so compare at lattice precision
    assert round(abs(best["mos"] - TRUE["mos"]), 6) <= 0.10


def test_fixed_weights_have_no_fold_leakage():
    """Nothing is estimated from the data, so CV must equal in-sample."""
    rows = _rows(60, 11, TRUE)
    assert R.cv_rho(rows, MAX_TEMPO, fixed=CUR) == R.rho_of(rows, CUR,
                                                            MAX_TEMPO)


def test_fit_overfits_in_sample_more_than_out_of_fold():
    """The gap is exactly why the gates use the cross-validated number."""
    rows = _rows(40, 5, None)             # noise: nothing real to learn
    _, in_sample = R.fit(rows, MAX_TEMPO)
    assert in_sample > R.cv_rho(rows, MAX_TEMPO)


# --- the guard that matters ---

@pytest.mark.parametrize("n,seed", [(35, 3), (60, 21)])
def test_noise_is_rejected_by_rho_and_permutation(n, seed):
    """THE regression guard. On ratings with no signal the incumbent-delta
    comparison is confounded and reads POSITIVE — the fitted weights drift
    toward zero correlation while a fixed weighting can sit at a spuriously
    negative one. Only the absolute floor and the permutation test catch it."""
    rows = _rows(n, seed, None)
    cv = R.cv_rho(rows, MAX_TEMPO)
    p = R.permutation_p(rows, MAX_TEMPO, cv, n_perm=15)
    assert cv < 0.30 or p > 0.05          # at least one real gate refuses


def test_permutation_p_is_bounded_and_deterministic():
    rows = _rows(30, 9, None)
    cv = R.cv_rho(rows, MAX_TEMPO)
    a = R.permutation_p(rows, MAX_TEMPO, cv, n_perm=10)
    b = R.permutation_p(rows, MAX_TEMPO, cv, n_perm=10)
    assert a == b                          # same ratings -> same p, always
    assert 0 < a <= 1.0                    # add-one smoothing: never exactly 0


def test_permutation_p_is_small_for_real_signal():
    rows = _rows(120, 7, TRUE)
    cv = R.cv_rho(rows, MAX_TEMPO)
    assert R.permutation_p(rows, MAX_TEMPO, cv, n_perm=15) <= 0.10


# --- ingest guards ---

def test_legacy_ratings_format_is_refused_not_silently_skipped(tmp_path,
                                                               monkeypatch):
    """ratings_test_en.json (2026-07-08) is {key, ratings:{id:stars}} with no
    qc metrics per row — unusable, and silently skipping it would understate
    how much evidence exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ratings_en.json").write_text(
        json.dumps({"key": "test_en", "ratings": {"u0001": 3}}))
    with pytest.raises(SystemExit, match="pre-flywheel format"):
        R._load(["en"])


def test_missing_ratings_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows, per_lang = R._load(["en", "fr"])
    assert rows == [] and per_lang == {"en": 0, "fr": 0}
