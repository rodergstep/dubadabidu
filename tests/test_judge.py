# SPDX-License-Identifier: AGPL-3.0-or-later
"""qc/judge.py statistics (PLAN_v0.5 P5) — the null must be exact, and loudness
must not be able to pick a winner."""
import math

import numpy as np

from qc.judge import GroupResult, agreement, poisson_binomial_tail, rms_match


def test_poisson_binomial_matches_the_binomial_when_probs_are_equal() -> None:
    n, p, k = 10, 0.5, 7
    want = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
    assert abs(poisson_binomial_tail([p] * n, k) - want) < 1e-12
    assert poisson_binomial_tail([0.3, 0.5], 0) == 1.0


def test_rms_match_removes_level_differences() -> None:
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    quiet, loud = 0.01 * np.sin(440 * t), 0.5 * np.sin(440 * t)
    a, b = rms_match(quiet), rms_match(loud)
    assert abs(np.sqrt(np.mean(a**2)) - np.sqrt(np.mean(b**2))) < 1e-4


def test_agreement_counts_argmax_hits_against_the_marked_winner() -> None:
    g = GroupResult("ru", "b", "g0", 2, "x", {"x": {"PQ": 2.0}, "y": {"PQ": 1.0}})
    miss = GroupResult("ru", "b", "g1", 2, "y", {"x": {"PQ": 2.0}, "y": {"PQ": 1.0}})
    hits, n, p = agreement([g, miss], "PQ")
    assert (hits, n) == (1, 2) and abs(p - 0.75) < 1e-12
