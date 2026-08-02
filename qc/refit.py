"""qc-weight re-fit (AUTOPILOT.md M4 — the flywheel's payload).

`dubadabidu refit` reads the accumulated (human rating, qc metrics) pairs that
`dubadabidu verdicts` writes into ratings_<lang>.json and re-derives
qc.eval.weights: the weights that make qc_score agree with YOUR ear as closely
as the metrics allow. Each cycle the objective gets closer to your judgment ->
the tune loop optimizes a better target -> fewer flags reach you next video.

It PROPOSES, never applies. specs/batch.yaml lists "edit eval weights" under
`never`, and that rail is the right one: a scoring function that silently
rewrites itself between videos makes every cross-video number incomparable.
The output is a paste-ready YAML block plus the evidence for it.

Method (the 2026-07-08 procedure, automated):
  - objective = Spearman(qc_score, human rating). Rank correlation, not RMSE:
    the ratings are ordinal (1..5 stars) and only the ORDERING matters — the
    review page is sorted worst-first, and the flag threshold is a cut point.
  - scored through qc.metrics.composite_score itself, so the fit optimizes the
    REAL production objective rather than a re-derivation of it that could
    drift.
  - search = grid over the simplex at `step` resolution, weights constrained to
    sum to 1. That constraint is not cosmetic: score_flag (0.55) and the spec's
    mean_score_min (0.60) are cut points on this scale, so weights summing to
    anything else would silently move every threshold in the project.
  - adoption = the inherited invariant, applied to the weights themselves: a
    proposal must beat the incumbent OUT OF FOLD, not just in sample. Fitting 4
    free parameters on a few dozen ratings overfits happily, and an in-sample
    improvement alone is not evidence.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qc import metrics as X  # noqa: E402

log = logging.getLogger("dubadabidu.qc.refit")

# Features the composite consumes, and the row keys they come from.
NEEDED = ("qc_sim_cal", "qc_mos", "qc_f0st")
KEYS = ("sim", "mos", "f0", "tempo")
# Reported alongside the fit: the raw per-feature correlations. These are the
# diagnostic that drove the original hand-fit (mos +.63, f0 +.48, sim -.30) and
# they stay meaningful at any n, even when the weight fit itself is refused.
DIAGNOSTIC = ("qc_sim_cal", "qc_sim2", "qc_mos", "qc_mos_min", "qc_f0st",
              "qc_wer", "tempo", "qc_score")


# ---------- pure stats ----------

def _ranks(xs: list[float]) -> list[float]:
    """Average ranks, ties shared — 1..5 star ratings are mostly ties, and
    naive ordinal ranks would invent an ordering inside each tied group."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. 0.0 when either side is constant (no ordering to
    agree with) or there are too few points to mean anything."""
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    return round(num / (dx * dy) ** 0.5, 4) if dx and dy else 0.0


def simplex(step: float = 0.05, keys: tuple = KEYS) -> list[dict]:
    """All non-negative weightings of `keys` summing to 1, on a `step` lattice.
    Sum-to-1 keeps qc_score on the scale every existing threshold is cut on."""
    n = int(round(1.0 / step))
    out = []

    def walk(i: int, left: int, acc: list[int]):
        if i == len(keys) - 1:
            # rounded: v*step accumulates float noise (0.6000000000000001),
            # which leaks into the proposal printed for a human to paste
            out.append({k: round(v * step, 4)
                        for k, v in zip(keys, acc + [left])})
            return
        for v in range(left + 1):
            walk(i + 1, left - v, acc + [v])

    walk(0, n, [])
    return out


# ---------- fitting ----------

def usable(rows: list[dict]) -> list[dict]:
    """Rows that can train: a human rating plus every feature the composite
    needs. Sorted deterministically so folds (and reruns) are reproducible."""
    ok = [r for r in rows
          if isinstance(r.get("rating"), (int, float))
          and all(k in r for k in NEEDED)]
    return sorted(ok, key=lambda r: (r.get("video", ""), r.get("id", "")))


def predict(row: dict, w: dict, max_tempo: float) -> float:
    """qc_score for this row under weights `w` — through the production
    composite, so a proposal means in production exactly what it means here."""
    return X.composite_score(
        row["qc_sim_cal"], row["qc_mos"],
        X.tempo_penalty(row.get("tempo", 1.0), max_tempo), w, row["qc_f0st"])


def rho_of(rows: list[dict], w: dict, max_tempo: float) -> float:
    return spearman([predict(r, w, max_tempo) for r in rows],
                    [r["rating"] for r in rows])


def fit(rows: list[dict], max_tempo: float, step: float = 0.05) -> tuple:
    """Best weights on `rows` by Spearman. Ties break toward the SMALLEST
    change from a balanced weighting, so a flat objective (common at small n)
    doesn't return an arbitrary extreme point."""
    best, best_rho = None, -2.0
    for w in simplex(step):
        r = rho_of(rows, w, max_tempo)
        if r > best_rho or (r == best_rho and best is not None
                            and _spread(w) < _spread(best)):
            best, best_rho = w, r
    return best, best_rho


def _spread(w: dict) -> float:
    """Distance from an equal weighting — the tie-breaker's notion of 'least
    committed'."""
    even = 1.0 / len(w)
    return sum((v - even) ** 2 for v in w.values())


def cv_rho(rows: list[dict], max_tempo: float, k: int = 5,
           step: float = 0.05, fixed: dict | None = None) -> float:
    """Cross-validated Spearman. Predictions from every held-out fold are
    POOLED and ranked together — per-fold correlations on 5-10 points are far
    too noisy to average.

    `fixed` scores a given weighting instead of re-fitting per fold. Nothing is
    estimated from the data in that case, so its CV number equals its in-sample
    number; it is computed the same way purely so the two are comparable.
    """
    preds, actual = [], []
    for f in range(k):
        test = [r for i, r in enumerate(rows) if i % k == f]
        train = [r for i, r in enumerate(rows) if i % k != f]
        if not test or (fixed is None and len(train) < 3):
            continue
        w = fixed or fit(train, max_tempo, step)[0]
        preds += [predict(r, w, max_tempo) for r in test]
        actual += [r["rating"] for r in test]
    return spearman(preds, actual)


def permutation_p(rows: list[dict], max_tempo: float, observed: float,
                  k: int = 5, step: float = 0.05, n_perm: int = 50,
                  seed: int = 0) -> float:
    """How often the SAME fit-and-cross-validate procedure reaches `observed`
    on shuffled ratings. This is the gate that matters at these sample sizes.

    Comparing the proposal's CV score against the incumbent's is not enough on
    its own, and not because of sampling noise: the two numbers are produced
    differently. The incumbent is a fixed weighting, so on ratings that carry no
    signal it lands wherever chance puts it — including a strongly NEGATIVE
    correlation — while the fitted weights are pulled toward whatever scores
    best in-fold, which on noise is near zero. The difference is then positive by
    construction. Measured on pure-noise ratings: the incumbent scored -0.136
    and the re-fit -0.015, a '+0.121 improvement' that is entirely an artifact.
    Shuffling the labels and re-running the WHOLE procedure prices that artifact
    in, because the null goes through the same machinery.

    Deterministic (fixed seed): the same ratings file always yields the same p.
    Add-one smoothing keeps p strictly positive — with n_perm shuffles the
    smallest reportable p is 1/(n_perm+1), not 0.
    """
    import random
    rnd = random.Random(seed)
    ratings = [r["rating"] for r in rows]
    hits = 0
    for _ in range(n_perm):
        shuffled = ratings[:]
        rnd.shuffle(shuffled)
        perm = [{**r, "rating": s} for r, s in zip(rows, shuffled)]
        if cv_rho(perm, max_tempo, k, step) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ---------- CLI ----------

def _load(langs: list[str]) -> tuple[list[dict], dict]:
    rows, per_lang = [], {}
    for lang in langs:
        p = Path(f"ratings_{lang}.json")
        if not p.exists():
            per_lang[lang] = 0
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p} is not valid JSON ({e}).")
        if not isinstance(data, list):
            # A {key, ratings:{id:stars}} dict is the PRE-FLYWHEEL export
            # format: stars with no qc metrics attached, so nothing to fit
            # against. The one such file (ratings_test_en.json, 2026-07-08) was
            # deleted 2026-08-02, but the guard stays — an old export could
            # reappear, and silently skipping it would understate how much
            # evidence exists.
            raise SystemExit(
                f"{p} is the pre-flywheel format ({{key, ratings}}) with no qc "
                f"metrics per row — nothing to fit against. Re-review that "
                f"video and ingest with `dubadabidu verdicts`, or delete it.")
        for r in data:
            r.setdefault("lang", lang)
        rows += data
        per_lang[lang] = len(data)
    return rows, per_lang


def run(cfg: dict, langs: list[str]) -> int:
    ecfg = cfg["qc"].get("eval", {})
    cur = ecfg.get("weights", {"sim": 0.25, "mos": 0.40, "f0": 0.20,
                               "tempo": 0.15})
    rcfg = cfg["qc"].get("refit", {})
    min_rows = int(rcfg.get("min_rows", 30))
    step = float(rcfg.get("step", 0.05))
    k = int(rcfg.get("folds", 5))
    margin = float(rcfg.get("adopt_margin", 0.02))
    min_rho = float(rcfg.get("min_rho", 0.30))
    max_p = float(rcfg.get("max_p", 0.05))
    n_perm = int(rcfg.get("permutations", 50))
    max_tempo = cfg["fit"]["max_tempo"]

    raw, per_lang = _load(langs)
    rows = usable(raw)
    print(f"\n[refit] {len(raw)} accumulated rows "
          f"({', '.join(f'{l}:{n}' for l, n in per_lang.items())}), "
          f"{len(rows)} usable (rated + fully scored)")
    if not rows:
        print("  Nothing to fit. Rate segments on a review page, export, then "
              "`dubadabidu verdicts <video> <export.json>`.")
        return 1

    # per-feature correlations: the diagnostic that drove the original hand-fit.
    # Useful at ANY n, so it prints even when the weight fit is refused.
    ratings = [r["rating"] for r in rows]
    print(f"\n  per-feature Spearman vs your rating (n={len(rows)}):")
    for f in DIAGNOSTIC:
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        if len(vals) == len(rows):
            print(f"    {f:12} {spearman(vals, ratings):+.3f}")
        elif vals:
            print(f"    {f:12} {'—':>6}  ({len(rows) - len(vals)} rows lack it)")

    cur_rho = rho_of(rows, cur, max_tempo)
    print(f"\n  current weights {cur} -> Spearman {cur_rho:+.3f}")

    if len(rows) < min_rows:
        print(f"\n  !! REFUSING to propose weights: {len(rows)} ratings < "
              f"qc.refit.min_rows ({min_rows}).")
        print(f"     Four free parameters fit to {len(rows)} points will "
              f"describe the noise, not your ear. The correlations above are "
              f"still informative — keep rating segments and re-run.")
        return 1

    best, best_rho = fit(rows, max_tempo, step)
    cv_new = cv_rho(rows, max_tempo, k, step)
    cv_cur = cv_rho(rows, max_tempo, k, step, fixed=cur)
    print(f"  best-fit weights {best} -> Spearman {best_rho:+.3f} (in-sample)")
    print(f"\n  {k}-fold cross-validated:")
    print(f"    current  {cv_cur:+.3f}")
    print(f"    proposed {cv_new:+.3f}   (delta {cv_new - cv_cur:+.3f})")

    print(f"\n  permutation test ({n_perm} shuffles, ~{n_perm // 25 or 1}x the "
          f"time of one fit) ...", flush=True)
    p = permutation_p(rows, max_tempo, cv_new, k, step, n_perm)

    # Three independent gates, ALL required. Any one alone is defeatable:
    # a high CV rho can be luck, a low p can accompany a useless-but-consistent
    # objective, and beating the incumbent is confounded (see permutation_p).
    gates = [
        ("tracks your ear", cv_new >= min_rho,
         f"cross-validated rho {cv_new:+.3f} >= {min_rho}"),
        ("not chance", p <= max_p,
         f"permutation p {p:.3f} <= {max_p} ({n_perm} shuffles)"),
        ("beats incumbent", cv_new - cv_cur >= margin,
         f"delta {cv_new - cv_cur:+.3f} >= {margin}"),
    ]
    print("\n  adoption gates:")
    for label, ok, detail in gates:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label:16} {detail}")

    if not all(ok for _, ok, _ in gates):
        print(f"\n  VERDICT: KEEP CURRENT WEIGHTS — {len(rows)} ratings are not "
              f"yet enough to justify moving the objective. Keep rating; the "
              f"per-feature correlations above are the useful output for now.")
        return 0

    print("\n  VERDICT: ADOPT — the proposal clears all three gates.")
    print("  Paste into config.yaml under qc.eval (a human applies this — "
          "specs/batch.yaml forbids the agent editing eval weights):\n")
    print("```yaml")
    print("qc:")
    print("  eval:")
    print("    weights: {" + ", ".join(f"{k2}: {best[k2]:.2f}"
                                       for k2 in KEYS) + "}")
    print("```")
    print("  Then re-score every video, so old and new scores are never "
          "compared:\n    dubadabidu evaluate <each video>")
    return 0
