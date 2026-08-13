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
# qc_mos_min, not qc_mos: qc.evaluate feeds the WINDOWED MINIMUM into
# composite_score since 2026-08-11, and this module exists to fit the real
# production objective rather than a re-derivation of it. Convenient side
# effect — qc_mos_min was written correctly by every page all along, so the
# provenance split that made qc_mos unfittable does not apply to it.
NEEDED = ("qc_sim_cal", "qc_mos_min", "qc_f0st")
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

def mos_provenance_warning(rows: list[dict]) -> str | None:
    """Detect rows whose `qc_mos` is not the same METRIC as everyone else's.

    qc/blind.py wrote `mos_min_window` into BOTH `qc_mos` and `qc_mos_min`
    until 2026-08-11, while qc/evaluate writes whole-take MOS into the first
    and the worst-3s window into the second. So a pooled file can hold two
    different measurements under one column name, and refit fits on that
    column. Measured on ratings_ru.json: review-page rows averaged 4.33 and
    variant-page rows 2.59 — a 1.74 gap on a pooled sd of 0.91, entirely an
    artefact of which page produced the row.

    A fit across that is partly fitting PROVENANCE, which is the kind of
    confound the permutation test cannot price in because it shuffles ratings,
    not features. Cheap, structural tell: a row where qc_mos == qc_mos_min
    exactly is almost certainly one of the collapsed ones (a genuine tie needs
    the worst window to equal the whole-take mean).
    """
    both = [r for r in rows if "qc_mos" in r and "qc_mos_min" in r]
    if len(both) < 10:
        return None
    # Prefer the EXPLICIT stamp. Value equality is only a fallback for rows
    # written before the stamp existed, and it over-counts: mos_min_window
    # falls back to mos() on clips shorter than its 3 s window, so 11 correct
    # review-page rows tie legitimately and would be misread as collapsed.
    if any("qc_mos_kind" in r for r in both):
        collapsed = [r for r in both if r.get("qc_mos_kind") == "window_min"]
    else:
        collapsed = [r for r in both if abs(r["qc_mos"] - r["qc_mos_min"]) < 1e-9]
    if not collapsed:
        return None
    frac = len(collapsed) / len(both)
    if not 0.05 < frac < 0.95:      # all-or-nothing is at least self-consistent
        return None
    import statistics as st
    a = [r["qc_mos"] for r in collapsed]
    b = [r["qc_mos"] for r in both if r not in collapsed]
    gap = abs(st.mean(a) - st.mean(b))
    sd = st.pstdev([r["qc_mos"] for r in both]) or 1e-9
    return (
        f"MIXED qc_mos PROVENANCE: {len(collapsed)}/{len(both)} rows have "
        f"qc_mos == qc_mos_min exactly, which is how qc/blind.py wrote rows "
        f"before 2026-08-11 (the windowed minimum in both columns). Their "
        f"qc_mos averages {st.mean(a):.2f} against {st.mean(b):.2f} for the "
        f"rest — a {gap:.2f} gap on a pooled sd of {sd:.2f}. Fitting across "
        f"them fits WHICH PAGE produced the row. Re-ingest those rows with a "
        f"current qc/blind.py, or fit the two groups separately.")


def length_confound_warning(rows: list[dict]) -> str | None:
    """Features that mostly track SEGMENT LENGTH rather than quality.

    Found the hard way on 2026-08-13, after a live objective change had already
    been made on the strength of a correlation this explains away.

    mos_min_window takes a MINIMUM over sliding windows, so a longer segment has
    more windows and a lower expected minimum by arithmetic alone. Measured on
    46 rated ru segments: duration vs qc_mos_min -0.719, vs qc_mos +0.854 (the
    other way), vs qc_sim_cal +0.548. Only qc_f0st is clean at +0.018.

    The listener also rates long segments worse — duration vs rating -0.304, and
    -0.500 against accept/reject — so ANY length-correlated feature scores a
    respectable rank correlation without measuring quality at all. qc_mos_min
    read +0.226 that way; inside an 8-16 s band it reverses to -0.276.

    qc/compare.py predicted exactly this in its header: "An absolute score on
    differing content confounds two variables — how good the take is, and how
    hard that sentence was. A 1.4 s fragment and a 14.9 s sentence are not on
    the same scale." That page exists to remove the confound and has never been
    used for a rating round.
    """
    have = [r for r in rows if isinstance(r.get("dur"), (int, float))]
    if len(have) < 20:
        return None
    dur = [r["dur"] for r in have]
    rat = [r["rating"] for r in have]
    hits = []
    for key, feat in (("qc_mos_min", "mos"), ("qc_mos", "mos"),
                      ("qc_sim_cal", "sim"), ("qc_f0st", "f0")):
        vals = [r.get(key) for r in have]
        if any(v is None for v in vals):
            continue
        rho_d = spearman(dur, vals)
        if abs(rho_d) >= 0.45:
            hits.append(f"{key} {rho_d:+.2f}")
    if not hits:
        return None
    return (
        f"LENGTH CONFOUND: {', '.join(hits)} vs segment duration, while the "
        f"ratings themselves run {spearman(dur, rat):+.2f} with duration. A "
        f"feature that tracks length will appear to predict the listener "
        f"without measuring quality. Weights fitted here encode 'this segment "
        f"is long', not 'this segment is bad' — use qc/compare.py, which holds "
        f"the sentence constant, before trusting a proposal.")


def constant_terms(rows: list[dict], max_tempo: float) -> list[str]:
    """Weight keys whose TERM has no variance across `rows`.

    A weight on such a term is UNIDENTIFIABLE: it cannot change any row's
    ranking, so Spearman is flat along that axis and the search returns an
    arbitrary point on a tie. It is not harmless. On the real ru file every row
    has tempo 1.0, so tempo_penalty is 0 everywhere and the grid happily
    returned `{sim: 0, mos: 0, f0: 0.05, tempo: 0.95}` — which is not "tempo
    matters most", it is "0.95 of the budget parked where it does nothing so f0
    can have the rest". Spearman is scale-invariant, so that ties with
    `{f0: 1.0}` and the tie-break picked whichever looked least committed.

    Pasting that into config would set a REAL 0.95 stretch penalty for the
    first segment that ever gets time-stretched — and, because `tempo` is also
    read as the pace-match reward in tts_engine._take_rank, would silently
    re-weight take selection too.
    """
    if len(rows) < 3:
        return []
    terms = {
        "sim": [r["qc_sim_cal"] for r in rows],
        "mos": [(r["qc_mos_min"] - 1.0) / 4.0 for r in rows],
        "f0": [min(1.0, r["qc_f0st"] / 4.0) for r in rows],
        "tempo": [X.tempo_penalty(r.get("tempo", 1.0), max_tempo) for r in rows],
    }
    return [k for k, v in terms.items() if max(v) - min(v) < 1e-9]


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
        row["qc_sim_cal"], row["qc_mos_min"],
        X.tempo_penalty(row.get("tempo", 1.0), max_tempo), w, row["qc_f0st"])


def rho_of(rows: list[dict], w: dict, max_tempo: float) -> float:
    return spearman([predict(r, w, max_tempo) for r in rows],
                    [r["rating"] for r in rows])


def fit(rows: list[dict], max_tempo: float, step: float = 0.05,
        hold: dict | None = None) -> tuple:
    """Best weights on `rows` by Spearman. Ties break toward the SMALLEST
    change from a balanced weighting, so a flat objective (common at small n)
    doesn't return an arbitrary extreme point.

    `hold` pins weights the data cannot identify (see constant_terms) at their
    incumbent values and searches only the rest, rescaled to the remaining
    budget so the total still sums to 1. Searching an axis the objective is
    flat along does not find a better fit — it finds an arbitrary point on a
    tie, and on the real ru file that was 0.95 parked on `tempo` purely so f0
    could take the remainder.
    """
    hold = hold or {}
    free = tuple(k for k in KEYS if k not in hold)
    budget = max(0.0, 1.0 - sum(hold.values()))
    best, best_rho = None, -2.0
    for part in simplex(step, free):
        w = {**hold, **{k: round(v * budget, 4) for k, v in part.items()}}
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
           step: float = 0.05, fixed: dict | None = None,
           hold: dict | None = None) -> float:
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
        w = fixed or fit(train, max_tempo, step, hold)[0]
        preds += [predict(r, w, max_tempo) for r in test]
        actual += [r["rating"] for r in test]
    return spearman(preds, actual)


def permutation_p(rows: list[dict], max_tempo: float, observed: float,
                  k: int = 5, step: float = 0.05, n_perm: int = 50,
                  seed: int = 0, hold: dict | None = None) -> float:
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
        if cv_rho(perm, max_tempo, k, step, hold=hold) >= observed:
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

    # DIAGNOSTIC ONLY since the switch to qc_mos_min. qc_mos still carries the
    # blind.py split, and it is worth saying so because it is printed in the
    # per-feature table above — but the fit no longer reads that column, so it
    # must neither gate adoption nor drop rows. qc_mos_min was written the same
    # way by every page all along, which is why the 20 rows that could not be
    # repaired are usable again.
    lenwarn = length_confound_warning(rows)
    if lenwarn:
        print(f"\n  !! {lenwarn}")
    fits_qc_mos = "qc_mos" in NEEDED
    warn = mos_provenance_warning(rows)
    if warn:
        print(f"\n  !! {warn}")
        print("     (diagnostic only — the fit reads qc_mos_min, which every "
              "page wrote the same way)" if not fits_qc_mos else "")

    cur_rho = rho_of(rows, cur, max_tempo)
    print(f"\n  current weights {cur} -> Spearman {cur_rho:+.3f}")

    if len(rows) < min_rows:
        print(f"\n  !! REFUSING to propose weights: {len(rows)} ratings < "
              f"qc.refit.min_rows ({min_rows}).")
        print(f"     Four free parameters fit to {len(rows)} points will "
              f"describe the noise, not your ear. The correlations above are "
              f"still informative — keep rating segments and re-run.")
        return 1

    flat = constant_terms(rows, max_tempo)
    if flat:
        print(f"\n  !! UNIDENTIFIABLE TERMS: {', '.join(flat)} — every row has "
              f"the same value, so the objective is FLAT along "
              f"{'those axes' if len(flat) > 1 else 'that axis'} and any weight "
              f"there is an arbitrary point on a tie. Weight parked on a "
              f"constant term is not evidence it matters; it is budget removed "
              f"from the terms that do.")

    hold = {kk: float(cur.get(kk, 0.0)) for kk in flat}
    best, best_rho = fit(rows, max_tempo, step, hold)
    cv_new = cv_rho(rows, max_tempo, k, step, hold=hold)
    cv_cur = cv_rho(rows, max_tempo, k, step, fixed=cur)
    parked = sum(best.get(kk, 0.0) for kk in flat)
    moved = sum(abs(best.get(kk, 0.0) - float(cur.get(kk, 0.0))) for kk in flat)
    note = (f"   <- {parked:.2f} of this is {'/'.join(flat)}, HELD at the "
            f"incumbent value because this data cannot fit it"
            if parked > 1e-9 else "")
    print(f"  best-fit weights {best} -> Spearman {best_rho:+.3f} (in-sample)"
          + note)
    print(f"\n  {k}-fold cross-validated:")
    print(f"    current  {cv_cur:+.3f}")
    print(f"    proposed {cv_new:+.3f}   (delta {cv_new - cv_cur:+.3f})")

    print(f"\n  permutation test ({n_perm} shuffles, ~{n_perm // 25 or 1}x the "
          f"time of one fit) ...", flush=True)
    p = permutation_p(rows, max_tempo, cv_new, k, step, n_perm, hold=hold)

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
        # A FOURTH GATE, and it is not optional. The permutation test shuffles
        # RATINGS, so it prices in a spurious rating-feature relationship but
        # is blind to a feature column that holds two different measurements.
        # Adopting weights fitted across mixed provenance would encode "which
        # page rated this" into the production objective.
        ("fitted mos is one metric", warn is None or not fits_qc_mos,
         "qc_mos_min: written identically by every page"
         if not fits_qc_mos else
         ("all rows measure qc_mos the same way" if warn is None
          else "mixed provenance — see the warning above")),
        # Adopting weight on a term the data cannot identify writes a number
        # into config that was never measured — and `tempo` in particular is
        # read by BOTH composite_score (as a stretch penalty) and
        # tts_engine._take_rank (as a pace-match reward), so it would move take
        # selection on no evidence at all.
        # The check is that a constant term was HELD at its incumbent value,
        # not that it is near zero: holding is the correct treatment, and
        # `tempo` legitimately keeps its 0.15. What must never happen is the
        # search MOVING it, because a move there is an arbitrary point on a tie.
        # A length-confounded feature will sail through the permutation test:
        # shuffling ratings cannot break a relationship that runs through a
        # third variable present in both.
        ("features measure quality, not length", lenwarn is None,
         "no feature tracks segment duration" if lenwarn is None
         else "see the length-confound warning above"),
        ("weights are identifiable", moved <= 1e-9,
         "constant terms held at their incumbent values" if moved <= 1e-9
         else f"the fit moved {'/'.join(flat)} by {moved:.2f} on flat data"),
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
