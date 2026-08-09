"""Did mixing rating axes break refit?

The listener said on 2026-08-09 that he had been rating STRESS, not overall
quality, on at least one page — while every page's instruction said "the one you
would ship". If different rounds were judged on different axes, refit has been
fitting one weight vector to several different questions, which would depress
cross-validated rho no matter how good the metrics are.

That is a hypothesis, not an observation, and two boring explanations have to be
ruled out first:

  1. RANGE RESTRICTION. A round where 15 of 18 clips are rated 1 carries almost
     no rating variance, so its correlation is near-meaningless regardless of
     axis.
  2. SMALL n. Per-round samples are 8-46. Spearman on n=8 swings wildly.

Test: how much do rounds disagree, versus how much would they disagree by chance
if the ratings came from ONE consistent process? Permute round labels within a
language (keeping round sizes fixed) and compare the observed between-round
spread of per-round Spearman against that null. A spread inside the null means
the data cannot support the mixed-axis story.

Then the decision-relevant part: if pooling heterogeneous rounds is what hurts,
ranking ratings WITHIN round before fitting should recover signal. That removes
any per-round scale or axis offset while keeping the ordering each round
actually expressed.
"""
import json
import random
import sys
from collections import defaultdict
from statistics import pstdev

sys.path.insert(0, "/Users/diadumenoss/Documents/projects/dubadabidu")
from qc.refit import spearman, usable, cv_rho, fit, _ranks, KEYS

FEATS = ("qc_sim2", "qc_mos", "qc_f0st")
RNG = random.Random(20260809)


def per_round_rho(rows, feat):
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    out = {}
    for v, rs in by.items():
        if len(rs) < 6:
            continue
        rat = [r["rating"] for r in rs]
        vals = [r[feat] for r in rs]
        if len(set(rat)) > 1 and len(set(vals)) > 1:
            out[v] = (spearman(vals, rat), len(rs))
    return out


def spread(d):
    """n-weighted sd of the per-round correlations."""
    if len(d) < 2:
        return 0.0
    xs = [v for v, _ in d.values()]
    return pstdev(xs)


def main():
    for lang in ("en", "ru"):
        rows = json.load(open(f"ratings_{lang}.json", encoding="utf-8"))
        print(f"\n{'='*70}\n{lang.upper()}  n={len(rows)}")

        # --- 0. how much rating variance does each round even have? ---
        by = defaultdict(list)
        for r in rows:
            by[r["variant"]].append(r)
        print("\n  rating variance per round (a floor/ceiling effect makes a "
              "round uninformative\n  regardless of which axis it was judged on):")
        for v, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            rat = [r["rating"] for r in rs]
            mode_frac = max(rat.count(x) for x in set(rat)) / len(rat)
            flag = "  <-- FLOOR/CEILING" if mode_frac >= 0.6 else ""
            print(f"    {v:22} n={len(rs):3}  sd={pstdev(rat):.2f}  "
                  f"most-common={mode_frac:.0%}{flag}")

        # --- 1. do rounds disagree more than chance? ---
        print("\n  between-round disagreement vs a same-process null "
              "(1000 permutations):")
        labels = [r["variant"] for r in rows]
        for feat in FEATS:
            obs = spread(per_round_rho(rows, feat))
            null = []
            for _ in range(1000):
                sh = labels[:]
                RNG.shuffle(sh)
                perm = [{**r, "variant": s} for r, s in zip(rows, sh)]
                null.append(spread(per_round_rho(perm, feat)))
            p = sum(1 for x in null if x >= obs) / len(null)
            verdict = "rounds DISAGREE" if p < 0.05 else "inside chance"
            print(f"    {feat:9} observed sd {obs:.3f} | null median "
                  f"{sorted(null)[len(null)//2]:.3f} | p={p:.3f}  {verdict}")

        # --- 2. does removing round effects recover signal? ---
        u = usable(rows)
        max_tempo = max((r.get("tempo") or 1.0) for r in u)
        base_w, base_in = fit(u, max_tempo)
        base_cv = cv_rho(u, max_tempo)

        within = []
        for v, rs in by.items():
            rs = [r for r in usable(rs)]
            if len(rs) < 6 or len({r["rating"] for r in rs}) < 2:
                continue
            rk = _ranks([r["rating"] for r in rs])
            n = len(rk)
            for r, k in zip(rs, rk):
                within.append({**r, "rating": k / n})   # 0..1 within its round
        w_w, in_w = fit(within, max_tempo)
        cv_w = cv_rho(within, max_tempo)
        print(f"\n  refit POOLED        n={len(u):3}  in-sample {base_in:+.3f}  "
              f"cross-val {base_cv:+.3f}")
        print(f"       weights {dict(sorted(base_w.items()))}")
        print(f"  refit WITHIN-ROUND  n={len(within):3}  in-sample {in_w:+.3f}  "
              f"cross-val {cv_w:+.3f}")
        print(f"       weights {dict(sorted(w_w.items()))}")
        delta = cv_w - base_cv
        print(f"  -> removing round effects moves cross-val rho by {delta:+.3f}")


if __name__ == "__main__":
    main()
