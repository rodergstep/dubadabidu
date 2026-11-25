# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline validation of learned audio judges against the listener's picks
(PLAN_v0.5 P5).

FINDINGS §2.1p closed take ranking "until a metric hears what these do not":
over 72 same-sentence groups every existing metric agreed with the listener's
pick at chance. This scores a NEW judge on exactly that data and statistic —
argmax agreement inside a group, chance = mean 1/k — so its result is directly
comparable to the §2.1p table. Nothing here is fitted (the judge's axes are
used as-is), so the agreement is already out-of-sample and the null is exact:
the number of hits under chance is Poisson-binomial over the group sizes.

Bias controls the literature reports for these judges:
  - loudness: judges prefer louder audio, so every clip is RMS-matched to the
    same level before scoring (the same pure-gain shift as mix.segment_norm);
  - position: irrelevant for a per-clip scorer; a PAIRWISE judge added later
    must score both orders and average.

Run:  python -m research.judge            (needs `uv sync --extra judge`)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from dubadabidu.review.refit import chance_rate, load_comparisons
from dubadabidu.core import ledger

TARGET_DBFS = -20.0


class ClipScorer(Protocol):
    """A per-clip judge: one score per named axis, higher = better."""
    name: str

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]: ...


@dataclass
class GroupResult:
    lang: str
    build: str
    group: str
    k: int
    winner_id: str
    scores: dict[str, dict[str, float]]   # clip id -> axis -> score


def rms_match(wav: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Pure-gain shift to a fixed RMS level, so loudness cannot pick a winner."""
    mono = wav.mean(axis=1) if wav.ndim > 1 else wav
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    if rms <= 0:
        return mono.astype(np.float32)
    gain = 10 ** (target_dbfs / 20) / rms
    return np.clip(mono * gain, -1.0, 1.0).astype(np.float32)


def clip_path(row: dict, work_dir: Path = Path("work")) -> Path:
    """comparisons row id `seg:variant:take` -> the original bake-off take."""
    _seg, variant, take = row["id"].split(":", 2)
    return work_dir / row["video"] / "bakeoff" / "seg" / variant / row["lang"] / take


def poisson_binomial_tail(probs: list[float], hits: int) -> float:
    """P(X >= hits) for X = sum of independent Bernoulli(p_i). Exact (DP)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[hits:].sum())


def agreement(results: list[GroupResult], axis: str) -> tuple[int, int, float]:
    """(hits, groups, one-sided exact p vs chance) for argmax-of-axis."""
    hits = 0
    for g in results:
        best = max(g.scores, key=lambda cid: g.scores[cid][axis])
        hits += best == g.winner_id
    p = poisson_binomial_tail([1.0 / g.k for g in results], hits)
    return hits, len(results), p


class AudioboxAesthetics:
    """Meta Audiobox Aesthetics (CC-BY-4.0): PQ production quality, CE content
    enjoyment, CU content usefulness, PC production complexity."""
    name = "audiobox"

    def __init__(self) -> None:
        from audiobox_aesthetics.infer import initialize_predictor
        self._pred = initialize_predictor()

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        import torch
        t = torch.from_numpy(wav).unsqueeze(0)
        row = self._pred.forward([{"path": t, "sample_rate": sr}])[0]
        return {k: float(v) for k, v in row.items()}


def score_groups(scorer: ClipScorer, langs: list[str],
                 work_dir: Path = Path("work")) -> list[GroupResult]:
    rows = load_comparisons(langs)
    by: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["video"], r["lang"], r["build"], r["group"]), []).append(r)
    out: list[GroupResult] = []
    for (_video, lang, build, group), members in sorted(by.items()):
        winners = [m for m in members if m.get("winner")]
        if len(members) < 2 or len(winners) != 1:
            continue
        scores: dict[str, dict[str, float]] = {}
        for m in members:
            p = clip_path(m, work_dir)
            if not p.exists():
                raise SystemExit(f"rated take missing: {p}")
            wav, sr = sf.read(str(p), dtype="float32")
            scores[m["id"]] = scorer.score(rms_match(wav), sr)
        out.append(GroupResult(lang, build, group, len(members),
                               winners[0]["id"], scores))
    return out


def report(results: list[GroupResult], gate_p: float = 0.05) -> dict:
    axes = sorted(next(iter(results[0].scores.values()))) if results else []
    summary: dict = {}
    print(f"{'lang':5s} {'axis':4s} {'agree':>9s} {'rate':>6s} {'chance':>7s} {'p':>7s}")
    for lang in sorted({g.lang for g in results}):
        sub = [g for g in results if g.lang == lang]
        ch = chance_rate([[None] * g.k for g in sub])
        for axis in axes:
            hits, n, p = agreement(sub, axis)
            summary.setdefault(lang, {})[axis] = {
                "hits": hits, "groups": n, "rate": round(hits / n, 3),
                "chance": ch, "p": round(p, 4)}
            print(f"{lang:5s} {axis:4s} {hits:4d}/{n:<4d} {hits / n:6.3f} "
                  f"{ch:7.3f} {p:7.4f}")
    passing = [a for a in axes
               if all(summary[lang][a]["p"] < gate_p for lang in summary)]
    summary["_gate"] = {"p_max": gate_p, "axes_passing_every_language": passing}
    print(f"\ngate (p < {gate_p} in EVERY language): "
          + (", ".join(passing) if passing else "NONE PASS"))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--langs", default="en,ru")
    ap.add_argument("--out", default=str(ledger.path("judge_validation.json")))
    a = ap.parse_args()
    results = score_groups(AudioboxAesthetics(), a.langs.split(","))
    summary = report(results)
    Path(a.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
