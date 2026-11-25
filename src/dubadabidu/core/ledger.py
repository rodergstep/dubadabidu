# SPDX-License-Identifier: AGPL-3.0-or-later
"""The evidence ledger: append-only files the measurements and verdicts live in.

  runs.jsonl                 every pod run's cost and shape (core.runlog)
  ratings_<lang>.json        human verdicts per segment (review.verdicts, blind)
  comparisons_<lang>.json    same-sentence picks (review.compare)
  stress_lexicon_<lang>.json words the listener marked mis-stressed (verdicts)
  judge_validation.json      the refuted Audiobox judge's evidence (research)
  FINDINGS.md                the conclusions drawn from all of the above

They live in ledger/ under the project root (the CLI runs from there), never
deleted or rewritten by the code — only appended to or merged into.
"""
from __future__ import annotations

from pathlib import Path

DIR = Path("ledger")


def path(name: str) -> Path:
    """A ledger file, relative to the project root."""
    return DIR / name


def for_writing(name: str) -> Path:
    """A ledger file about to be written: ledger/ is created in a project
    that has none yet (a fresh project directory next to an installed wheel)."""
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
