# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load-bearing comments travel with their code (ARCHITECTURE_PLAN.md §8 A7).

Every paragraph in tests/comment_census.json — a comment or docstring that
cites FINDINGS, a §, a measurement, a pin or a date — must still occur,
whitespace-normalised, somewhere in the codebase. A move may re-indent or
re-wrap it; it may not drop it. When a paragraph's text changes on purpose
(a path inside it moved), refresh the census in the same commit:
    python scripts/comment_census.py --write
so the diff shows exactly which recorded comment changed.
"""
from __future__ import annotations

import json
import sys

from tests.paths import REPO

sys.path.insert(0, str(REPO / "scripts"))
import comment_census  # noqa: E402


def test_every_load_bearing_comment_survives() -> None:
    want = json.loads(comment_census.CENSUS.read_text(encoding="utf-8"))
    corpus = comment_census.corpus()
    missing = [p for p in want if p not in corpus]
    assert not missing, (f"{len(missing)} load-bearing comment(s) vanished:\n  "
                         + "\n  ".join(m[:160] for m in missing[:10]))
