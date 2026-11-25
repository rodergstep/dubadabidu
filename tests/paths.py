# SPDX-License-Identifier: AGPL-3.0-or-later
"""The repository root, in one place (tests used to recompute it per file)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
