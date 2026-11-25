# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the config machinery lives TODAY (moves edit only this file)."""
from __future__ import annotations

from pathlib import Path

from dubadabidu.core.config.schema import Config, validate
from dubadabidu.core.logic import deep_merge

REPO = Path(__file__).resolve().parents[2]

__all__ = ["Config", "deep_merge", "validate"]


def overlay_files() -> list[str]:
    from dubadabidu.core.config.loader import overlay_files as found
    return sorted(str(p) for p in [REPO / "config.yaml", *found(REPO)])


def config_path(name: str) -> str:
    from dubadabidu.core.config.loader import overlay_path
    return str(overlay_path(REPO, name))
