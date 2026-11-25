# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the config machinery lives TODAY (moves edit only this file)."""
from __future__ import annotations

import glob
from collections.abc import Mapping
from pathlib import Path

from dubadabidu.lab import prep
from dubadabidu.infra.remote import runpod_infra
from dubadabidu.lab import tune
from dubadabidu.core.config.schema import Config, validate
from dubadabidu.core.logic import deep_merge

REPO = Path(__file__).resolve().parents[2]

__all__ = ["Config", "deep_merge", "validate"]


def overlay_files() -> list[str]:
    return sorted(glob.glob(str(REPO / "config*.yaml")))


def config_path(name: str) -> str:
    return str(REPO / name)


def defaults_tables() -> Mapping[str, Mapping[str, object]]:
    """Module dicts merged UNDER a config block ({**DEFAULTS, **cfg[block]})."""
    return {"runpod": runpod_infra.DEFAULTS, "tune": tune.DEFAULTS,
            "prep": prep.PREP_DEFAULTS}
