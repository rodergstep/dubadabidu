# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed configs for tests: the shipped config.yaml (+ overlays) with a patch.

Stages read a validated Config, so a test that used to hand them a partial
dict builds a real one instead: settings({"work_dir": str(tmp)}) is
config.yaml with work_dir replaced — every other key at its shipped value,
validated like the CLI validates it.
"""
from __future__ import annotations

from collections.abc import Mapping

import yaml

from dubadabidu.core.config.schema import Config, Tts, validate
from dubadabidu.core.logic import deep_merge
from tests.paths import REPO, config

Patch = Mapping[str, object]


def raw(patch: Patch | None = None, *overlays: str) -> dict[str, object]:
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    for ov in overlays:
        cfg = deep_merge(cfg, yaml.safe_load(config(ov).read_text(encoding="utf-8")))
    return deep_merge(cfg, dict(patch or {}))


def settings(patch: Patch | None = None, *overlays: str) -> Config:
    return validate(raw(patch, *overlays))


def tts(**fields: object) -> Tts:
    return settings({"tts": fields}).tts
