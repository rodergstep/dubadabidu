# SPDX-License-Identifier: AGPL-3.0-or-later
"""PLAN_v0.5 P7: every shipped config combination validates, and a typo fails
loudly with its dotted path instead of being silently ignored."""
import copy

import pytest
import yaml

from dubadabidu.core.config.schema import ConfigError, validate
from dubadabidu.core.logic import deep_merge
from tests.paths import REPO, config


def _load(name: str) -> dict:
    with open(config(name), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _combos() -> list[tuple[str, ...]]:
    # anchored at REPO: parametrize runs at COLLECTION time, before the
    # conftest chdir, and a cwd-relative glob silently found no overlays
    out: list[tuple[str, ...]] = [("config.yaml",), ("config.yaml", "config.gpu.yaml")]
    from dubadabidu.core.config.loader import overlay_files
    for f in sorted(p.name for p in overlay_files(REPO)):
        if f != "config.gpu.yaml":
            out.append(("config.yaml", "config.gpu.yaml", f))
    return out


@pytest.mark.parametrize("files", _combos(), ids=lambda c: c[-1])
def test_every_shipped_combination_validates(files: tuple[str, ...]) -> None:
    cfg = _load(files[0])
    for f in files[1:]:
        cfg = deep_merge(cfg, _load(f))
    validate(cfg)


def test_a_misspelled_key_fails_with_its_path() -> None:
    cfg = copy.deepcopy(_load("config.yaml"))
    cfg["tts"]["best_off"] = 3
    with pytest.raises(ConfigError, match=r"tts\.best_off"):
        validate(cfg)


def test_a_wrong_enum_value_fails() -> None:
    cfg = copy.deepcopy(_load("config.yaml"))
    cfg["translation"]["response_format"] = "json"
    with pytest.raises(ConfigError, match=r"translation\.response_format"):
        validate(cfg)


def test_tune_round_r2_is_still_accepted() -> None:
    """R2 (the chatterbox cfg_weight x exaggeration grid) was removed as
    unreachable; an old overlay that still lists it must validate, and tune
    logs it as skipped rather than failing."""
    cfg = copy.deepcopy(_load("config.yaml"))
    cfg["tune"]["rounds"] = ["R1", "R2", "R3"]
    validate(cfg)
