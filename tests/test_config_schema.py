# SPDX-License-Identifier: AGPL-3.0-or-later
"""PLAN_v0.5 P7: every shipped config combination validates, and a typo fails
loudly with its dotted path instead of being silently ignored."""
import copy
import glob

import pytest
import yaml

from pipeline.config_schema import ConfigError, validate
from pipeline.logic import deep_merge


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _combos() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [("config.yaml",), ("config.yaml", "config.gpu.yaml")]
    for f in sorted(glob.glob("config.*.yaml")):
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


@pytest.mark.parametrize("block,key,src", [
    ("translation", "avoid_mis_stressed", "pipeline/s3_translate.py"),
])
def test_schema_default_matches_the_code_default(block: str, key: str,
                                                 src: str) -> None:
    """Stages read the raw dict, so the code's .get() default is the one that
    applies; the schema's declared default must not claim otherwise."""
    import re

    from pipeline.config_schema import Config
    model = Config.model_fields[block].annotation
    declared = model.model_fields[key].default
    code = open(src, encoding="utf-8").read()
    found = re.findall(rf'get\("{key}", (True|False)\)', code)
    assert found, f"{src} no longer reads {key} with a literal default"
    assert {repr(declared)} == set(found)
