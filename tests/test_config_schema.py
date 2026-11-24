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
