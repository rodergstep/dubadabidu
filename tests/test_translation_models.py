# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model-name drift guards: doctor's endpoint check and the answering-model
record (PLAN_v0.5 P1)."""
from types import SimpleNamespace

from dubadabidu.stages import s3_translate as s3
from tests.cfgs import settings
from tests.paths import config


class _Models:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self._ids])


class _Completions:
    def __init__(self, answered_by: str) -> None:
        self.answered_by = answered_by

    def create(self, **_: object) -> SimpleNamespace:
        msg = SimpleNamespace(content='{"segments": []}')
        return SimpleNamespace(model=self.answered_by,
                               choices=[SimpleNamespace(message=msg)])


def _client(ids: list[str], answered_by: str = "x") -> SimpleNamespace:
    return SimpleNamespace(models=_Models(ids),
                           chat=SimpleNamespace(completions=_Completions(answered_by)))


def test_configured_models_collects_every_pass_and_dedups() -> None:
    tcfg = settings({"translation": {"model": "a", "reflect_model": "b",
                                     "adequacy_model": "a"}}).translation
    assert s3.configured_models(tcfg) == ["a", "b"]


def test_missing_models_flags_a_retired_name() -> None:
    tcfg = settings({"translation": {"model": "deepseek-v4-flash",
                                     "reflect_model": "deepseek-v4-pro"}}).translation
    client = _client(["deepseek-flash", "deepseek-v4-pro"])
    assert s3.missing_models(client, tcfg) == ["deepseek-v4-flash"]


def test_missing_models_empty_when_all_listed() -> None:
    client = _client(["deepseek-flash"])
    assert s3.missing_models(client, settings({"translation": {"model": "deepseek-flash"}}).translation) == []


def test_chat_records_the_model_that_answered() -> None:
    seen: set[str] = set()
    client = _client([], answered_by="DeepSeek-V4.1-Flash-1017")
    tcfg = settings({"translation": {"model": "deepseek-flash"}}).translation
    s3._chat(client, tcfg, "sys", "user", seen=seen)
    assert seen == {"DeepSeek-V4.1-Flash-1017"}


def test_shipped_config_uses_the_current_deepseek_alias() -> None:
    import yaml
    for f in ("config.yaml", "config.deepseek.yaml"):
        with open(config(f), encoding="utf-8") as fh:
            assert yaml.safe_load(fh)["translation"]["model"] == "deepseek-flash", f
