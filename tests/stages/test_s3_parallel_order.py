# SPDX-License-Identifier: AGPL-3.0-or-later
"""s3 translates languages in parallel; the manifest it writes must not depend
on which language finishes first (ARCHITECTURE_PLAN.md §10.2)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dubadabidu.stages import s3_translate as s3
from tests.cfgs import settings
from tests.neutrality import fakes

UTTERANCES = [{"id": f"u{i:04d}", "start": 2.0 * i, "end": 2.0 * i + 1.8,
               "text_src": f"речення номер {i} про ультрамарин", "tr": {}}
              for i in range(1, 6)]


class _SlowEnglish:
    """fakes.OpenAI, except every English request is held back so Russian
    finishes first whenever the two run at the same time."""

    def __init__(self, **_: object) -> None:
        self.models = fakes.OpenAI().models
        self.chat = self
        self.completions = self

    def create(self, **kw: object) -> object:
        messages = kw["messages"]
        if "English" in messages[0]["content"]:          # type: ignore[index]
            time.sleep(0.05)
        return fakes._Completions().create(**kw)         # type: ignore[arg-type]


def _translate(tmp: Path, workers: int, monkeypatch: pytest.MonkeyPatch) -> str:
    import openai
    monkeypatch.setattr(openai, "OpenAI", _SlowEnglish)
    monkeypatch.setenv("TRANSLATE_API_KEY", "fake-key")
    wd = tmp / "work" / "clip"
    wd.mkdir(parents=True)
    (wd / "manifest.json").write_text(json.dumps(
        {"video": "input/clip.mp4", "duration": 12.0,
         "stages": {"s1_extract": "done", "s2_transcribe": "done"},
         "utterances": UTTERANCES}, ensure_ascii=False), encoding="utf-8")
    cfg = settings({"work_dir": str(tmp / "work"),
                    "translation": {"lang_workers": workers, "batch_size": 2}})
    s3.run(cfg, "input/clip.mp4", ["en", "ru"])
    return (wd / "manifest.json").read_text(encoding="utf-8")


def test_parallel_languages_write_the_sequential_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    one = _translate(tmp_path / "one", 1, monkeypatch)
    two = _translate(tmp_path / "two", 2, monkeypatch)
    man = json.loads(two)
    assert list(man["utterances"][0]["tr"]) == ["en", "ru"]
    assert [k for k in man["stages"] if k.startswith("s3_")] == ["s3_en", "s3_ru"]
    assert list(man["translation_models"]) == ["en", "ru"]
    assert two == one


def test_keys_from_an_earlier_run_keep_their_place() -> None:
    d = {"ru": 1, "fr": 2, "en": 3}
    assert list(s3._ordered(d, ["fr"], ["en", "ru"])) == ["fr", "en", "ru"]
    assert list(s3._ordered({"x": 0, **d}, [], ["en", "ru"])) == ["en", "ru", "x", "fr"]
