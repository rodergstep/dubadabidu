# SPDX-License-Identifier: AGPL-3.0-or-later
"""`dubadabidu migrate`: text_uk -> text_src on disk, dry run by default."""
from __future__ import annotations

import json
from pathlib import Path

from dubadabidu.core import manifest as M
from dubadabidu.workflows import migrate
from tests.cfgs import settings

LEGACY_MANIFEST = {
    "video": "input/l1.mp4", "duration": 4.0, "stages": {},
    "utterances": [
        {"id": "u0001", "start": 0, "end": 2, "text_uk": "Привіт", "tr": {}},
        {"id": "u0002", "start": 2, "end": 4, "text_src": "вже нове", "tr": {}},
    ]}
LEGACY_REFS = {"ref_01.wav": {"text_uk": "Привіт", "start": 0, "end": 2}}
LEGACY_TERMS = {"terms": [{"uk": "пензель", "tr": "brush"}]}


def _project(tmp: Path) -> tuple[Path, dict[str, str]]:
    wd = tmp / "work" / "l1"
    wd.mkdir(parents=True)
    files = {
        "manifest.json": json.dumps(LEGACY_MANIFEST, ensure_ascii=False, indent=2),
        "refs.json": json.dumps(LEGACY_REFS, ensure_ascii=False, indent=2),
        "terms_en.json": json.dumps(LEGACY_TERMS, ensure_ascii=False, indent=2),
    }
    for name, text in files.items():
        (wd / name).write_text(text, encoding="utf-8")
    ledger = tmp / "ledger"
    ledger.mkdir()
    (ledger / "runs.jsonl").write_text('{"text_uk": "kept"}\n', encoding="utf-8")
    return wd, files


def test_dry_run_lists_and_writes_nothing(tmp_path: Path) -> None:
    wd, files = _project(tmp_path)
    cfg = settings({"work_dir": str(tmp_path / "work")})
    changes = migrate.run(cfg, None, apply=False)
    assert sorted((c.path.name, c.renamed) for c in changes) == [
        ("manifest.json", 1), ("refs.json", 1), ("terms_en.json", 1)]
    for name, text in files.items():
        assert (wd / name).read_text(encoding="utf-8") == text
    assert len(migrate.pending(cfg)) == 3


def test_apply_writes_what_the_next_save_would(tmp_path: Path) -> None:
    wd, _ = _project(tmp_path)
    cfg = settings({"work_dir": str(tmp_path / "work")})
    # the bytes a read (with its rename shim) followed by a save produce
    man = M.load(cfg, "input/l1.mp4")
    want_manifest = json.dumps(man, ensure_ascii=False, indent=2)

    migrate.run(cfg, ["input/l1.mp4"], apply=True)
    assert (wd / "manifest.json").read_text(encoding="utf-8") == want_manifest
    refs = json.loads((wd / "refs.json").read_text(encoding="utf-8"))
    assert refs["ref_01.wav"]["text_src"] == "Привіт" and "text_uk" not in refs["ref_01.wav"]
    terms = json.loads((wd / "terms_en.json").read_text(encoding="utf-8"))["terms"]
    assert terms == [{"tr": "brush", "src": "пензель"}]
    assert not list(wd.glob("*.tmp"))


def test_idempotent_and_ledger_untouched(tmp_path: Path) -> None:
    _project(tmp_path)
    cfg = settings({"work_dir": str(tmp_path / "work")})
    migrate.run(cfg, None, apply=True)
    assert migrate.run(cfg, None, apply=True) == []
    assert migrate.pending(cfg) == []
    assert (tmp_path / "ledger" / "runs.jsonl").read_text(encoding="utf-8") == \
        '{"text_uk": "kept"}\n'


def test_unknown_video_and_missing_work_dir_create_nothing(tmp_path: Path) -> None:
    cfg = settings({"work_dir": str(tmp_path / "work")})
    assert migrate.run(cfg, ["input/nope.mp4"], apply=True) == []
    assert not (tmp_path / "work").exists()
