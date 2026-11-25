# SPDX-License-Identifier: AGPL-3.0-or-later
"""`dubadabidu migrate [video...] [--apply]`: rewrite pre-`text_src` files on disk.

Files written before the source language became configurable name the source
transcript `text_uk` (manifest.json, refs.json) and the source term `uk`
(terms_<lang>.json). The readers rename on load (manifest._rename_source_text,
s3.load_terms), so nothing breaks, but a file is only rewritten when a stage
happens to save it. This makes it explicit:

  - a dry run by default: it lists what would change and writes nothing;
  - --apply rewrites each file the way its own writer would have, so the
    bytes equal what the next save after a read would produce;
  - idempotent: a migrated file has nothing left to rename;
  - only work/<video>/ is touched, never ledger/.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from dubadabidu.core.config.schema import Config


@dataclass(frozen=True)
class Change:
    path: Path
    renamed: int          # rows whose old key was renamed


def _rename(rows: Iterable[object], old: str, new: str) -> int:
    n = 0
    for r in rows:
        if isinstance(r, dict) and old in r and new not in r:
            r[new] = r.pop(old)
            n += 1
    return n


def _dump(data: object) -> str:
    # what manifest.save, prep (refs.json) and s3 (terms_*.json) all write
    return json.dumps(data, ensure_ascii=False, indent=2)


def _migrate_dir(wd: Path, apply: bool) -> list[Change]:
    out: list[Change] = []
    targets: list[tuple[Path, str, str]] = [
        (wd / "manifest.json", "text_uk", "text_src"),
        (wd / "refs.json", "text_uk", "text_src"),
        *((p, "uk", "src") for p in sorted(wd.glob("terms_*.json"))),
    ]
    for path, old, new in targets:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "manifest.json":
            rows: list[object] = data.get("utterances", [])
        elif path.name == "refs.json":
            rows = list(data.values())
        else:
            rows = data.get("terms", [])
        n = _rename(rows, old, new)
        if not n:
            continue
        out.append(Change(path, n))
        if apply:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(_dump(data), encoding="utf-8")
            tmp.replace(path)
    return out


def workdirs(cfg: Config, videos: list[str] | None) -> list[Path]:
    """The work/<video>/ directories to look at: the given videos, or every
    directory under work_dir. Never creates one."""
    root = Path(cfg.work_dir)
    if videos:
        return [d for v in videos if (d := root / Path(v).stem).is_dir()]
    return sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []


def pending(cfg: Config) -> list[Change]:
    """What a dry run would report for all of work_dir (doctor uses this)."""
    return [c for wd in workdirs(cfg, None) for c in _migrate_dir(wd, apply=False)]


def run(cfg: Config, videos: list[str] | None, apply: bool) -> list[Change]:
    changes = [c for wd in workdirs(cfg, videos) for c in _migrate_dir(wd, apply)]
    verb = "renamed" if apply else "would rename"
    for c in changes:
        print(f"[migrate] {c.path}: {verb} {c.renamed} row(s)")
    if not changes:
        print("[migrate] nothing to migrate")
    elif not apply:
        print("[migrate] dry run: pass --apply to write these files")
    return changes

