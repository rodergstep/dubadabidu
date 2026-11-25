# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project-root resolution (dubadabidu.core.project): --project, then
$DUBADABIDU_HOME, then the nearest config.yaml above the cwd, then the
source checkout — never site-packages."""
from __future__ import annotations

from pathlib import Path

import pytest

from dubadabidu.core.project import NoProject, resolve


def _proj(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text("work_dir: work\n", encoding="utf-8")
    return d


def test_flag_wins_over_everything(tmp_path: Path) -> None:
    a, b = _proj(tmp_path / "a"), _proj(tmp_path / "b")
    p = resolve(str(a), {"DUBADABIDU_HOME": str(b)}, b, checkout=b)
    assert (p.root, p.found_by) == (a.resolve(), "--project")


def test_env_wins_over_cwd(tmp_path: Path) -> None:
    a, b = _proj(tmp_path / "a"), _proj(tmp_path / "b")
    p = resolve(None, {"DUBADABIDU_HOME": str(a)}, b, checkout=b)
    assert (p.root, p.found_by) == (a.resolve(), "$DUBADABIDU_HOME")


def test_walks_up_from_a_subdirectory(tmp_path: Path) -> None:
    a = _proj(tmp_path / "a")
    deep = a / "input" / "lessons"
    deep.mkdir(parents=True)
    p = resolve(None, {}, deep, checkout=tmp_path / "nowhere")
    assert (p.root, p.found_by) == (a.resolve(), "cwd")


def test_falls_back_to_the_checkout(tmp_path: Path) -> None:
    clone = _proj(tmp_path / "clone")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    p = resolve(None, {}, elsewhere, checkout=clone)
    assert (p.root, p.found_by) == (clone.resolve(), "checkout")


def test_no_project_is_a_clean_error(tmp_path: Path) -> None:
    with pytest.raises(NoProject, match="--project"):
        resolve(None, {}, tmp_path, checkout=tmp_path / "site-packages")


def test_a_bad_flag_is_reported(tmp_path: Path) -> None:
    with pytest.raises(NoProject, match="not a directory"):
        resolve(str(tmp_path / "missing"), {}, tmp_path)
