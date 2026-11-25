# SPDX-License-Identifier: AGPL-3.0-or-later
"""Overlays are found by name in the project root, configs/ or
configs/experiments/ — so a basename may exist in only one of them, and every
overlay a command line, REMOTE_TASK or the experiment queue names resolves."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from dubadabidu.core.config.loader import SEARCH, overlay_files, overlay_path
from tests.paths import REPO


def test_overlay_basenames_are_unique_across_the_search_path() -> None:
    names = Counter(p.name for d in SEARCH for p in (REPO / d).glob("*.yaml"))
    dup = sorted(n for n, k in names.items() if k > 1)
    assert not dup, f"ambiguous overlay names (found in two search dirs): {dup}"


def test_every_overlay_name_the_code_and_specs_use_resolves() -> None:
    texts = [p.read_text(encoding="utf-8") for p in
             [*(REPO / "src").rglob("*.py"), *(REPO / "specs").rglob("*.yaml"),
              *REPO.glob("*.yaml"), *(REPO / "configs").rglob("*.yaml")]
             if p.exists()]
    named = {m for t in texts for m in re.findall(r"\bconfig\.[a-z0-9.-]+\.yaml\b", t)}
    named -= {"config.exp.06b.yaml"}       # a test fixture name, not a shipped file
    missing = sorted(n for n in named if not overlay_path(REPO, n).is_file())
    assert not missing, f"named but not found on the search path: {missing}"


def test_search_order_and_fallback(tmp_path: Path) -> None:
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    (tmp_path / "configs" / "experiments" / "config.exp.a.yaml").write_text("{}")
    (tmp_path / "configs" / "config.gpu.yaml").write_text("{}")
    assert overlay_path(tmp_path, "config.exp.a.yaml").parent.name == "experiments"
    assert overlay_path(tmp_path, "config.gpu.yaml").parent.name == "configs"
    assert overlay_path(tmp_path, "nope.yaml") == tmp_path / "nope.yaml"
    assert {p.name for p in overlay_files(tmp_path)} == {"config.exp.a.yaml", "config.gpu.yaml"}
