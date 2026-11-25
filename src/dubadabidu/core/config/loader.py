# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where config files are found.

`--overlay NAME` resolves against the project root first, then configs/, then
configs/experiments/ (ARCHITECTURE_PLAN.md §7, Q5). That lets the overlays
leave the root without touching a single reference to them: command lines,
the pod-side REMOTE_TASK strings, specs/experiments.yaml and FINDINGS.md all
name overlays by basename, and every one of those names still resolves. A
basename must therefore exist in only one of these directories
(tests/test_config_layout.py).
"""
from __future__ import annotations

from pathlib import Path

SEARCH = (".", "configs", "configs/experiments")


def overlay_path(project_root: Path, name: str) -> Path:
    """The file an --overlay argument refers to. Absolute paths are taken as
    given; a name found nowhere resolves to the project root, so the error a
    typo produces names the path the user typed."""
    p = Path(name)
    if p.is_absolute():
        return p
    for d in SEARCH:
        cand = project_root / d / p
        if cand.is_file():
            return cand
    return project_root / p


def overlay_files(project_root: Path) -> list[Path]:
    """Every overlay the search path can see (config*.yaml at each level,
    except the base config.yaml itself)."""
    out: list[Path] = []
    for d in SEARCH:
        out += sorted((project_root / d).glob("config*.yaml"))
    return [p for p in out if not (p.name == "config.yaml" and p.parent == project_root)]
