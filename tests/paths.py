# SPDX-License-Identifier: AGPL-3.0-or-later
"""The repository root, in one place (tests used to recompute it per file)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dubadabidu"


def source(module: str) -> Path:
    """The file a module is loaded from (`source("dubadabidu.stages.s5_fit")`),
    so a test that reads code follows the import system, not a hard-coded path."""
    import importlib.util
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, f"no module {module}"
    return Path(spec.origin)


def config(name: str) -> Path:
    """A shipped config file by name, found the way `--overlay NAME` finds it
    (project root, configs/, configs/experiments/)."""
    from dubadabidu.core.config.loader import overlay_path
    return overlay_path(REPO, name)


def ledger_file(root: Path, name: str) -> Path:
    """`root`/ledger/`name`, with ledger/ created (a test's throwaway project)."""
    p = root / "ledger" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
