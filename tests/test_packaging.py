# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the wheel ships: the dubadabidu package (every subpackage) and the
CLI entry module — and nothing from research/ or the pipeline/ and qc/ shims."""
from __future__ import annotations

import tomllib

from tests.paths import REPO


def _setuptools() -> dict[str, object]:
    with open(REPO / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["tool"]["setuptools"]


def test_every_subpackage_is_shipped() -> None:
    src = REPO / "src"
    found = {".".join(p.parent.relative_to(src).parts) for p in src.rglob("__init__.py")}
    st = _setuptools()
    listed = set(st.get("packages") or [])
    assert found == listed, f"pyproject packages out of date: missing {found - listed}, extra {listed - found}"


def test_the_cli_entry_point_is_shipped() -> None:
    with open(REPO / "pyproject.toml", "rb") as fh:
        script = tomllib.load(fh)["project"]["scripts"]["dubadabidu"]
    module = script.split(":")[0]
    st = _setuptools()
    shipped = set(st.get("py-modules") or []) | set(st.get("packages") or [])
    assert module in shipped or module.rsplit(".", 1)[0] in shipped, \
        f"console script {script} points at a module the wheel does not contain"


def test_research_and_shims_are_not_shipped() -> None:
    shipped = set(_setuptools().get("packages") or [])
    assert not {p for p in shipped if p.split(".")[0] in ("research", "pipeline", "qc")}
