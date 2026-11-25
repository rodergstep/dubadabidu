# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repo paths named in the living docs exist. (MIGRATION_VAST.md and the
CHANGELOG describe the past on purpose and are not checked.)"""
from __future__ import annotations

import re

import pytest

from tests.paths import REPO

DOCS = ["README.md", "docs/ARCHITECTURE.md", "docs/AUTOPILOT.md"]
ROOTS = ("src/", "docs/", "configs/", "ledger/", "specs/", "tests/", "scripts/",
         "research/", "glossary/")
# `path`, or [text](path)
PATH = re.compile(r"`([^`\s]+)`|\]\(([^)\s#]+)\)")


def _paths(text: str) -> set[str]:
    out = set()
    for m in PATH.finditer(text):
        p = (m.group(1) or m.group(2)).split("::")[0].rstrip(".,:;")
        if p.startswith(ROOTS) and not any(c in p for c in "*<>{}"):
            out.add(p)
    return out


@pytest.mark.parametrize("doc", DOCS)
def test_named_paths_exist(doc: str) -> None:
    base = (REPO / doc).parent
    text = (REPO / doc).read_text(encoding="utf-8")
    missing = [p for p in sorted(_paths(text))
               if not (REPO / p).exists() and not (base / p).exists()]
    assert not missing, f"{doc} names paths that do not exist: {missing}"


def test_markdown_links_resolve() -> None:
    for doc in DOCS:
        base = (REPO / doc).parent
        for m in re.finditer(r"\]\(([^)\s#]+)\)", (REPO / doc).read_text(encoding="utf-8")):
            target = m.group(1)
            if "://" in target:
                continue
            assert (base / target).exists(), f"{doc}: broken link {target}"
