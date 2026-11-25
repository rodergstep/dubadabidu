#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Census of load-bearing comments: the ones a move must carry along.

    python scripts/comment_census.py           # list today's census paragraphs
    python scripts/comment_census.py --write   # refresh tests/comment_census.json

A "load-bearing" comment records a measurement or a decision: it mentions
FINDINGS or a §, says PINNED / load-bearing / measured, or carries a date.
Comments are taken as PARAGRAPHS (consecutive # lines, or a docstring),
whitespace-normalised, so re-indenting or re-wrapping code during a move does
not count as losing one. tests/test_comment_census.py checks that every
recorded paragraph still occurs, verbatim modulo whitespace, somewhere in the
codebase's comments and docstrings.
"""
from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "tests" / "comment_census.json"
MARKER = re.compile(r"FINDINGS|§|PINNED|load-bearing|LOAD-BEARING|[Mm]easured|"
                    r"\b20\d\d-\d\d-\d\d\b")


def sources() -> list[Path]:
    roots = ["dub.py", "pipeline", "qc", "src", "research", "scripts", "docker"]
    out: list[Path] = []
    for r in roots:
        p = REPO / r
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(p.rglob("*.py")) + sorted(p.rglob("Dockerfile"))
    out += sorted(REPO.glob("config*.yaml")) + sorted((REPO / "configs").rglob("*.yaml")) \
        + sorted((REPO / "specs").rglob("*.yaml")) + [REPO / "pyproject.toml"]
    return [p for p in out if p.exists()]


def _norm(s: str) -> str:
    return " ".join(s.split())


def paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    block: list[str] = []
    last = -2

    def flush() -> None:
        if block:
            out.append(_norm(" ".join(block)))
            block.clear()

    if path.suffix == ".py":
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                if tok.start[0] != last + 1:
                    flush()
                block.append(tok.string.lstrip("#"))
                last = tok.start[0]
        flush()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=True)
                if doc:
                    out += [_norm(p) for p in doc.split("\n\n")]
    else:   # yaml / toml / Dockerfile: '#' comments, whole-line or trailing
        for i, line in enumerate(text.splitlines()):
            if "#" not in line:
                continue
            c = line.split("#", 1)[1]
            if i != last + 1 or not line.lstrip().startswith("#"):
                flush()
            block.append(c)
            last = i
        flush()
    return [p for p in out if p]


def corpus() -> str:
    return " ".join(p for f in sources() for p in paragraphs(f))


def census() -> list[str]:
    return sorted({p for f in sources() for p in paragraphs(f) if MARKER.search(p)})


def main() -> None:
    rows = census()
    if "--write" in sys.argv:
        CENSUS.write_text(json.dumps(rows, ensure_ascii=False, indent=0) + "\n",
                          encoding="utf-8")
        print(f"{len(rows)} paragraphs -> {CENSUS.relative_to(REPO)}")
        return
    print("\n".join(rows))


if __name__ == "__main__":
    main()
