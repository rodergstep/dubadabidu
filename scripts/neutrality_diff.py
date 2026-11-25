#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the neutrality harness on two revisions and compare their output.

    python scripts/neutrality_diff.py [BASE] [--head REF] [--full]

BASE defaults to HEAD~1; the head defaults to the current working tree
(uncommitted changes included). Each revision runs ITS OWN harness, in its own
checkout (a temporary `git worktree` for anything that is not the working
tree), with this interpreter, so the comparison needs no stored golden and is
independent of the machine's ffmpeg / torch builds. Exit status 1 if anything
the pipeline wrote, asked the LLM, ran as a subprocess or printed differs.

Both revisions must contain tests/neutrality/harness.py (added in the commit
that introduced this script's harness).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def _dump(tree: Path, out: Path, full: bool) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tree / "src"), str(tree)])
    r = subprocess.run([sys.executable, "-m", "tests.neutrality.harness",
                        "--dump", str(out), *(["--full"] if full else [])],
                       cwd=tree, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"harness failed in {tree}:\n{r.stderr[-3000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def _compare(a: dict[str, object], b: dict[str, object]) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        if k == "fingerprint" or a.get(k) == b.get(k):
            continue
        av, bv = a.get(k), b.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            for kk in sorted(set(av) | set(bv), key=str):
                if av.get(kk) != bv.get(kk):
                    out.append(f"{k}[{kk!r}]")
        elif isinstance(av, list) and isinstance(bv, list):
            if len(av) != len(bv):
                out.append(f"{k}: {len(av)} -> {len(bv)} entries")
            out += [f"{k}[{i}]: {str(x)[:200]} -> {str(y)[:200]}"
                    for i, (x, y) in enumerate(zip(av, bv, strict=False)) if x != y][:5]
        else:
            out.append(f"{k}: {av!r} -> {bv!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="HEAD~1")
    ap.add_argument("--head", default=None,
                    help="a git ref; default: the working tree as it is")
    ap.add_argument("--full", action="store_true",
                    help="real ECAPA / Distill-MOS (needs work/.models/ecapa)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="dub-ndiff-") as tmp:
        t = Path(tmp)
        trees: list[tuple[str, Path]] = []
        for label, ref in (("base", a.base), ("head", a.head)):
            if ref is None:
                trees.append((label, REPO))
                continue
            wt = t / label
            _git("worktree", "add", "--detach", str(wt), ref)
            if a.full:   # the ECAPA cache is git-ignored; lend it to the worktree
                (wt / "work" / ".models").mkdir(parents=True)
                (wt / "work" / ".models" / "ecapa").symlink_to(
                    REPO / "work" / ".models" / "ecapa", target_is_directory=True)
            trees.append((label, wt))
        try:
            snaps = {label: _dump(tree, t / f"{label}.json", a.full)
                     for label, tree in trees}
        finally:
            for _label, tree in trees:
                if tree != REPO:
                    _git("worktree", "remove", "--force", str(tree))
    diffs = _compare(snaps["base"], snaps["head"])
    head = a.head or "working tree"
    if diffs:
        print(f"OUTPUT MOVED between {a.base} and {head}:")
        print("\n".join(f"  {d}" for d in diffs[:60]))
        sys.exit(1)
    n = len(snaps["head"].get("files") or {})
    print(f"neutral: {a.base} and {head} produce identical output "
          f"({n} files, {'full' if a.full else 'fast'} mode)")


if __name__ == "__main__":
    main()
