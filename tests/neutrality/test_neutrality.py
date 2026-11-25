# SPDX-License-Identifier: AGPL-3.0-or-later
"""The refactor's safety net: the local pipeline's output must not move.

Runs tests/neutrality/harness.py in a fresh interpreter (the fakes replace
third-party modules in sys.modules, so they must not leak into other tests)
and compares its snapshot with golden_fast.json, section by section.

The golden is only meaningful on the toolchain that recorded it: audio bytes
depend on the ffmpeg build, and the fake model scores on torch/torchaudio. On
another toolchain the test SKIPS and says so; scripts/neutrality_diff.py is
the machine-independent check (it runs two git revisions side by side).

Regenerate ONLY when a change is meant to alter output, and say so in the
commit:  python -m tests.neutrality.harness --dump tests/neutrality/golden_fast.json
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDEN = HERE / "golden_fast.json"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or any(importlib.util.find_spec(m) is None
           for m in ("torch", "torchaudio", "librosa", "pedalboard", "openai")),
    reason="needs ffmpeg + the local audio stack (runs on the dev Mac, not CI)")


def _diff(name: str, got: object, want: object) -> list[str]:
    if got == want:
        return []
    if isinstance(got, dict) and isinstance(want, dict):
        out = []
        for k in sorted(set(got) | set(want), key=str):
            if got.get(k) != want.get(k):
                out.append(f"{name}[{k!r}]: {str(want.get(k))[:160]!r} -> "
                           f"{str(got.get(k))[:160]!r}")
        return out
    if isinstance(got, list) and isinstance(want, list):
        out = [f"{name}: {len(want)} -> {len(got)} entries"] if len(got) != len(want) else []
        for i, (g, w) in enumerate(zip(got, want, strict=False)):
            if g != w:
                out.append(f"{name}[{i}]: {str(w)[:160]!r} -> {str(g)[:160]!r}")
                break
        return out
    return [f"{name}: {want!r} -> {got!r}"]


def test_local_pipeline_output_is_unchanged(tmp_path: Path) -> None:
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO))
    from tests.neutrality.harness import fingerprint
    if fingerprint() != want["fingerprint"]:
        pytest.skip(f"golden recorded on {want['fingerprint']}, this machine is "
                    f"{fingerprint()} — use scripts/neutrality_diff.py")
    dump = tmp_path / "snap.json"
    r = subprocess.run([sys.executable, "-m", "tests.neutrality.harness",
                        "--dump", str(dump)], cwd=REPO, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-4000:]
    got = json.loads(dump.read_text(encoding="utf-8"))
    problems = [p for k in sorted(set(got) | set(want))
                for p in _diff(k, got.get(k), want.get(k))]
    assert not problems, "output moved:\n  " + "\n  ".join(problems[:40])
