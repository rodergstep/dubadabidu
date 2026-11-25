# SPDX-License-Identifier: AGPL-3.0-or-later
"""The refactor's safety net: the local pipeline's output must not move.

Runs tests/neutrality/harness.py in a fresh interpreter (the fakes replace
third-party modules in sys.modules, so they must not leak into other tests)
and compares its snapshot with the golden, section by section:

  fast  golden_fast.json  every model faked (deterministic stand-ins)
  full  golden_full.json  real ECAPA + Distill-MOS from local caches, so the
                          QC metrics themselves must stay bit-identical
                          (72 human-rated groups depend on them)

The golden is only meaningful on the toolchain that recorded it: audio bytes
depend on the ffmpeg build, and the fake model scores on torch/torchaudio. On
another toolchain the test SKIPS and says so; scripts/neutrality_diff.py is
the machine-independent check (it runs two git revisions side by side).

Regenerate ONLY when a change is meant to alter output, and say so in the
commit:
    python -m tests.neutrality.harness --dump tests/neutrality/golden_fast.json
    python -m tests.neutrality.harness --full --dump tests/neutrality/golden_full.json
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
ECAPA_CACHE = REPO / "work" / ".models" / "ecapa"

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


@pytest.mark.parametrize("mode", ["fast", "full"])
def test_local_pipeline_output_is_unchanged(mode: str, tmp_path: Path) -> None:
    if mode == "full" and not (ECAPA_CACHE / "embedding_model.ckpt").exists():
        pytest.skip("full mode needs the ECAPA weights in work/.models/ecapa")
    want = json.loads((HERE / f"golden_{mode}.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO))
    from tests.neutrality.harness import fingerprint
    if fingerprint() != want["fingerprint"]:
        pytest.skip(f"golden recorded on {want['fingerprint']}, this machine is "
                    f"{fingerprint()} — use scripts/neutrality_diff.py")
    dump = tmp_path / "snap.json"
    r = subprocess.run([sys.executable, "-m", "tests.neutrality.harness",
                        "--dump", str(dump), *(["--full"] if mode == "full" else [])],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-4000:]
    got = json.loads(dump.read_text(encoding="utf-8"))
    problems = [p for k in sorted(set(got) | set(want))
                for p in _diff(k, got.get(k), want.get(k))]
    assert not problems, "output moved:\n  " + "\n  ".join(problems[:40])
