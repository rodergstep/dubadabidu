# SPDX-License-Identifier: AGPL-3.0-or-later
"""The torch wheel, the image and the host CUDA filter are one decision."""
from dubadabidu.core.config.schema import validate
from tests.paths import REPO, config


def test_bootstrap_pins_torch_itself_now_that_chatterbox_is_gone():
    """torch/torchaudio used to arrive via the `clone` extra (chatterbox-tts),
    which also dragged diffusers, s3tokenizer, resemble-perth, conformer and
    spacy-pkuseg plus exact pins that made pip backtrack. chatterbox was removed
    2026-08-02, so the pin has to be ours or the pod gets whatever pip picks.

    2.14.1 is only safe because the driver floor guarantees a driver that
    supports it — see test_torch_pin_and_cuda_filter_move_together."""
    from dubadabidu.infra.remote.bootstrap import REMOTE_SETUP
    setup = REMOTE_SETUP.format(dir="~/d")
    assert "chatterbox" not in setup
    # v0.5: the pin lives in pyproject/uv.lock and the pod installs the lock
    assert "uv sync --frozen" in setup
    assert '"torch==2.14.1"' in open("pyproject.toml", encoding="utf-8").read()


def test_torch_pin_and_cuda_filter_move_together():
    """THE coupling that a first attempt at this bump got wrong, on a pod:
    torch 2.11+'s oldest build is cu126 (no 2.11+cu124 wheel exists), so on a
    default-scheduled host with a CUDA 12.4 driver it installs fine and then
    reports CUDA: False -- "The NVIDIA driver on your system is too old (found
    version 12040)". A silent CPU fallback at full GPU price is this project's
    most expensive recurring failure.

    So the pin and the host filter are ONE decision. Emptying
    allowed_cuda_versions without dropping torch back to 2.6.0 re-creates the
    exact failure, and nothing else in the repo would catch it before a pod."""
    import yaml
    from dubadabidu.core.logic import deep_merge
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8")))
    rp = validate(cfg).runpod
    setup = open("pyproject.toml", encoding="utf-8").read()
    import re
    m = re.search(r'"torch==(\d+)\.(\d+)\.\d+"', setup)
    assert m, "no torch pin found in pyproject.toml"
    major, minor = int(m.group(1)), int(m.group(2))
    # driver floor actually requested: v2 minCudaVersion, or min of an exact list
    allowed = ([float(rp.min_cuda_version)] if rp.min_cuda_version
               and not rp.allowed_cuda_versions else
               [float(v) for v in rp.allowed_cuda_versions or []])
    if (major, minor) > (2, 6):
        assert allowed, (
            f"torch {major}.{minor} needs a CUDA>=12.6 driver, but "
            f"allowed_cuda_versions is empty -> the scheduler may hand over a "
            f"12.4 host and CUDA silently goes False. Pin torch 2.6.0 or set "
            f"the filter.")
        assert min(allowed) >= 12.6, (
            f"allowed_cuda_versions {allowed} admits a driver older than 12.6, "
            f"which cannot run torch {major}.{minor} (oldest build is cu126)")


def test_torch_wheel_matches_the_oldest_allowed_driver():
    """The wheel's CUDA build must be <= the OLDEST driver the filter admits.

    Drivers are backward compatible with older CUDA runtimes but not forward:
    a cu130 wheel needs a 13.0 driver. Plain PyPI serves torch as +cu130, so
    with allowed_cuda_versions ["12.8","12.9","13.0"] two thirds of the
    permitted hosts could not run it — and the failure is the silent one
    (installs fine, CUDA: False, synthesis crawls on CPU at full GPU price).
    A pod on 2026-08-02 passed only because it drew a 13.0 driver.

    Hence the explicit cuXYZ index (cu130 since torch 2.14.1, under a 13.0
    floor). This test is the thing that notices when someone lowers the floor
    below the wheel's CUDA, or moves the wheel above the floor (e.g. to
    torch 2.15, which exists only as cu132)."""
    import re
    import yaml
    from dubadabidu.core.logic import deep_merge
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8")))
    rp = validate(cfg).runpod
    # driver floor actually requested: v2 minCudaVersion, or min of an exact list
    allowed = ([float(rp.min_cuda_version)] if rp.min_cuda_version
               and not rp.allowed_cuda_versions else
               [float(v) for v in rp.allowed_cuda_versions or []])
    # v0.5: the index is declared in pyproject [tool.uv.index] and the lock
    # records which build Linux resolves; both must name the same cuXYZ.
    setup = open("pyproject.toml", encoding="utf-8").read()
    lock = open("uv.lock", encoding="utf-8").read()
    m = re.search(r"download\.pytorch\.org/whl/cu(\d)(\d+)", setup)
    if m:
        assert f"+cu{m.group(1)}{m.group(2)}" in lock, "uv.lock is stale vs the index"
    if not allowed:
        return                      # no filter -> torch must be 2.6.0 anyway
    assert m, (
        f"allowed_cuda_versions is {allowed} but the torch install has no "
        f"explicit cuXYZ --index-url, so pip picks the default wheel (cu130 for "
        f"torch) which the oldest allowed driver may not support")
    wheel_cuda = float(f"{m.group(1)}.{m.group(2)}")
    assert wheel_cuda <= min(allowed), (
        f"torch wheel is cu{m.group(1)}{m.group(2)} ({wheel_cuda}) but the filter "
        f"admits drivers as old as {min(allowed)} — those hosts would install it "
        f"and report CUDA: False")


def test_qc_does_not_call_torchaudio_load():
    """torchaudio.load needs torchcodec from 2.9 on — and `hasattr(torchaudio,
    "load")` is still True, so the break shows up only when a scoring run
    actually reads a file, on a pod, mid-billing. Cheap to assert statically."""
    import ast
    root = REPO
    # AST, not a substring: the first cut of this test flagged the comment in
    # metrics.py that EXPLAINS the migration.
    for py in sorted((root / "src" / "dubadabidu").rglob("*.py")):
        for n in ast.walk(ast.parse(py.read_text())):
            called = n.func if isinstance(n, ast.Call) else None
            assert not (isinstance(called, ast.Attribute)
                        and called.attr == "load"
                        and getattr(called.value, "id", "") == "torchaudio"), (
                f"{py.name}:{n.lineno} calls torchaudio.load, which requires the "
                f"torchcodec package on torch>=2.9 — read with soundfile instead")


def test_runtime_cap_fits_a_full_hour_lesson():
    """max_runtime_hours is the pod-side watchdog's self-destruct, so a value
    below the real workload kills a run mid-flight AFTER paying for it.

    MEASURED 2026-08-02 on the first production lesson: 8.13 min of video x 5
    languages took 38.8 min of pod wall — 30.9 min synthesis (scales with
    content) + 7.8 min bootstrap (fixed). A 1-hour lesson therefore needs
    ~3.94 h, which the old 3.0 h cap would have cut off at ~76% done."""
    import yaml
    from dubadabidu.core.logic import deep_merge
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8")))
    rp = validate(cfg).runpod
    SYNTH_MIN_PER_LANG_PER_VIDEO_MIN = 30.9 / 5 / 8.13   # measured
    BOOTSTRAP_H = 7.8 / 60
    langs = len(cfg["languages"])
    need = 60.0 * SYNTH_MIN_PER_LANG_PER_VIDEO_MIN * langs / 60.0 + BOOTSTRAP_H
    assert rp.max_runtime_hours >= need, (
        f"max_runtime_hours={rp.max_runtime_hours} but a 1-hour lesson in "
        f"{langs} languages needs ~{need:.2f} h — the watchdog would kill the "
        f"pod mid-run, after billing for the whole thing")
