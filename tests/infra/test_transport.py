# SPDX-License-Identifier: AGPL-3.0-or-later
"""ssh probes and rsync scope."""
from dubadabidu.core.config.schema import Runpod
from tests.paths import remote_source


def test_ready_probe_is_conservative(monkeypatch):
    """Any doubt re-runs setup: skipping wrongly costs a whole experiment,
    running needlessly costs ~50 s."""
    from dubadabidu.infra.remote import transport
    monkeypatch.setattr(transport, "ssh_exec", lambda *a, **k: 1)      # probe fails
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["qwen"]) is False
    # no engines to prove anything about -> do not skip
    monkeypatch.setattr(transport, "ssh_exec", lambda *a, **k: 0)
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", []) is False
    # an engine with no module mapping (chatterbox lives in the base venv)
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["edge"]) is False


def test_run_upload_sends_only_this_videos_workdir():
    """Uploading all of work/ shipped every other lesson (2.3 GB for a
    339 MB video) and was most of a 22.6 min bootstrap."""
    src = remote_source()
    assert ("_rsync_paths(rp, port, [str(M.video_workdir(cfg, v))\n"
            "                                    for v in videos], up)") in src
    assert 'extra_excludes=("input", "output"))' not in src


def test_ready_probe_checks_cuda_in_the_main_venv():
    """Skipping setup is only safe if the probe checks what setup guarantees.
    A CPU-only torch is silent — it cost a week of 126 s/take runs — so the
    probe must assert cuda availability, not merely that the module imports."""
    from dubadabidu.infra.remote import transport
    seen = {}
    R_ssh = transport.ssh_exec
    try:
        transport.ssh_exec = lambda rp, h, p, cmd, timeout=None: (
            seen.__setitem__("cmd", cmd) or 0)
        assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["qwen"]) is True
    finally:
        transport.ssh_exec = R_ssh
    assert ".venv/bin/python" in seen["cmd"]
    assert "torch.cuda.is_available()" in seen["cmd"]
    assert "import qwen_tts" in seen["cmd"]


def test_edge_alone_still_refuses_to_skip_setup(monkeypatch):
    """edge needs no install, so it proves nothing about the venv."""
    from dubadabidu.infra.remote import transport
    monkeypatch.setattr(transport, "ssh_exec", lambda *a, **k: 0)
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["edge"]) is False
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["edge", "qwen"]) is True
