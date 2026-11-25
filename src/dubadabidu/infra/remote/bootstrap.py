# SPDX-License-Identifier: AGPL-3.0-or-later
"""What runs ON the pod: bootstrap script, apt step, task commands,
engine install. Pinned byte-for-byte by tests/golden/pod_commands.json."""
from __future__ import annotations
import logging


log = logging.getLogger("dubadabidu.runpod")


# uv pinned on the pod to the version that wrote uv.lock (lock format drifts).
UV_VERSION = "0.8.15"


REMOTE_SETUP = (  # rsync/ffmpeg already installed in step 0
    "set -e; cd {dir}; "
    # Retries/timeout on the BOOTSTRAP too, not just the engine installs
    # (engine_install_cmd). This pip pulls ~2.5 GB of
    # CUDA torch — the largest download of the whole run — and on 2026-07-30 it
    # died on a bare `ReadTimeoutError ... files.pythonhosted.org`, leaving no
    # torch and failing the CUDA check 7 min in. It was the one big transfer left
    # without retry protection, because the engine path was hardened first simply
    # for having failed first.
    "export PIP_RETRIES=10 PIP_TIMEOUT=60; "
    # The old form was `python3 -m venv .venv 2>/dev/null || true`, which hid the
    # one failure that matters: an image whose python ships no venv module (the
    # ubuntu2404/py3.12 images need python3-venv). `set -e` then aborted one line
    # later on `activate`, with the actual cause discarded. Self-heal instead.
    # NOTE the doubled braces: REMOTE_SETUP is a .format() template, so a literal
    # shell `{ ...; }` group must be written {{ ...; }} or format() reads it as a
    # placeholder and raises KeyError. Caught by the tests, before a pod.
    # PREBUILT IMAGE (docker/Dockerfile, PLAN_v0.5 P3): link its venv in when
    # its stamp matches THIS project's uv.lock; anything else (no image, old
    # image, edited lock) falls through to the normal build below.
    "if [ ! -e .venv ] && [ -f /opt/venv/.uv-lock-sha256 ] && "
    "[ \"$(cat /opt/venv/.uv-lock-sha256)\" = \"$(sha256sum uv.lock | cut -c1-64)\" ]; "
    "then ln -s /opt/venv .venv; "
    "echo '[setup] prebuilt image venv matches uv.lock — linked'; fi; "
    "[ -x .venv/bin/python ] || "
    "python3 -m venv .venv || {{ echo '[setup] no venv module — installing'; "
    "apt-get install -y -q python3-venv && python3 -m venv .venv; }}; "
    ". .venv/bin/activate; "
    # UPGRADE PIP FIRST — the single biggest bootstrap win. Ubuntu 22.04's
    # python3.10-venv ships pip 22.0.2, whose resolver backtracks badly on
    # a heavily-pinned dependency graph.
    # Measured on a pod 2026-07-30: 26 min elapsed, 39 s of CPU, 3.3 GB pulled
    # and NOTHING installed — it was downloading 100-700 MB nvidia wheels,
    # reading their metadata, rejecting them and trying other versions. The link
    # itself did 8.45 MB/s to files.pythonhosted.org, so effective throughput was
    # ~25%: the cost was resolver churn, not bandwidth.
    # engine_install_cmd already did this; the bootstrap did not, purely because
    # the engine path happened to fail first and get hardened first.
    "pip install -q --upgrade pip; "
    # v0.5: ONE resolver run, from the lock. uv.lock pins every package
    # including torch 2.14.1+cu130 on Linux (pyproject [tool.uv.sources]), so
    # the pod installs exactly what was locked instead of whatever PyPI served
    # that morning — the failure behind the transformers-5.x incident. uv
    # lives INSIDE the venv (the ubuntu2404 system python is PEP-668 managed)
    # and `sync` is a fast no-op on a reused pod, so the old "skip if torch
    # imports" guard is gone. --inexact keeps what engine_setup added later
    # (qwen_tts, faster-qwen3-tts) on --reuse; --frozen refuses a stale lock.
    # torch pin + cu130 index are asserted from pyproject/uv.lock by
    # tests/test_runpod.py (the wheel must match the OLDEST allowed driver).
    "pip install -q 'uv==" + UV_VERSION + "'; "
    "UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=120 "
    "uv sync --frozen --inexact --extra dev --python .venv/bin/python; "
    # FAIL if CUDA is missing — otherwise the run would silently synth on CPU,
    # which is uselessly slow and defeats the point of renting a GPU
    "python -c 'import torch,sys; ok=torch.cuda.is_available(); "
    "print(\"CUDA:\",ok); sys.exit(0 if ok else 1)'")


REMOTE_TASK = {
    "bakeoff": "dubadabidu bakeoff {video} --langs {langs} "
               "--overlay config.gpu.yaml",
    # --no-mux: stops at s7 like `run`; the mux happens locally after sync-back
    "autopilot": "dubadabidu autopilot {video} --langs {langs} --no-mux "
                 "--overlay config.gpu.yaml --overlay config.deepseek.yaml",
    # stops at s7: the pod produces dubbed audio + subs; the final mux (a video
    # stream-copy needing the 4K source) runs LOCALLY after sync-back, so the
    # source video never uploads.
    # {stages} lets the caller narrow what the POD does. Only s4 needs a GPU:
    # separation, ASR, translation, fit, mix and subtitles all run fine on the
    # laptop for free, and translation in particular should NEVER run here —
    # it is API-latency-bound, so the pod would bill while waiting on DeepSeek.
    # Default keeps the old whole-pipeline behaviour.
    "run": "dubadabidu run {video} --langs {langs} {stages} --keep-going "
           "--overlay config.gpu.yaml --overlay config.deepseek.yaml",
    # preamble = prep's other half: tune R1 scores every ref candidate by real
    # ECAPA similarity on synthesized output and writes the winner (plus its
    # transcript, for ICL) into the manifest's tts_overrides. It SYNTHESIZES, so
    # it cannot run on the laptop with a GPU-only engine — and without this
    # entry `remote preamble` KeyError'd on REMOTE_TASK[task] AFTER provisioning
    # a pod. {stages} is unused here but kept so the format() call is uniform.
    "preamble": "dubadabidu preamble {video} --langs {langs} "
                "--overlay config.gpu.yaml --overlay config.deepseek.yaml",
}


# The base image (runpod/pytorch, Ubuntu) lacks rsync/ffmpeg — install them
# BEFORE the project sync (rsync-over-ssh needs rsync on the pod). The retry loop
# rides out the container's own boot-time apt holding the dpkg lock. Ubuntu's apt
# ffmpeg carries librubberband, so s5 auto-selects the rubberband stretcher
# (cleaner than atempo); probe it and log NON-fatally (atempo is a valid
# fallback, so a missing filter must not fail setup — the probe block exits 0).
#
# --no-install-recommends is LOAD-BEARING, not tidiness. Without it, installing
# ffmpeg on this (older) base image expanded to "10 upgraded, 143 newly
# installed" and the upgrades included the C/C++ runtime -- libgcc-s1,
# libstdc++6, gcc-12-base, plus libc-bin triggers. Swapping those out from under
# the running sshd killed it, and sshd is what keeps a RunPod box reachable: apt
# reported success and then the very next SSH connection was refused. That took
# down two pods in a row (2026-07-30), at the rsync on one and at REMOTE_SETUP on
# the other -- same step, different pods, so not spot preemption.
# The -s (simulate) pass logs apt's PLAN first: if core runtime libs ever appear
# in "[apt] plan:" again, this is why the pod died, and the fix is to stop using
# apt for ffmpeg (static build) rather than to guess.
def apt_setup(with_ffmpeg: bool = True) -> str:
    """Bootstrap command for the pod. with_ffmpeg=False installs ONLY rsync.

    ffmpeg is what drags in the ~128-package tree that upgrades the C runtime and
    kills sshd (see above). The BAKE-OFF never invokes the ffmpeg binary: it
    synthesizes through the engines and measures with soundfile / torchaudio /
    faster-whisper (PyAV), and even the side-by-side UA slices are cut by
    soundfile in review_page._ua_slice. s5's atempo/rubberband, s6's loudnorm,
    _synth_edge is the only ffmpeg user, and a bake-off
    runs none of them. So for bakeoff and setup-check we skip ffmpeg entirely and
    the failure mode disappears rather than being worked around.

    run/autopilot DO need it (s5/s6), so they still pass with_ffmpeg=True and
    remain exposed to the sshd kill. Fixing those needs a route that never lets
    apt touch the runtime — a static ffmpeg build dropped into /usr/local/bin, or
    a newer base image whose packages are already current. Not attempted here:
    one unvalidated change at a time.
    """
    # sox: qwen_tts imports the `sox` Python wrapper, which logs "SoX could
    # not be found!" without the binary (seen on every run 2026-11-24). The
    # clone path never calls it — takes were fine — so this only silences the
    # warning; it rides with ffmpeg so the bake-off's minimal apt is untouched.
    pkgs = "rsync ffmpeg sox" if with_ffmpeg else "rsync"
    tail = "which rsync"
    if with_ffmpeg:
        tail += (" && which ffmpeg && "
                 "{ ffmpeg -hide_banner -filters 2>/dev/null | grep -q rubberband "
                 "&& echo '[ffmpeg] rubberband filter present — s5 clean stretch' "
                 "|| echo '[ffmpeg] WARNING: no rubberband filter, s5 uses atempo'; }")
    return ("export DEBIAN_FRONTEND=noninteractive; "
            "for i in 1 2 3 4 5 6; do "
            "apt-get update && "
            "{ apt-get install -y --no-install-recommends -s " + pkgs +
            " | grep -E '^[0-9]+ upgraded' | sed 's/^/[apt] plan: /' || true; } && "
            "apt-get install -y --no-install-recommends " + pkgs + " && break; "
            "echo \"[apt] attempt $i failed (lock?), retrying in 10s\"; "
            "sleep 10; done; " + tail)


# faster-whisper short name -> import module, for the engines the bake-off/run
# may need on the pod. edge is CPU/PyPI (no git-clone) so it's not probed here.
ENGINE_MODULE = {"qwen": "qwen_tts"}


# installed into every engine venv alongside the snippet: soundfile because
# carry it. Deliberately minimal — the venv is the ENGINE's resolver's turf.


def engine_install_cmd(remote: str, engine: str, snippet: str) -> str:
    """Install `engine` into the MAIN venv.

    It used to get its own venvs/<engine>, because four engines had colliding
    pins. Three are gone and chatterbox took its torch pin with it, and the
    surviving constraint sets agree: faster-qwen3-tts wants torch>=2.5.1 (the
    base has 2.6.0), qwen-tts pins transformers==4.57.3 which nothing in the qc
    stack requires, and huggingface-hub<1.0 satisfies speechbrain>=0.8 and
    faster-whisper>=0.21. Isolation was therefore costing a SECOND ~2.5 GB torch
    download on every fresh pod — the largest single item in the bootstrap — to
    prevent a collision that can no longer happen.

    If a second cloning engine returns, revert rather than reinvent: the venv
    machinery (engine_client, engine_worker, the worker pool) is in git."""
    # PIP_CONSTRAINT pins the setuptools pip uses to BUILD wheels. Measured
    # 2026-07-30: CosyVoice's requirements include a source dist whose setup.py
    # does `import pkg_resources`, which current setuptools no longer ships, so
    # the build died with ModuleNotFoundError before anything downloaded.
    # Installing setuptools into the venv does NOT fix that — pip's build
    # isolation uses its own overlay env and ignores the venv — but PIP_CONSTRAINT
    # does apply to build dependencies. setuptools is ALSO installed in the venv
    # because some engines import pkg_resources at RUNTIME (chatterbox's `perth`
    # does), which isolation can't help with.
    # PIP_RETRIES/PIP_TIMEOUT: engine wheels are hundreds of MB (nvidia_cusolver
    # alone is 267 MB) and pod downloads were seen truncating mid-transfer.
    return (f"set -e; cd {remote}; "
            f". .venv/bin/activate; "
            f"printf 'setuptools<80\\n' > .venv/pip-constraint.txt; "
            f"export PIP_CONSTRAINT=$PWD/.venv/pip-constraint.txt; "
            f"export PIP_RETRIES=10 PIP_TIMEOUT=60; "
            f"pip install -q --upgrade pip; "
            f"pip install -q 'setuptools<80' wheel; "
            f"{snippet}")


def engines_for_task(cfg: dict, task: str, langs: list[str]) -> list[str]:
    """Which TTS engines this pod must end up with.

    ONE definition, because two callers must agree on it: _install_engines (what
    to install) and _verify_ready (what proves a --reuse pod is already set up).
    They used to disagree — the reuse probe was hardcoded to the BAKE-OFF's
    engine list and saw `[]` for every other task, and an empty list is False by
    design. `remote run --reuse` could therefore never skip setup, so course.py
    re-ran the bootstrap and re-installed qwen once per video: exactly the
    per-video cost its phase B exists to pay only once."""
    if task == "bakeoff":
        return list(dict.fromkeys(cfg.get("bakeoff", {}).get("engines", [])))
    tc = cfg.get("tts", {})
    ebl = tc.get("engine_by_lang") or {}
    return list(dict.fromkeys(ebl.get(lg, tc.get("engine")) for lg in langs))
