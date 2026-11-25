# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where each golden-checked function lives TODAY.

The golden values in tests/golden/*.json never change during the refactor;
this adapter does. When a module moves, the move's commit edits only the
import below, so the diff shows exactly where a function went and the test
proves it still returns the same thing.
"""
from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import dub
from pipeline import manifest, runpod_infra, s3_translate
from pipeline.logic import deep_merge
from qc.bakeoff import variant_key

REPO = Path(__file__).resolve().parents[2]

__all__ = ["deep_merge", "variant_key"]

synth_hash = manifest.synth_hash
resolve_engine = manifest.resolve_engine
resolve_qwen_model = manifest.resolve_qwen_model
apt_setup = runpod_infra.apt_setup
engine_install_cmd = runpod_infra.engine_install_cmd
pod_payload = runpod_infra.pod_payload
terms_block = s3_translate.terms_block
cli_main = dub.main


def config_path(name: str) -> str:
    return str(REPO / name)


def overlay_files() -> list[str]:
    return sorted(glob.glob(str(REPO / "config*.yaml")))


def render_prompt(src: str, lang: str, tol: float, nvar: int, name: str) -> str:
    return s3_translate._prompt(src, lang, tol, nvar, name)


def glossary_block(src: str, lang: str) -> str:
    return s3_translate._glossary(src, lang)


def runpod_defaults() -> dict[str, object]:
    return dict(runpod_infra.DEFAULTS)


def remote_setup(remote_dir: str) -> str:
    return runpod_infra.REMOTE_SETUP.format(dir=remote_dir)


def remote_tasks() -> Mapping[str, str]:
    return runpod_infra.REMOTE_TASK


def watchdog_command(rp: Mapping[str, object], seconds: float) -> str:
    """The shell command arm_pod_watchdog sends to the pod."""
    with mock.patch.object(runpod_infra, "ssh_exec", return_value=0) as ssh:
        runpod_infra.arm_pod_watchdog(dict(rp), "203.0.113.7", 22022, seconds)
    return str(ssh.call_args.args[3])


def stage_order() -> list[str]:
    return list(dub.ORDER)


def remote_plan(ns: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    return dub._remote_plan(ns)


def remote_stages(ns: argparse.Namespace) -> str:
    return dub._remote_stages(ns)
