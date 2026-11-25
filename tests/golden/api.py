# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where each golden-checked function lives TODAY.

The golden values in tests/golden/*.json never change during the refactor;
this adapter does. When a module moves, the move's commit edits only the
import below, so the diff shows exactly where a function went and the test
proves it still returns the same thing.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

from dubadabidu import cli as dub
from dubadabidu.cli import remote
from dubadabidu.core import manifest
from dubadabidu.infra.remote import bootstrap, runpod, transport
from dubadabidu.stages import s3_translate
from dubadabidu.core.logic import deep_merge
from dubadabidu.lab.bakeoff import variant_key

REPO = Path(__file__).resolve().parents[2]

__all__ = ["deep_merge", "variant_key"]

synth_hash = manifest.synth_hash
resolve_engine = manifest.resolve_engine
resolve_qwen_model = manifest.resolve_qwen_model
apt_setup = bootstrap.apt_setup
engine_install_cmd = bootstrap.engine_install_cmd
pod_payload = runpod.pod_payload
terms_block = s3_translate.terms_block
cli_main = dub.main


def config_path(name: str) -> str:
    from dubadabidu.core.config.loader import overlay_path
    return str(overlay_path(REPO, name))


def overlay_files() -> list[str]:
    from dubadabidu.core.config.loader import overlay_files as found
    return sorted(str(p) for p in [REPO / "config.yaml", *found(REPO)])


def render_prompt(src: str, lang: str, tol: float, nvar: int, name: str) -> str:
    return s3_translate._prompt(src, lang, tol, nvar, name)


def glossary_block(src: str, lang: str) -> str:
    return s3_translate._glossary(src, lang)


def runpod_defaults() -> dict[str, object]:
    return dict(runpod.DEFAULTS)


def remote_setup(remote_dir: str) -> str:
    return bootstrap.REMOTE_SETUP.format(dir=remote_dir)


def remote_tasks() -> Mapping[str, str]:
    return bootstrap.REMOTE_TASK


def watchdog_command(rp: Mapping[str, object], seconds: float) -> str:
    """The shell command arm_pod_watchdog sends to the pod."""
    with mock.patch.object(transport, "ssh_exec", return_value=0) as ssh:
        runpod.arm_pod_watchdog(dict(rp), "203.0.113.7", 22022, seconds)
    return str(ssh.call_args.args[3])


def stage_order() -> list[str]:
    return list(dub.ORDER)


def remote_plan(ns: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    return remote._remote_plan(ns)


def remote_stages(ns: argparse.Namespace) -> str:
    return remote._remote_stages(ns)


def typed_tts(raw: Mapping[str, object]) -> object:
    """The typed tts block the CLI would hand on for this merged config."""
    from dubadabidu.core.config.schema import validate
    return validate(dict(raw)).tts


def derive(block: object, **updates: object) -> object:
    from dubadabidu.core.config.schema import derive as _derive
    return _derive(block, **updates)
