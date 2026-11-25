# SPDX-License-Identifier: AGPL-3.0-or-later
"""Golden values the refactor must not move (pure Python, runs in CI).

  cache keys      synth_hash, bakeoff.variant_key, engine / checkpoint routing
                  for every shipped overlay combination x language x text —
                  existing work/ caches stay valid only if these hold.
  prompts         the rendered s3 system prompts and glossary blocks, by
                  sha256, for every (source, target) pair.
  pod commands    REMOTE_SETUP, apt_setup, engine_install_cmd, REMOTE_TASK,
                  the watchdog command and the REST create body, verbatim —
                  the pod-side bootstrap must stay byte-identical.
  cli             `--help`, the parsed namespace for an argv matrix, and the
                  local/pod split of `remote run` for every --from/--to pair.

Values live in tests/golden/*.json. Regenerate ONLY for an intended change,
and say so in the commit:   DUB_REGEN_GOLDEN=1 pytest tests/golden
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from tests.golden import api
from tests import cfgs

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LANGS = ["en", "fr", "de", "es", "ru", "uk", "pl"]
TEXTS = ["Mix 25 ml of white.", "Сегодня 3 слоя, ё и е."]
JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


def _golden(name: str, value: JSONValue) -> None:
    path = HERE / f"{name}.json"
    text = json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if os.environ.get("DUB_REGEN_GOLDEN") == "1":
        path.write_text(text, encoding="utf-8")
        return
    want = json.loads(path.read_text(encoding="utf-8"))
    got = json.loads(text)
    if got != want:
        moved = [k for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)]
        pytest.fail(f"{name}: {len(moved)} golden entries moved, e.g. "
                    + "; ".join(f"{k}: {str(want.get(k))[:200]!r} -> "
                                f"{str(got.get(k))[:200]!r}" for k in moved[:3]))


def _load(path: str) -> dict[str, JSONValue]:
    with open(REPO / path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _combos() -> dict[str, dict[str, JSONValue]]:
    """Every shipped overlay combination, merged as the CLI merges them."""
    base = _load(api.config_path("config.yaml"))
    gpu = _load(api.config_path("config.gpu.yaml"))
    out = {"base": base, "base+gpu": api.deep_merge(base, gpu),
           "base+source-ru": api.deep_merge(base, _load(api.config_path("config.source-ru.yaml")))}
    for f in sorted(api.overlay_files()):
        name = Path(f).name
        if name in ("config.yaml", "config.gpu.yaml"):
            continue
        out[f"base+gpu+{name}"] = api.deep_merge(out["base+gpu"], _load(f))
    return out


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def test_cache_keys() -> None:
    keys: dict[str, JSONValue] = {}
    for label, raw in _combos().items():
        typed = api.typed_tts(raw)          # the validated, typed tts block
        for engine in (None, "edge"):
            tts = api.derive(typed, engine=engine) if engine else typed
            for lang in LANGS:
                for i, text in enumerate(TEXTS):
                    keys[f"{label}|{engine or '-'}|{lang}|{i}"] = api.synth_hash(text, lang, tts)
                keys[f"{label}|{engine or '-'}|{lang}|route"] = [
                    api.resolve_engine(tts, lang), api.resolve_qwen_model(tts, lang),
                    api.variant_key(api.resolve_engine(tts, lang), tts, lang)]
    _golden("cache_keys", keys)


def test_rendered_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    out: dict[str, JSONValue] = {}
    for src in ("uk", "ru"):
        for lang in LANGS:
            if lang == src:
                continue
            for name in ("translate_system", "translate_reflect", "translate_adapt"):
                text = api.render_prompt(src, lang, 0.10, 2, name)
                out[f"{name}|{src}|{lang}"] = [_sha(text), len(text)]
            g = api.glossary_block(src, lang)
            out[f"glossary|{src}|{lang}"] = [_sha(g), len(g)]
            cfg = cfgs.settings({"source_language": src})
            tb = api.terms_block(cfg, [{"src": "ультрамарин", "tr": "ultramarine"}])
            out[f"terms|{src}|{lang}"] = tb
    _golden("prompts", out)


def test_pod_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRANSLATE_API_KEY", raising=False)
    gpu = api.deep_merge(_load(api.config_path("config.yaml")),
                         _load(api.config_path("config.gpu.yaml")))
    key = tmp_path / "id_test"
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAAtest golden\n", encoding="utf-8")
    rp = {**api.runpod_defaults(), **gpu["runpod"], "ssh_key": str(key)}
    out: dict[str, JSONValue] = {
        "REMOTE_SETUP": api.remote_setup("~/dubadabidu"),
        "apt_setup(True)": api.apt_setup(True),
        "apt_setup(False)": api.apt_setup(False),
        "engine_install_cmd": api.engine_install_cmd(
            "~/dubadabidu", "qwen", rp["engine_setup"]["qwen"]),
        "pod_payload": api.pod_payload(rp, rp["gpu_type_ids"][0]),
        "watchdog": api.watchdog_command(rp, 3600.0),
    }
    for task, tpl in api.remote_tasks().items():
        out[f"REMOTE_TASK[{task}]"] = tpl.format(
            video="'input/a b.mp4'", langs="en,ru", stages="--from s4_synthesize --to s5_fit")
    _golden("pod_commands", out)


ARGVS = [
    ["run", "input/a.mp4"],
    ["run", "input/a.mp4", "--langs", "en,ru", "--from", "s4_synthesize", "--to", "s5_fit"],
    ["run", "a.mp4", "b.mp4", "--engine", "edge", "--force", "--keep-going", "-v"],
    ["stage", "s3_translate", "input/a.mp4", "--force"],
    ["remote", "run", "input/a.mp4", "--overlay", "config.gpu.yaml",
     "--overlay", "config.deepseek.yaml", "--budget", "5", "--reuse", "--keep-alive"],
    ["autopilot", "input/a.mp4", "--no-mux", "--spec", "specs/batch.yaml"],
    ["verdicts", "input/a.mp4", "export.json"],
    ["doctor", "--config", "other.yaml"],
]


def _capture_namespaces(entry: Callable[[], None], argvs: list[list[str]],
                        monkeypatch: pytest.MonkeyPatch) -> dict[str, JSONValue]:
    class Parsed(Exception):
        pass
    seen: dict[str, JSONValue] = {}
    real = argparse.ArgumentParser.parse_args

    def parse(self: argparse.ArgumentParser, args: list[str] | None = None,
              namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        ns = real(self, args, namespace)
        seen["ns"] = {k: v for k, v in sorted(vars(ns).items())}
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    out: dict[str, JSONValue] = {}
    for argv in argvs:
        monkeypatch.setattr("sys.argv", ["dubadabidu", *argv])
        with contextlib.suppress(Parsed):
            entry()
        out[" ".join(argv)] = seen.pop("ns")
    return out


def test_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.StringIO()
    monkeypatch.setattr("sys.argv", ["dubadabidu", "--help"])
    monkeypatch.setenv("COLUMNS", "100")
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        api.cli_main()
    out: dict[str, JSONValue] = {"--help": buf.getvalue()}
    out.update(_capture_namespaces(api.cli_main, ARGVS, monkeypatch))
    order = api.stage_order()
    for i, frm in enumerate(order):
        for to in order[i:]:
            ns = argparse.Namespace(from_stage=frm, to_stage=to)
            before, pod, after = api.remote_plan(ns)
            out[f"remote_plan|{frm}|{to}"] = [before, pod, after, api.remote_stages(ns)]
    _golden("cli", out)


def test_every_shipped_overlay_is_in_the_matrix() -> None:
    names = {Path(f).name for f in api.overlay_files()}
    assert {"config.yaml", "config.gpu.yaml", "config.deepseek.yaml",
            "config.source-ru.yaml"} <= names
    assert sum(n.startswith("config.exp.") for n in names) == 17
