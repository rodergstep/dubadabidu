# SPDX-License-Identifier: AGPL-3.0-or-later
"""What runs on the pod: engine installs, setup-check refusals, task commands."""
from dubadabidu.infra.remote.bootstrap import engine_install_cmd
from dubadabidu.core.config.schema import validate
from tests.cfgs import settings
from tests.paths import REPO, config


# --- per-engine venv isolation (the pod-side install command) ---



def _probe(engines, snippets):
    from dubadabidu.infra.remote.bootstrap import ENGINE_MODULE
    return [(e, ENGINE_MODULE[e], e in snippets)
            for e in engines if e != "chatterbox" and e in ENGINE_MODULE]


def test_no_engines_configured_is_refused_before_provisioning():
    import pytest
    from dubadabidu.infra.remote.diagnostics import _require_something_to_validate
    with pytest.raises(SystemExit, match="no bakeoff.engines configured"):
        _require_something_to_validate([], [])


def test_incumbent_only_is_refused():
    """chatterbox installs via REMOTE_SETUP on every run — probing it alone
    tells you nothing a real run wouldn't."""
    import pytest
    from dubadabidu.infra.remote.diagnostics import _require_something_to_validate
    with pytest.raises(SystemExit, match="only the incumbent"):
        _require_something_to_validate(["chatterbox"], _probe(["chatterbox"], {}))


def test_challengers_without_install_snippets_are_refused():
    import pytest
    from dubadabidu.infra.remote.diagnostics import _require_something_to_validate
    engines = ["chatterbox", "indextts", "qwen"]
    with pytest.raises(SystemExit, match="no runpod.engine_setup snippet"):
        _require_something_to_validate(engines, _probe(engines, {}))


def test_one_snippet_is_enough_to_proceed():
    """A partial bake-off is legitimate — only a total absence is refused."""
    from dubadabidu.infra.remote.diagnostics import _require_something_to_validate
    engines = ["chatterbox", "indextts", "qwen"]
    _require_something_to_validate(engines, _probe(engines, {"qwen"}))


def test_the_actual_missing_overlay_case_is_caught():
    """THE regression guard: `remote setup-check` with no --overlay. Both the
    bakeoff and runpod sections live in config.gpu.yaml, so plain config.yaml
    falls back to the schema defaults (engine_setup: {}) -> a paid pod that probes nothing
    and still returns True because the incumbent imports."""
    import pytest
    import yaml
    from dubadabidu.infra.remote.bootstrap import ENGINE_MODULE
    from dubadabidu.infra.remote.diagnostics import _require_something_to_validate
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    rp = validate(cfg).runpod
    engines = list(dict.fromkeys(cfg.get("bakeoff", {}).get("engines", [])))
    probe = [(e, ENGINE_MODULE[e], bool(rp.engine_setup.get(e)))
             for e in engines if e != "chatterbox" and e in ENGINE_MODULE]
    with pytest.raises(SystemExit, match="config.gpu.yaml"):
        _require_something_to_validate(engines, probe)


def test_the_correct_invocation_passes():
    """config.yaml + config.gpu.yaml (what --overlay produces) must proceed."""
    import yaml
    from dubadabidu.core.logic import deep_merge
    from dubadabidu.infra.remote.bootstrap import ENGINE_MODULE
    from dubadabidu.infra.remote.diagnostics import _require_something_to_validate
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8")))
    rp = validate(cfg).runpod
    engines = list(dict.fromkeys(cfg.get("bakeoff", {}).get("engines", [])))
    probe = [(e, ENGINE_MODULE[e], bool(rp.engine_setup.get(e)))
             for e in engines if e != "chatterbox" and e in ENGINE_MODULE]
    _require_something_to_validate(engines, probe)      # must not raise


def test_extra_overlays_reach_the_pod_command():
    """REMOTE_TASK hardcodes --overlay config.gpu.yaml, so an experiment overlay
    used to shape the LOCAL config and be silently dropped on the pod — the run
    then executed the BASE config while the logs implied otherwise. Measured
    2026-08-01: ~10 min of pod time re-sweeping references during what was
    supposed to be an ICL test."""
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    base = REMOTE_TASK["bakeoff"].format(video="v.mp4", langs="en")
    extras = [o for o in ["config.gpu.yaml", "config.exp.icl.yaml"]
              if o != "config.gpu.yaml"]
    cmd = base + "".join(f" --overlay {o}" for o in extras)
    assert cmd.count("--overlay config.gpu.yaml") == 1   # not duplicated
    assert cmd.endswith("--overlay config.exp.icl.yaml")  # extras win (last)


# --- experiment queue: many experiments, one pod ---

def test_experiment_commands_reuse_and_keep_alive(tmp_path):
    """Every experiment must attach to the SAME pod. Without --reuse each one
    re-provisions (5-10 min and ~$0.10 of bootstrap before any measurement),
    which is what made small questions 'not worth a pod' and got them answered
    by inference instead."""
    from dubadabidu.lab.experiments import _cmd
    cmd = _cmd("v.mp4", ["en"], ["config.exp.06b.yaml"])
    assert "--reuse" in cmd and "--keep-alive" in cmd
    # base overlay first, experiment overlay after, so the experiment wins
    i_base = cmd.index("config.gpu.yaml")
    i_exp = cmd.index("config.exp.06b.yaml")
    assert i_base < i_exp


def test_every_enabled_experiment_isolates_one_axis():
    """An experiment sharing a variant_key with another silently overwrites its
    row AND its audio — the failure that already cost the plain-indextts
    recordings. Enabled experiments must therefore produce distinct keys.

    Keyed by (lang, variant): results live in results_<lang>.json and audio in
    seg/<variant>/<lang>/, so the same variant in two languages is NOT a
    collision — an en control and a ru control coexist correctly."""
    import yaml
    from dubadabidu.lab.bakeoff import variant_key
    root = REPO
    spec = yaml.safe_load((root / "specs" / "experiments.yaml").read_text())
    base = yaml.safe_load(config("config.gpu.yaml").read_text())["tts"]
    keys = {}
    for e in spec["experiments"]:
        if not e.get("enabled"):
            continue
        t = dict(base, engine="qwen")
        for ov in e.get("overlays") or []:
            t.update((yaml.safe_load((root / ov).read_text()).get("tts") or {}))
        for lang in e["langs"]:
            k = (lang, variant_key("qwen", t, lang))
            assert k not in keys, (
                f"{e['name']} and {keys[k]} both key to {k!r} — the second "
                f"would overwrite the first's scorecard row and audio")
            keys[k] = e["name"]


def test_pod_never_runs_a_stage_needing_an_extra_it_does_not_install():
    """The pod installs `.[dev]`. s6's mastering chain needs `.[master]`
    (pedalboard, GPLv3, opt-in on purpose), so a pod that reaches s6 with
    mix.master.enabled dies AFTER all the billed GPU work is done."""
    import argparse
    from dubadabidu.cli.remote import ORDER, _remote_stages
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    default = argparse.Namespace(from_stage="s1_extract", to_stage="s8_mux")
    stop = _remote_stages(default).split("--to ")[1].strip()
    assert ORDER.index(stop) < ORDER.index("s6_mix"), (
        f"the pod stops at {stop}, which reaches s6_mix — that stage imports "
        f"pedalboard from the [master] extra, which no pod installs")
    # and the fallback inside remote_run must agree with it
    fallback = REMOTE_TASK["run"].format(video="v", langs="en", stages="")
    assert "{stages}" not in fallback


def test_engine_installs_into_the_main_venv():
    """Per-engine venvs existed for four engines with colliding pins. Three are
    gone and chatterbox took its torch pin with it, so isolation was costing a
    SECOND ~2.5 GB torch download per pod to prevent a collision that can no
    longer happen. Removed 2026-08-02."""
    cmd = engine_install_cmd("~/dubadabidu", "qwen", "pip install -q qwen")
    assert ". .venv/bin/activate" in cmd
    assert "venvs/" not in cmd, "no per-engine venv should be created"
    assert cmd.rstrip().endswith("pip install -q qwen")


# --- pod reuse: the probe and the installer must ask the same question ---

def test_reuse_probe_covers_run_tasks_not_just_bakeoff():
    """The reuse probe was hardcoded to bakeoff.engines and saw [] for `run`,
    and _verify_ready([]) is False by design — so `remote run --reuse` could
    never skip setup and course.py re-bootstrapped once per video."""
    from dubadabidu.infra.remote.bootstrap import engines_for_task
    cfg = settings({"tts": {"engine": "qwen"}, "bakeoff": {"engines": ["qwen"]}})
    assert engines_for_task(cfg, "run", ["en", "ru"]) == ["qwen"]
    assert engines_for_task(cfg, "autopilot", ["en"]) == ["qwen"]
    assert engines_for_task(cfg, "bakeoff", ["en"]) == ["qwen"]


def test_engines_for_task_honours_per_language_routing():
    from dubadabidu.infra.remote.bootstrap import engines_for_task
    cfg = settings({"tts": {"engine": "qwen", "engine_by_lang": {"en": "edge"}}})
    assert engines_for_task(cfg, "run", ["en", "de"]) == ["edge", "qwen"]


def test_preamble_has_a_pod_command_because_tune_synthesizes():
    """preamble runs tune R1, which SYNTHESIZES to score each reference by real
    ECAPA similarity — so it cannot run on the laptop with a GPU-only engine."""
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    cmd = REMOTE_TASK["preamble"].format(video="v.mp4", langs="en", stages="")
    assert cmd.startswith("dubadabidu preamble ")
    assert "--overlay config.gpu.yaml" in cmd        # engine + pod settings
    assert "--overlay config.deepseek.yaml" in cmd   # preamble runs s3 first


def test_engine_list_handed_to_the_installer_is_engine_names():
    """Defence in depth: the silent part of the bug was that _install_engines
    treats an unrecognised name as 'no snippet, skip'. That is correct for edge
    and catastrophic for a file path, so nothing else may flow into it.

    engines_for_task is the only source, and every name it can return for the
    shipped config must be a known engine."""
    import yaml
    from dubadabidu.core.logic import deep_merge
    from dubadabidu.infra.remote.bootstrap import engines_for_task, ENGINE_MODULE
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8")))
    for task in ("bakeoff", "run"):
        for e in engines_for_task(validate(cfg), task, ["en", "ru"]):
            assert e in ENGINE_MODULE, (
                f"{task}: {e!r} is not an engine name — if a path reaches "
                f"_install_engines it silently installs nothing")


def test_bootstrap_reuses_a_prebuilt_venv_only_when_the_lock_matches():
    """P3: the image's venv is linked in ONLY when its uv.lock stamp equals the
    project's. A mismatched image must fall through to a fresh build, never be
    used — an old venv against a new lock is the silent-wrong-env failure."""
    from dubadabidu.infra.remote.bootstrap import REMOTE_SETUP
    setup = REMOTE_SETUP.format(dir="~/d")
    assert "/opt/venv/.uv-lock-sha256" in setup
    assert "sha256sum uv.lock" in setup
    assert setup.index("ln -s /opt/venv .venv") < setup.index("python3 -m venv .venv")
    assert "[ -x .venv/bin/python ] ||" in setup, "venv must not be re-created"
    assert "uv sync --frozen" in setup, "the lock still has the last word"


def test_dockerfile_stamps_the_lock_and_pins_the_engine_like_engine_setup():
    import re
    import yaml
    df = open("docker/Dockerfile", encoding="utf-8").read()
    assert "/opt/venv/.uv-lock-sha256" in df
    sha = re.search(r"ARG QWEN_SHA=([0-9a-f]{40})", df).group(1)
    gpu = yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8"))
    snippet = gpu["runpod"]["engine_setup"]["qwen"]
    assert sha in snippet, "image and engine_setup pin different qwen commits"
    fq = re.search(r"ARG FASTER_QWEN=(\S+)", df).group(1)
    assert f"faster-qwen3-tts=={fq}" in snippet


def test_a_ghcr_image_in_config_is_tagged_with_the_current_lock():
    """If runpod.image points at our GHCR image, its tag must be the current
    lock's — otherwise every pod pays the full bootstrap (stamp mismatch)."""
    import hashlib
    import yaml
    gpu = yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8"))
    image = gpu["runpod"]["image"]
    if "dubadabidu-pod" not in image:
        return
    want = "lock-" + hashlib.sha256(open("uv.lock", "rb").read()).hexdigest()[:12]
    assert image.endswith(":" + want), f"{image} is not built from this uv.lock ({want})"
