import time
from pipeline.runpod_infra import (_deadline, engine_install_cmd, ssh_target,
                                   DEFAULTS)


def _rp(**over):
    return {**DEFAULTS, **over}


# --- budget -> deadline guardrail ---

def test_deadline_scales_with_budget():
    rp = _rp(assumed_price_per_hr=0.20, max_runtime_hours=100)
    hrs = (_deadline(rp, 2.0) - time.time()) / 3600
    assert abs(hrs - 10.0) < 0.05          # $2 / $0.20 = 10h


def test_deadline_capped_by_max_runtime():
    rp = _rp(assumed_price_per_hr=0.10, max_runtime_hours=3.0)
    hrs = (_deadline(rp, 100.0) - time.time()) / 3600
    assert abs(hrs - 3.0) < 0.05           # cap wins over 1000h of budget


def test_deadline_small_budget_short():
    rp = _rp(assumed_price_per_hr=0.25, max_runtime_hours=3.0)
    hrs = (_deadline(rp, 10.0) - time.time()) / 3600
    assert abs(hrs - 3.0) < 0.05           # $10/$0.25=40h -> capped to 3h


# --- SSH detail parsing (tolerant of API field-shape variants) ---

def test_ssh_target_real_rest_v1_shape():
    # the shape the live REST v1 API actually returns (captured 2026-07)
    pod = {"publicIp": "213.144.200.243", "ports": ["22/tcp"],
           "portMappings": {"22": 10345}, "runtime": None}
    assert ssh_target(pod) == ("213.144.200.243", 10345)


def test_ssh_target_string_ports_never_crash():
    # ports as a list of strings ("22/tcp") must return None, not raise —
    # the exact bug that leaked a pod on the first smoke test
    assert ssh_target({"ports": ["22/tcp"], "runtime": None}) is None


def test_ssh_target_list_of_dicts_fallback():
    pod = {"publicIp": "1.2.3.4",
           "ports": [{"privatePort": 22, "publicPort": 40022}]}
    assert ssh_target(pod) == ("1.2.3.4", 40022)


def test_ssh_target_runtime_nested():
    pod = {"ip": "9.9.9.9",
           "runtime": {"ports": [{"privatePort": 22, "publicPort": 12345}]}}
    assert ssh_target(pod) == ("9.9.9.9", 12345)


def test_ssh_target_not_ready():
    assert ssh_target({"desiredStatus": "PENDING", "ports": []}) is None
    assert ssh_target({"ports": [{"privatePort": 8888, "publicPort": 1}]}) is None


# --- per-engine venv isolation (the pod-side install command) ---

def test_engine_install_cmd_isolates_per_engine():
    cmd = engine_install_cmd("~/dubadabidu", "qwen", "pip install -q qwen")
    # the snippet must run inside the ENGINE's venv, never the main .venv
    assert "venvs/qwen" in cmd
    assert ". venvs/qwen/bin/activate" in cmd
    assert ".venv/bin/activate" not in cmd.replace("venvs/qwen/bin/activate", "")
    assert cmd.rstrip().endswith("pip install -q qwen")


def test_engine_install_cmd_distinct_venvs():
    a = engine_install_cmd("~/d", "indextts", "true")
    b = engine_install_cmd("~/d", "qwen", "true")
    assert "venvs/indextts" in a and "venvs/indextts" not in b
    assert "venvs/qwen" in b and "venvs/qwen" not in a


# --- setup-check refuses a run that would validate nothing ---

def _probe(engines, snippets):
    from pipeline.runpod_infra import ENGINE_MODULE
    return [(e, ENGINE_MODULE[e], e in snippets)
            for e in engines if e != "chatterbox" and e in ENGINE_MODULE]


def test_no_engines_configured_is_refused_before_provisioning():
    import pytest
    from pipeline.runpod_infra import _require_something_to_validate
    with pytest.raises(SystemExit, match="no bakeoff.engines configured"):
        _require_something_to_validate([], [])


def test_incumbent_only_is_refused():
    """chatterbox installs via REMOTE_SETUP on every run — probing it alone
    tells you nothing a real run wouldn't."""
    import pytest
    from pipeline.runpod_infra import _require_something_to_validate
    with pytest.raises(SystemExit, match="only the incumbent"):
        _require_something_to_validate(["chatterbox"], _probe(["chatterbox"], {}))


def test_challengers_without_install_snippets_are_refused():
    import pytest
    from pipeline.runpod_infra import _require_something_to_validate
    engines = ["chatterbox", "indextts", "qwen"]
    with pytest.raises(SystemExit, match="no runpod.engine_setup snippet"):
        _require_something_to_validate(engines, _probe(engines, {}))


def test_one_snippet_is_enough_to_proceed():
    """A partial bake-off is legitimate — only a total absence is refused."""
    from pipeline.runpod_infra import _require_something_to_validate
    engines = ["chatterbox", "indextts", "qwen"]
    _require_something_to_validate(engines, _probe(engines, {"qwen"}))


def test_the_actual_missing_overlay_case_is_caught():
    """THE regression guard: `remote setup-check` with no --overlay. Both the
    bakeoff and runpod sections live in config.gpu.yaml, so plain config.yaml
    falls back to DEFAULTS (engine_setup: {}) -> a paid pod that probes nothing
    and still returns True because the incumbent imports."""
    import pytest
    import yaml
    from pipeline.runpod_infra import (DEFAULTS, ENGINE_MODULE,
                                       _require_something_to_validate)
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    engines = list(dict.fromkeys(cfg.get("bakeoff", {}).get("engines", [])))
    probe = [(e, ENGINE_MODULE[e], bool(rp.get("engine_setup", {}).get(e)))
             for e in engines if e != "chatterbox" and e in ENGINE_MODULE]
    with pytest.raises(SystemExit, match="config.gpu.yaml"):
        _require_something_to_validate(engines, probe)


def test_the_correct_invocation_passes():
    """config.yaml + config.gpu.yaml (what --overlay produces) must proceed."""
    import yaml
    from pipeline.logic import deep_merge
    from pipeline.runpod_infra import (DEFAULTS, ENGINE_MODULE,
                                       _require_something_to_validate)
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open("config.gpu.yaml", encoding="utf-8")))
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    engines = list(dict.fromkeys(cfg.get("bakeoff", {}).get("engines", [])))
    probe = [(e, ENGINE_MODULE[e], bool(rp.get("engine_setup", {}).get(e)))
             for e in engines if e != "chatterbox" and e in ENGINE_MODULE]
    _require_something_to_validate(engines, probe)      # must not raise


def test_extra_overlays_reach_the_pod_command():
    """REMOTE_TASK hardcodes --overlay config.gpu.yaml, so an experiment overlay
    used to shape the LOCAL config and be silently dropped on the pod — the run
    then executed the BASE config while the logs implied otherwise. Measured
    2026-08-01: ~10 min of pod time re-sweeping references during what was
    supposed to be an ICL test."""
    from pipeline.runpod_infra import REMOTE_TASK
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
    from pipeline.experiments import _cmd
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
    from pathlib import Path
    from qc.bakeoff import variant_key
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "experiments.yaml").read_text())
    base = yaml.safe_load((root / "config.gpu.yaml").read_text())["tts"]
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


def test_provisioning_logs_the_price_not_a_gpu_name():
    """REST v1 GET /pods/{id} returns an EMPTY machine object and no gpuTypeId
    (checked 2026-08-01), so the card's identity is not retrievable. Price is
    the better signal anyway — it is what the budget is spent in."""
    from pipeline.runpod_infra import provision
    import inspect
    src = inspect.getsource(provision)
    assert "costPerHr" in src and "OVER the budgeted" in src


def test_gpu_type_priority_defaults_to_custom_so_list_order_matters():
    """With "availability" RunPod ignores the order and picks whatever is free,
    which is why a cheapest-first list still landed on a 4090 every run (7% GPU,
    7% CPU, 17% RAM on a top-rate card). Only "availability" and "custom" are
    accepted by the API."""
    from pipeline.runpod_infra import create_pod
    import inspect
    src = inspect.getsource(create_pod)
    assert '"gpuTypePriority": rp.get("gpu_type_priority", "custom")' in src
    assert '"gpuTypePriority": "availability"' not in src


def test_config_puts_cheap_gpus_first_and_4090_last():
    """The ordering is load-bearing now, so a careless re-sort would silently
    put us back on the most expensive card."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config.gpu.yaml").read_text())
    ids = cfg["runpod"]["gpu_type_ids"]
    assert cfg["runpod"]["gpu_type_priority"] == "custom"
    assert ids[0] == "NVIDIA RTX A4000"
    assert "4090" in ids[-1]


def test_ready_probe_checks_cuda_inside_the_ENGINE_venv(monkeypatch):
    """Skipping setup is only safe if the probe checks what setup guarantees.
    The base .venv cannot see a CPU-only torch in venvs/<engine> — that exact
    blind spot cost a week of 126 s/take runs — so the probe must run inside
    the engine's own venv."""
    import pipeline.runpod_infra as R
    seen = {}

    def fake_exec(rp, host, port, cmd, timeout=None):
        seen["cmd"] = cmd
        return 0

    monkeypatch.setattr(R, "ssh_exec", fake_exec)
    assert R._verify_ready({}, "h", 22, "~/d", ["qwen"]) is True
    cmd = seen["cmd"]
    assert "venvs/qwen/bin/python" in cmd
    assert "torch.cuda.is_available()" in cmd
    assert "import qwen_tts" in cmd


def test_ready_probe_is_conservative(monkeypatch):
    """Any doubt re-runs setup: skipping wrongly costs a whole experiment,
    running needlessly costs ~50 s."""
    import pipeline.runpod_infra as R
    monkeypatch.setattr(R, "ssh_exec", lambda *a, **k: 1)      # probe fails
    assert R._verify_ready({}, "h", 22, "~/d", ["qwen"]) is False
    # no engines to prove anything about -> do not skip
    monkeypatch.setattr(R, "ssh_exec", lambda *a, **k: 0)
    assert R._verify_ready({}, "h", 22, "~/d", []) is False
    # an engine with no module mapping (chatterbox lives in the base venv)
    assert R._verify_ready({}, "h", 22, "~/d", ["edge"]) is False
