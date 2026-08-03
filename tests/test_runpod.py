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


def test_remote_run_has_no_function_local_shlex_import():
    """A local `import shlex` deep inside remote_run made the name
    function-local for the WHOLE function, so shlex.quote(video) near the top
    raised UnboundLocalError — after provisioning a pod. Python scoping, not a
    typo: the failure is invisible to reading the call site."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "pipeline" / "runpod_infra.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "remote_run")
    shadowed = [a.name for n in ast.walk(fn) if isinstance(n, ast.Import)
                for a in n.names if a.name == "shlex"]
    assert not shadowed, f"local import shadows module-level shlex: {shadowed}"


def test_remote_stages_reads_the_real_argparse_dests():
    """argparse stores --from/--to as from_stage/to_stage. Reading "from"/"to"
    returned None for both, so the pod fell back to the whole pipeline, ran
    s1/s2 and died on the input video that is deliberately never uploaded."""
    import argparse
    from dub import _remote_stages
    narrowed = argparse.Namespace(from_stage="s4_synthesize",
                                  to_stage="s4_synthesize")
    assert _remote_stages(narrowed) == "--from s4_synthesize --to s4_synthesize"
    # untouched flags keep the long-standing pod default: stop at s7, because
    # the mux needs the source video and that stays local
    default = argparse.Namespace(from_stage="s1_extract", to_stage="s8_mux")
    assert _remote_stages(default) == "--to s7_subtitles"


def test_bootstrap_pins_torch_itself_now_that_chatterbox_is_gone():
    """torch/torchaudio used to arrive via the `clone` extra (chatterbox-tts),
    which also dragged diffusers, s3tokenizer, resemble-perth, conformer and
    spacy-pkuseg plus exact pins that made pip backtrack. chatterbox was removed
    2026-08-02, so the pin has to be ours or the pod gets whatever pip picks.

    2.11.0 is only safe because allowed_cuda_versions guarantees a driver that
    supports it — see test_torch_pin_and_cuda_filter_move_together."""
    from pipeline.runpod_infra import REMOTE_SETUP
    setup = REMOTE_SETUP.format(dir="~/d")
    assert "chatterbox" not in setup
    assert "torch==2.11.0 torchaudio==2.11.0" in setup


def test_torch_pin_and_cuda_filter_move_together():
    """THE coupling that a first attempt at this bump got wrong, on a pod:
    torch 2.11's oldest build is cu126 (no 2.11+cu124 wheel exists), so on a
    default-scheduled host with a CUDA 12.4 driver it installs fine and then
    reports CUDA: False -- "The NVIDIA driver on your system is too old (found
    version 12040)". A silent CPU fallback at full GPU price is this project's
    most expensive recurring failure.

    So the pin and the host filter are ONE decision. Emptying
    allowed_cuda_versions without dropping torch back to 2.6.0 re-creates the
    exact failure, and nothing else in the repo would catch it before a pod."""
    import yaml
    from pipeline.logic import deep_merge
    from pipeline.runpod_infra import DEFAULTS, REMOTE_SETUP
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open("config.gpu.yaml", encoding="utf-8")))
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    setup = REMOTE_SETUP.format(dir="~/d")
    import re
    m = re.search(r"torch==(\d+)\.(\d+)\.\d+", setup)
    assert m, "no torch pin found in REMOTE_SETUP"
    major, minor = int(m.group(1)), int(m.group(2))
    allowed = [float(v) for v in rp.get("allowed_cuda_versions") or []]
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
    a cu130 wheel needs a 13.0 driver. Plain PyPI serves torch 2.11 as +cu130,
    so with allowed_cuda_versions ["12.8","12.9","13.0"] two thirds of the
    permitted hosts could not run it — and the failure is the silent one
    (installs fine, CUDA: False, synthesis crawls on CPU at full GPU price).
    A pod on 2026-08-02 passed only because it drew a 13.0 driver.

    Hence the explicit --index-url. This test is the thing that notices when
    someone widens the filter to 12.6 and forgets the wheel."""
    import re
    import yaml
    from pipeline.logic import deep_merge
    from pipeline.runpod_infra import DEFAULTS, REMOTE_SETUP
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open("config.gpu.yaml", encoding="utf-8")))
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    allowed = [float(v) for v in rp.get("allowed_cuda_versions") or []]
    setup = REMOTE_SETUP.format(dir="~/d")
    m = re.search(r"download\.pytorch\.org/whl/cu(\d)(\d+)", setup)
    if not allowed:
        return                      # no filter -> torch must be 2.6.0 anyway
    assert m, (
        f"allowed_cuda_versions is {allowed} but the torch install has no "
        f"explicit cuXYZ --index-url, so pip picks the default wheel (cu130 for "
        f"torch 2.11) which the oldest allowed driver may not support")
    wheel_cuda = float(f"{m.group(1)}.{m.group(2)}")
    assert wheel_cuda <= min(allowed), (
        f"torch wheel is cu{m.group(1)}{m.group(2)} ({wheel_cuda}) but the filter "
        f"admits drivers as old as {min(allowed)} — those hosts would install it "
        f"and report CUDA: False")


def test_pod_payload_sends_the_cuda_filter():
    """It is only a guarantee if it reaches the API."""
    from pipeline.runpod_infra import DEFAULTS
    import pipeline.runpod_infra as R
    sent = {}
    orig = R._req
    R._req = lambda method, path, body=None: sent.update(body or {}) or {}
    try:
        R.create_pod({**DEFAULTS, "allowed_cuda_versions": ["12.9"]})
        assert sent.get("allowedCudaVersions") == ["12.9"]
        sent.clear()
        R.create_pod({**DEFAULTS, "allowed_cuda_versions": []})
        assert "allowedCudaVersions" not in sent, (
            "an empty list must be omitted, not sent — it would filter to no hosts")
    finally:
        R._req = orig


def test_qc_does_not_call_torchaudio_load():
    """torchaudio.load needs torchcodec from 2.9 on — and `hasattr(torchaudio,
    "load")` is still True, so the break shows up only when a scoring run
    actually reads a file, on a pod, mid-billing. Cheap to assert statically."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    # AST, not a substring: the first cut of this test flagged the comment in
    # metrics.py that EXPLAINS the migration.
    for py in sorted(root.glob("qc/*.py")) + sorted(root.glob("pipeline/*.py")):
        for n in ast.walk(ast.parse(py.read_text())):
            called = n.func if isinstance(n, ast.Call) else None
            assert not (isinstance(called, ast.Attribute)
                        and called.attr == "load"
                        and getattr(called.value, "id", "") == "torchaudio"), (
                f"{py.name}:{n.lineno} calls torchaudio.load, which requires the "
                f"torchcodec package on torch>=2.9 — read with soundfile instead")


def test_engine_installs_into_the_main_venv():
    """Per-engine venvs existed for four engines with colliding pins. Three are
    gone and chatterbox took its torch pin with it, so isolation was costing a
    SECOND ~2.5 GB torch download per pod to prevent a collision that can no
    longer happen. Removed 2026-08-02."""
    from pipeline.runpod_infra import engine_install_cmd
    cmd = engine_install_cmd("~/dubadabidu", "qwen", "pip install -q qwen")
    assert ". .venv/bin/activate" in cmd
    assert "venvs/" not in cmd, "no per-engine venv should be created"
    assert cmd.rstrip().endswith("pip install -q qwen")


def test_ready_probe_checks_cuda_in_the_main_venv():
    """Skipping setup is only safe if the probe checks what setup guarantees.
    A CPU-only torch is silent — it cost a week of 126 s/take runs — so the
    probe must assert cuda availability, not merely that the module imports."""
    import pipeline.runpod_infra as R
    seen = {}
    R_ssh = R.ssh_exec
    try:
        R.ssh_exec = lambda rp, h, p, cmd, timeout=None: (
            seen.__setitem__("cmd", cmd) or 0)
        assert R._verify_ready({}, "h", 22, "~/d", ["qwen"]) is True
    finally:
        R.ssh_exec = R_ssh
    assert ".venv/bin/python" in seen["cmd"]
    assert "torch.cuda.is_available()" in seen["cmd"]
    assert "import qwen_tts" in seen["cmd"]


def test_runtime_cap_fits_a_full_hour_lesson():
    """max_runtime_hours is the pod-side watchdog's self-destruct, so a value
    below the real workload kills a run mid-flight AFTER paying for it.

    MEASURED 2026-08-02 on the first production lesson: 8.13 min of video x 5
    languages took 38.8 min of pod wall — 30.9 min synthesis (scales with
    content) + 7.8 min bootstrap (fixed). A 1-hour lesson therefore needs
    ~3.94 h, which the old 3.0 h cap would have cut off at ~76% done."""
    import yaml
    from pipeline.logic import deep_merge
    from pipeline.runpod_infra import DEFAULTS
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open("config.gpu.yaml", encoding="utf-8")))
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    SYNTH_MIN_PER_LANG_PER_VIDEO_MIN = 30.9 / 5 / 8.13   # measured
    BOOTSTRAP_H = 7.8 / 60
    langs = len(cfg["languages"])
    need = 60.0 * SYNTH_MIN_PER_LANG_PER_VIDEO_MIN * langs / 60.0 + BOOTSTRAP_H
    assert rp["max_runtime_hours"] >= need, (
        f"max_runtime_hours={rp['max_runtime_hours']} but a 1-hour lesson in "
        f"{langs} languages needs ~{need:.2f} h — the watchdog would kill the "
        f"pod mid-run, after billing for the whole thing")


# --- pod reuse: the probe and the installer must ask the same question ---

def test_reuse_probe_covers_run_tasks_not_just_bakeoff():
    """The reuse probe was hardcoded to bakeoff.engines and saw [] for `run`,
    and _verify_ready([]) is False by design — so `remote run --reuse` could
    never skip setup and course.py re-bootstrapped once per video."""
    from pipeline.runpod_infra import engines_for_task
    cfg = {"tts": {"engine": "qwen", "engine_by_lang": {}},
           "bakeoff": {"engines": ["qwen"]}}
    assert engines_for_task(cfg, "run", ["en", "ru"]) == ["qwen"]
    assert engines_for_task(cfg, "autopilot", ["en"]) == ["qwen"]
    assert engines_for_task(cfg, "bakeoff", ["en"]) == ["qwen"]


def test_engines_for_task_honours_per_language_routing():
    from pipeline.runpod_infra import engines_for_task
    cfg = {"tts": {"engine": "qwen", "engine_by_lang": {"en": "edge"}}}
    assert engines_for_task(cfg, "run", ["en", "de"]) == ["edge", "qwen"]


def test_remote_run_probes_with_the_engines_it_installs():
    """Both call sites must read the SAME list, or the bug returns wearing a
    different hat."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "pipeline" / "runpod_infra.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "remote_run")
    calls = {n.func.id: [a.id for a in n.args if isinstance(a, ast.Name)]
             for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "needed" in calls.get("_verify_ready", []), \
        "_verify_ready must probe the engines the task actually needs"
    assert "needed" in calls.get("_install_engines", [])


def test_edge_alone_still_refuses_to_skip_setup(monkeypatch):
    """edge needs no install, so it proves nothing about the venv."""
    import pipeline.runpod_infra as R
    monkeypatch.setattr(R, "ssh_exec", lambda *a, **k: 0)
    assert R._verify_ready({}, "h", 22, "~/d", ["edge"]) is False
    assert R._verify_ready({}, "h", 22, "~/d", ["edge", "qwen"]) is True


# --- watchdog: re-armable, or a reused pod dies on the FIRST run's clock ---

def test_watchdog_kills_the_previous_sleeper_before_arming(monkeypatch):
    import pipeline.runpod_infra as R
    sent = []
    monkeypatch.setattr(R, "ssh_exec",
                        lambda rp, h, p, cmd, timeout=30: sent.append(cmd) or 0)
    R.arm_pod_watchdog({}, "h", 22, 3600)
    cmd = sent[0]
    assert f"kill \"$(cat {R.WATCHDOG_PID})\"" in cmd, \
        "re-arming must cancel the deadline the previous run set"
    assert cmd.index("kill") < cmd.index("nohup"), "kill the old one FIRST"
    assert f"echo $! > {R.WATCHDOG_PID}" in cmd, "record the pid to kill next time"
    assert "sleep 3600" in cmd


def test_reuse_path_rearms_the_watchdog():
    """course.py attaches ONE pod for twenty videos; each call recomputes a
    fresh deadline locally, so the pod must be told about it."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "pipeline" / "runpod_infra.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "remote_run")
    n_arm = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "arm_pod_watchdog")
    assert n_arm == 2, ("arm on BOTH paths: the fresh provision and the reuse "
                        f"attach (found {n_arm})")


def test_unknown_remote_task_is_refused_before_provisioning():
    """remote_run already refuses an unknown task before provisioning (a check
    added with the sweep/reuse ordering). This pins that it stays BEFORE the
    spend: REMOTE_TASK[task] is read deep inside the try block, so without it a
    typo would KeyError only after a pod was provisioned, bootstrapped and had
    qwen installed."""
    import pytest
    import yaml
    from pipeline.logic import deep_merge
    from pipeline.runpod_infra import REMOTE_TASK, remote_run
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open("config.gpu.yaml", encoding="utf-8")))
    with pytest.raises(SystemExit, match="unknown remote task"):
        remote_run(cfg, "v.mp4", ["en"], "typoed-task")
    # and the tasks that DO exist must stay reachable
    assert {"run", "bakeoff", "autopilot", "preamble"} <= set(REMOTE_TASK)


def test_preamble_has_a_pod_command_because_tune_synthesizes():
    """preamble runs tune R1, which SYNTHESIZES to score each reference by real
    ECAPA similarity — so it cannot run on the laptop with a GPU-only engine."""
    from pipeline.runpod_infra import REMOTE_TASK
    cmd = REMOTE_TASK["preamble"].format(video="v.mp4", langs="en", stages="")
    assert cmd.startswith("dubadabidu preamble ")
    assert "--overlay config.gpu.yaml" in cmd        # engine + pod settings
    assert "--overlay config.deepseek.yaml" in cmd   # preamble runs s3 first
