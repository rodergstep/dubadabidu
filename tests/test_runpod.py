# SPDX-License-Identifier: AGPL-3.0-or-later
import time
from dubadabidu.infra.remote.bootstrap import engine_install_cmd
from dubadabidu.infra.remote.orchestrator import _deadline
from dubadabidu.core.config.schema import Runpod, derive, validate
from dubadabidu.infra.remote.runpod import ssh_target
from tests.cfgs import settings
from tests.paths import REPO, config, remote_source


def _rp(**over):
    return Runpod(**over)


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


def test_provisioning_logs_the_price_not_a_gpu_name():
    """REST v1 GET /pods/{id} returns an EMPTY machine object and no gpuTypeId
    (checked 2026-08-01), so the card's identity is not retrievable. Price is
    the better signal anyway — it is what the budget is spent in."""
    from dubadabidu.infra.remote.orchestrator import provision
    import inspect
    src = inspect.getsource(provision)
    assert "provider.active().price(info)" in src and "OVER the budgeted" in src


def test_gpu_types_are_tried_in_list_order_and_400_falls_through():
    """REST v2 takes ONE gpu id per create, so the cheapest-first ordering is
    ours to enforce. (v1's "availability" priority ignored the order and landed
    every run on a 4090 at 7% utilisation.) A 400 means no capacity for that
    type -> next one; a 422 is a schema bug and must NOT be retried."""
    from dubadabidu.infra.remote import runpod
    tried: list[str] = []
    orig = runpod._req

    def fake(method, path, body=None):
        tried.append(body["gpu"]["id"])
        if len(tried) < 3:
            raise RuntimeError("RunPod POST /pods -> HTTP 400: no capacity")
        return {"id": "p1"}
    runpod._req = fake
    try:
        rp = Runpod(gpu_type_ids=["a", "b", "c", "d"])
        assert runpod.create_pod(rp) == {"id": "p1"}
        assert tried == ["a", "b", "c"]
        tried.clear()

        def schema_err(method, path, body=None):
            tried.append(body["gpu"]["id"])
            raise RuntimeError("RunPod POST /pods -> HTTP 422: bad field")
        runpod._req = schema_err
        import pytest
        with pytest.raises(RuntimeError, match="422"):
            runpod.create_pod(rp)
        assert tried == ["a"], "a schema error must not be retried per GPU type"
    finally:
        runpod._req = orig


def test_v2_payload_has_only_documented_fields():
    """v2 rejects unknown keys with 422, so a v1 field left behind fails every
    create. This is the shape from docs.runpod.io/api-reference-v2."""
    from dubadabidu.infra.remote.runpod import pod_payload
    body = pod_payload(Runpod(network_volume_id="v1"), "NVIDIA RTX A4000")
    allowed = {"name", "image", "gpu", "cloud", "disk", "ports", "env",
               "mounts", "registry", "startSsh"}
    assert body["startSsh"] is True, "v2 starts no sshd without it"
    assert set(body) <= allowed, set(body) - allowed
    assert body["gpu"]["id"] == "NVIDIA RTX A4000"
    assert "22/tcp" in body["ports"], "no 22/tcp -> no ssh.direct mapping"
    assert body["mounts"]["network"][0]["path"] == "/workspace"


def test_v2_pod_shapes_are_read():
    from dubadabidu.infra.remote.runpod import pod_price, pod_ram_gb, pod_status, pod_vcpu, ssh_target
    pod = {"status": "RUNNING", "cost": 0.44,
           "gpu": {"id": "x", "count": 1, "vcpuCount": 16, "memory": 64},
           "ssh": {"direct": {"host": "195.26.233.3", "port": 34446}},
           "runtime": {"ports": [{"private": 22, "public": 34446,
                                  "type": "tcp", "ip": "195.26.233.3"}]}}
    assert pod_status(pod) == "RUNNING" and pod_price(pod) == 0.44
    assert pod_vcpu(pod) == 16 and pod_ram_gb(pod) == 64
    assert ssh_target(pod) == ("195.26.233.3", 34446)
    only_runtime = {**pod, "ssh": {"direct": None}}
    assert ssh_target(only_runtime) == ("195.26.233.3", 34446)


def test_config_puts_cheap_gpus_first_and_4090_last():
    """The ordering is load-bearing now, so a careless re-sort would silently
    put us back on the most expensive card."""
    import yaml
    cfg = yaml.safe_load(
        config("config.gpu.yaml").read_text())
    ids = cfg["runpod"]["gpu_type_ids"]
    assert "gpu_type_priority" not in cfg["runpod"], "v1-only key; v2 has no such field"
    assert ids[0] == "NVIDIA RTX A4000"
    assert "4090" in ids[-1]



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


def test_remote_run_has_no_function_local_shlex_import():
    """A local `import shlex` deep inside remote_run made the name
    function-local for the WHOLE function, so shlex.quote(video) near the top
    raised UnboundLocalError — after provisioning a pod. Python scoping, not a
    typo: the failure is invisible to reading the call site."""
    import ast
    src = remote_source()
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
    from dubadabidu.cli.remote import _remote_stages
    narrowed = argparse.Namespace(from_stage="s4_synthesize",
                                  to_stage="s4_synthesize")
    assert _remote_stages(narrowed) == "--from s4_synthesize --to s4_synthesize"
    # untouched flags stop the pod at s5: s4 is the only stage needing a GPU,
    # and s6/s7 are CPU work on files that sync both ways. Was s7 until
    # 2026-08-09, when s6's pedalboard import (the [master] extra, never
    # installed on a pod) failed a run in which all five languages had already
    # synthesized and fitted successfully.
    default = argparse.Namespace(from_stage="s1_extract", to_stage="s8_mux")
    assert _remote_stages(default) == "--from s4_synthesize --to s5_fit"
    # a later --to cannot drag the pod past s5: s6..s8 run locally instead
    further = argparse.Namespace(from_stage="s1_extract",
                                 to_stage="s7_subtitles")
    assert _remote_stages(further) == "--from s4_synthesize --to s5_fit"


def test_remote_plan_translates_locally_and_finishes_locally():
    """2026-11-24: `remote run` with untouched flags re-ran s2 ON THE POD
    (s2 has no cache: it discards every translation) and crashed there; and
    the local follow-up muxed without s6, i.e. stale dub_<lang>.m4a files.
    The plan is now local s3 -> pod s4..s5 -> local s6..s8."""
    import argparse
    from dubadabidu.cli.remote import _remote_plan
    before, pod, after = _remote_plan(
        argparse.Namespace(from_stage="s1_extract", to_stage="s8_mux"))
    assert before == ["s3_translate"]
    assert pod == ["s4_synthesize", "s5_fit"]
    assert after == ["s6_mix", "s7_subtitles", "s8_mux"]
    # an explicit --from s2 is honoured, locally
    before, pod, _ = _remote_plan(
        argparse.Namespace(from_stage="s2_transcribe", to_stage="s8_mux"))
    assert before == ["s2_transcribe", "s3_translate"]
    # nothing for the GPU -> no pod at all
    _, pod, after = _remote_plan(
        argparse.Namespace(from_stage="s6_mix", to_stage="s8_mux"))
    assert pod == [] and after == ["s6_mix", "s7_subtitles", "s8_mux"]


def test_run_upload_sends_only_this_videos_workdir():
    """Uploading all of work/ shipped every other lesson (2.3 GB for a
    339 MB video) and was most of a 22.6 min bootstrap."""
    src = remote_source()
    assert ("_rsync_paths(rp, port, [str(M.video_workdir(cfg, v))\n"
            "                                    for v in videos], up)") in src
    assert 'extra_excludes=("input", "output"))' not in src


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


def test_pod_payload_sends_the_cuda_filter():
    """It is only a guarantee if it reaches the API."""
    from dubadabidu.infra.remote import runpod
    sent = {}
    orig = runpod._req
    runpod._req = lambda method, path, body=None: sent.update(body or {}) or {}
    try:
        runpod.create_pod(Runpod(allowed_cuda_versions=["12.9"]))
        assert sent["gpu"].get("allowedCudaVersions") == ["12.9"]
        assert "minCudaVersion" not in sent["gpu"], "v2 400s on both together"
        sent.clear()
        runpod.create_pod(Runpod(allowed_cuda_versions=[]))
        assert sent["gpu"].get("minCudaVersion") == "13.0"
        assert "allowedCudaVersions" not in sent["gpu"], (
            "an empty list must be omitted, not sent — it would filter to no hosts")
    finally:
        runpod._req = orig


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


def test_engine_installs_into_the_main_venv():
    """Per-engine venvs existed for four engines with colliding pins. Three are
    gone and chatterbox took its torch pin with it, so isolation was costing a
    SECOND ~2.5 GB torch download per pod to prevent a collision that can no
    longer happen. Removed 2026-08-02."""
    cmd = engine_install_cmd("~/dubadabidu", "qwen", "pip install -q qwen")
    assert ". .venv/bin/activate" in cmd
    assert "venvs/" not in cmd, "no per-engine venv should be created"
    assert cmd.rstrip().endswith("pip install -q qwen")


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


# --- pod reuse: the probe and the installer must ask the same question ---

def test_reuse_probe_covers_run_tasks_not_just_bakeoff():
    """The reuse probe was hardcoded to bakeoff.engines and saw [] for `run`,
    and _verify_ready([]) is False by design — so `remote run --reuse` could
    never skip setup and course.py re-bootstrapped once per video."""
    from dubadabidu.infra.remote.bootstrap import engines_for_task
    cfg = {"tts": {"engine": "qwen", "engine_by_lang": {}},
           "bakeoff": {"engines": ["qwen"]}}
    assert engines_for_task(cfg, "run", ["en", "ru"]) == ["qwen"]
    assert engines_for_task(cfg, "autopilot", ["en"]) == ["qwen"]
    assert engines_for_task(cfg, "bakeoff", ["en"]) == ["qwen"]


def test_engines_for_task_honours_per_language_routing():
    from dubadabidu.infra.remote.bootstrap import engines_for_task
    cfg = {"tts": {"engine": "qwen", "engine_by_lang": {"en": "edge"}}}
    assert engines_for_task(cfg, "run", ["en", "de"]) == ["edge", "qwen"]


def test_remote_run_probes_with_the_engines_it_installs():
    """Both call sites must read the SAME list, or the bug returns wearing a
    different hat."""
    import ast
    src = remote_source()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "remote_run")
    calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id):
             [a.id for a in n.args if isinstance(a, ast.Name)]
             for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert "needed" in calls.get("_verify_ready", []), \
        "_verify_ready must probe the engines the task actually needs"
    assert "needed" in calls.get("_install_engines", [])


def test_edge_alone_still_refuses_to_skip_setup(monkeypatch):
    """edge needs no install, so it proves nothing about the venv."""
    from dubadabidu.infra.remote import transport
    monkeypatch.setattr(transport, "ssh_exec", lambda *a, **k: 0)
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["edge"]) is False
    assert transport._verify_ready(Runpod(), "h", 22, "~/d", ["edge", "qwen"]) is True


# --- watchdog: re-armable, or a reused pod dies on the FIRST run's clock ---

def test_watchdog_kills_the_previous_sleeper_before_arming(monkeypatch):
    from dubadabidu.infra.remote import runpod, transport
    sent = []
    monkeypatch.setattr(transport, "ssh_exec",
                        lambda rp, h, p, cmd, timeout=30: sent.append(cmd) or 0)
    runpod.arm_pod_watchdog(Runpod(), "h", 22, 3600)
    cmd = sent[0]
    assert f"kill \"$(cat {runpod.WATCHDOG_PID})\"" in cmd, \
        "re-arming must cancel the deadline the previous run set"
    assert cmd.index("kill") < cmd.index("nohup"), "kill the old one FIRST"
    assert f"echo $! > {runpod.WATCHDOG_PID}" in cmd, "record the pid to kill next time"
    assert "sleep 3600" in cmd


def test_reuse_path_rearms_the_watchdog():
    """course.py attaches ONE pod for twenty videos; each call recomputes a
    fresh deadline locally, so the pod must be told about it."""
    import ast
    src = remote_source()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "remote_run")
    n_arm = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "arm_watchdog")
    # 3: the fresh provision, the reuse attach, and the failure HOLD (which
    # shortens the deadline to hold_on_failure_min)
    assert n_arm == 3, ("arm on every path: fresh provision, reuse attach, "
                        f"hotfix hold (found {n_arm})")


def test_unknown_remote_task_is_refused_before_provisioning():
    """remote_run already refuses an unknown task before provisioning (a check
    added with the sweep/reuse ordering). This pins that it stays BEFORE the
    spend: REMOTE_TASK[task] is read deep inside the try block, so without it a
    typo would KeyError only after a pod was provisioned, bootstrapped and had
    qwen installed."""
    import pytest
    import yaml
    from dubadabidu.core.logic import deep_merge
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    from dubadabidu.infra.remote.orchestrator import remote_run
    cfg = deep_merge(yaml.safe_load(open("config.yaml", encoding="utf-8")),
                     yaml.safe_load(open(config("config.gpu.yaml"), encoding="utf-8")))
    # Neutralise the two API calls that run BEFORE the guard, so this tests the
    # guard and not whether RUNPOD_API_KEY happens to be exported: without the
    # key, sweep_orphans() raises about credentials first and the assertion
    # passes or fails depending on the shell.
    from dubadabidu.infra.remote import state
    orig_sweep, orig_live = state.sweep_orphans, state._live_pod
    state.sweep_orphans = lambda *a, **k: None
    state._live_pod = lambda *a, **k: None
    try:
        with pytest.raises(SystemExit, match="unknown remote task"):
            remote_run(validate(cfg), "v.mp4", ["en"], "typoed-task")
    finally:
        state.sweep_orphans, state._live_pod = orig_sweep, orig_live
    # and the tasks that DO exist must stay reachable
    assert {"run", "bakeoff", "autopilot", "preamble"} <= set(REMOTE_TASK)


def test_preamble_has_a_pod_command_because_tune_synthesizes():
    """preamble runs tune R1, which SYNTHESIZES to score each reference by real
    ECAPA similarity — so it cannot run on the laptop with a GPU-only engine."""
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    cmd = REMOTE_TASK["preamble"].format(video="v.mp4", langs="en", stages="")
    assert cmd.startswith("dubadabidu preamble ")
    assert "--overlay config.gpu.yaml" in cmd        # engine + pod settings
    assert "--overlay config.deepseek.yaml" in cmd   # preamble runs s3 first


def test_bakeoff_rsync_does_not_clobber_the_engine_list():
    """`needed` holds the ENGINE list from engines_for_task and is consumed far
    below by _verify_ready and _install_engines. The bake-off branch reassigned
    it to a list of rsync file paths, so _install_engines iterated over paths,
    found no engine_setup snippet for "…/vocals.wav", and installed NOTHING —
    silently, since a missing snippet is a legitimate skip.

    Effect: every `remote bakeoff` on a FRESH pod reported "qwen unavailable"
    and measured zero engines. Masked because bake-offs are usually run with
    --reuse against a pod an earlier `run` had already set up. Two ICL arms
    provisioned their own pod on 2026-08-09 and both returned empty scorecards.
    """
    import re
    src = remote_source()
    body = src.split("def remote_run", 1)[1].split("\ndef ", 1)[0]
    # the engine list is assigned exactly once in remote_run
    assigns = re.findall(r"^\s*needed\s*=", body, re.M)
    assert len(assigns) == 1, (
        f"`needed` is assigned {len(assigns)}x in remote_run — the engine list "
        f"must not be reused as a scratch variable")
    assert "needed = bootstrap.engines_for_task" in body, (
        "the single assignment must be the engine list itself")
    # and it must still be the value handed to the installer
    assert "_install_engines(rp, host, port, remote, needed" in body


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
        for e in engines_for_task(cfg, task, ["en", "ru"]):
            assert e in ENGINE_MODULE, (
                f"{task}: {e!r} is not an engine name — if a path reaches "
                f"_install_engines it silently installs nothing")


# --- cost controls -------------------------------------------------------------

def test_deadline_is_recomputed_from_the_price_actually_paid():
    """--budget only ever meant dollars if assumed_price_per_hr was right. The
    ledger says it often was not: prices actually paid are 0.25, 0.26, 0.28 and
    0.49. The cap is on TIME, so on the 0.49 pod a "$10 cap" bought nearly twice
    the dollars it promised."""
    src = remote_source()
    block = src[src.index("RE-DERIVE THE DEADLINE"):]
    block = block[:block.index("arm_watchdog")]
    assert "_LAST_POD" in block, "the actual price is never consulted"
    assert "if tighter < deadline" in block, (
        "the correction must only ever SHORTEN — extending would spend past "
        "what the caller asked for")


def test_bootstrap_time_is_recorded_in_the_ledger():
    """runlog.cost_per_video_hour subtracts bootstrap_s to separate fixed setup
    from real work, and nothing ever wrote it — 15 of 17 rows carry 0, so setup
    was being billed as synthesis. 'Is batching worth it' has to be answerable
    from the ledger, not from memory."""
    src = remote_source()
    assert '"bootstrap_s": _boot_s' in src, "bootstrap_s still never written"
    assert src.index("_boot_s = round(") < src.index("rc = transport.ssh_exec(rp, host, port, full"), (
        "bootstrap must be stamped BEFORE the task runs, or it measures the task")


def test_runlog_consumes_the_field_that_is_now_written():
    from dubadabidu.core.runlog import cost_per_video_hour
    rows = [{"video_minutes": 8.0, "wall_s": 900, "bootstrap_s": 600,
             "price_per_hr": 0.30, "langs": ["ru"]}]
    out = cost_per_video_hour(rows)
    assert out["mean_bootstrap_min"] == 10.0
    # per-language cost must exclude the bootstrap: 300 s of real work, not 900
    assert out["usd_per_video_hour_per_language"] == round(
        (900 - 600) / 3600 * 0.30 / (8.0 / 60), 3)


def test_missing_engine_setup_fails_before_provisioning():
    """The overlay stack is asymmetric: REMOTE_TASK hardcodes
    `--overlay config.gpu.yaml` for the POD, but the LOCAL cfg only has it if
    the caller passed it — and the local half is what supplies engine_setup to
    the installer. Omit it and the pod provisions, bootstraps, installs nothing,
    and the bake-off reports "engine unavailable" over zero measurements.
    Happened 2026-08-13: ~10 minutes billed for a result that could not exist.
    The refusal has to come BEFORE the money starts."""
    src = remote_source()
    guard = src.index("FAIL BEFORE PROVISIONING")
    assert guard < src.index("deadline = _deadline(rp, budget)"), \
        "the guard must run before the deadline/provision path"
    assert guard < src.index("pid, host, port = provision("), \
        "the guard must run before anything bills"
    block = src[guard:src.index("deadline = _deadline(rp, budget)")]
    assert "--overlay config.gpu.yaml" in block, \
        "the error must name the fix, not just the symptom"
    assert "if reuse:" in block, \
        "a live pod may already have the engine — reuse should warn, not refuse"


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


def test_payload_sends_only_the_public_half_of_the_ssh_key(tmp_path):
    from dubadabidu.infra.remote.runpod import pod_payload
    key = tmp_path / "k"
    key.write_text("PRIVATE-NEVER-SENT")
    (tmp_path / "k.pub").write_text("ssh-ed25519 AAAA test\n")
    body = pod_payload(Runpod(ssh_key=str(key)), "x")
    assert body["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA test"
    assert "PRIVATE-NEVER-SENT" not in str(body)


def test_a_proxy_only_host_is_retried_on_the_next_gpu_type(monkeypatch):
    """REST v2 cannot require a public IP. A pod that sits RUNNING with only
    the ssh.runpod.io proxy (no rsync possible) must be replaced by a pod on
    the NEXT gpu type, not waited on for the whole provision timeout."""
    import time as _t
    from dubadabidu.infra.remote import orchestrator, provider, runpod
    starts: list[str] = []

    def fake_once(rp, deadline):
        starts.append(rp.gpu_type_ids[0])
        if len(starts) == 1:
            raise provider.NoDirectSsh("pod p1 (NVIDIA RTX 4000 Ada Generation) RUNNING 150s ...")
        return "p2", "1.2.3.4", 2222, "NVIDIA RTX A4500"
    monkeypatch.setattr(orchestrator, "_provision_once", fake_once)
    monkeypatch.setattr(runpod, "get_pod", lambda pid: {"cost": 0.25})
    rp = Runpod(gpu_type_ids=["NVIDIA RTX A4000",
                                         "NVIDIA RTX 4000 Ada Generation",
                                         "NVIDIA RTX A4500"])
    assert orchestrator.provision(rp, _t.time() + 3600) == ("p2", "1.2.3.4", 2222)
    assert starts == ["NVIDIA RTX A4000", "NVIDIA RTX A4500"]


def test_failed_task_holds_the_pod_and_next_run_reuses_it(tmp_path, monkeypatch):
    """A failure after a 20+ min bootstrap must not throw the pod away: it is
    held (state file flag) and the next remote command picks it up."""
    import json
    from dubadabidu.infra.remote import state
    monkeypatch.setattr(state, "STATE_FILE", tmp_path / "s.json")
    assert state._held_pod() is None
    state._save_state({"pod_id": "abc", "created_at": 0})
    assert state._held_pod() is None            # plain tracked pod: not held
    state._save_state({"pod_id": "abc", "created_at": 0, "held_for_hotfix": True})
    assert state._held_pod() == "abc"
    assert json.loads((tmp_path / "s.json").read_text())["pod_id"] == "abc"
    assert Runpod().hold_on_failure_min > 0


def test_proxy_only_clock_starts_at_container_start():
    """v2 reports RUNNING while the image is still pulling (runtime null);
    only a started container without ssh.direct is a proxy-only host."""
    from dubadabidu.infra.remote.runpod import container_started
    assert not container_started({"status": "RUNNING", "runtime": None})
    assert container_started({"status": "RUNNING",
                              "runtime": {"uptime": 3, "ports": []}})


def test_no_capacity_anywhere_waits_and_resweeps(monkeypatch):
    """All types empty -> sleep and sweep the list again; succeed when stock
    returns, give up after capacity_wait_min."""
    import pytest
    from dubadabidu.infra.remote import runpod
    calls: list[str] = []
    sleeps: list[float] = []

    def fake(method, path, body=None):
        calls.append(body["gpu"]["id"])
        if len(calls) <= 4:          # first full sweep of 2 types x 2 sweeps
            raise RuntimeError("RunPod POST /pods -> HTTP 400: no capacity")
        return {"id": "p9"}
    monkeypatch.setattr(runpod, "_req", fake)
    monkeypatch.setattr(runpod.time, "sleep", sleeps.append)
    rp = Runpod(gpu_type_ids=["a", "b"],
          capacity_wait_min=10, capacity_retry_s=60)
    assert runpod.create_pod(rp) == {"id": "p9"}
    assert calls == ["a", "b", "a", "b", "a"] and sleeps == [60, 60]

    calls.clear()
    monkeypatch.setattr(runpod, "_req", lambda m, p, b=None: (_ for _ in ()).throw(
        RuntimeError("RunPod POST /pods -> HTTP 400: no capacity")))
    with pytest.raises(RuntimeError, match="no capacity"):
        runpod.create_pod(derive(rp, capacity_wait_min=0))


class _FakeRpi:
    """Stands in for pipeline.runpod_infra in the batch tests: records the
    call and marks the pod stages done in each "synced-back" manifest."""

    def __init__(self, tmp_path, fail: frozenset[str] = frozenset()) -> None:
        self.tmp = tmp_path
        self.calls: list[tuple[list[str], str, bool, bool]] = []
        self.fail = fail

    def remote_run(self, cfg: dict, videos: list[str], langs: list[str],
                   task: str, budget: float | None, keep_alive: bool,
                   reuse: bool, overlays: list[str], stages: str) -> bool:
        import json
        self.calls.append((list(videos), stages, keep_alive, reuse))
        for v in videos:
            st = {} if v in self.fail else {
                f"s{n}_{lg}": "done" for n in (4, 5) for lg in langs}
            d = self.tmp / v
            d.mkdir(exist_ok=True)
            (d / "manifest.json").write_text(json.dumps({"stages": st}))
        return not self.fail


def _batch_args(**kw: object):
    import argparse
    base = {"from_stage": "s1_extract", "to_stage": "s8_mux", "budget": None,
            "keep_alive": False, "reuse": False, "overlay": []}
    base.update(kw)
    return argparse.Namespace(**base)


def test_multi_video_remote_run_is_one_pod_one_call(monkeypatch, tmp_path):
    """All videos go to the pod in ONE remote_run (one process: models load
    once); s3 runs locally for all first, s6..s8 for all after."""
    from dubadabidu.cli import remote as dub
    local: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(dub, "_run_local",
                        lambda cfg, v, langs, st: local.append((v, tuple(st))))
    rpi = _FakeRpi(tmp_path)
    cfg = settings({"work_dir": str(tmp_path)})
    assert dub._remote_run_batch(cfg, ["a", "b", "c"], ["en"], _batch_args(), rpi)
    assert rpi.calls == [(["a", "b", "c"], "--from s4_synthesize --to s5_fit",
                          False, False)]
    assert [v for v, st in local if st == ("s3_translate",)] == ["a", "b", "c"]
    assert [v for v, st in local if st[:1] == ("s6_mix",)] == ["a", "b", "c"]


def test_batch_video_that_did_not_finish_on_the_pod_skips_its_finish(
        monkeypatch, tmp_path):
    from dubadabidu.cli import remote as dub
    local: list[str] = []
    monkeypatch.setattr(dub, "_run_local",
                        lambda cfg, v, langs, st: local.append(f"{v}:{st[0]}")
                        if st else None)
    rpi = _FakeRpi(tmp_path, fail=frozenset({"b"}))
    cfg = settings({"work_dir": str(tmp_path)})
    assert not dub._remote_run_batch(cfg, ["a", "b", "c"], ["en"],
                                     _batch_args(), rpi)
    assert "a:s6_mix" in local and "c:s6_mix" in local
    assert "b:s6_mix" not in local


def test_pod_command_keeps_going_and_carries_every_video():
    """The pod-side `run` must not stop at the first bad video, and every
    path is quoted (lesson names have spaces)."""
    import shlex
    from dubadabidu.infra.remote.bootstrap import REMOTE_TASK
    assert "--keep-going" in REMOTE_TASK["run"]
    vids = ["input/a b.mp4", "input/c.mp4"]
    cmd = REMOTE_TASK["run"].format(
        video=" ".join(shlex.quote(v) for v in vids), langs="en", stages="")
    assert "'input/a b.mp4' input/c.mp4" in cmd


def test_run_keep_going_continues_past_a_failing_video(monkeypatch, tmp_path):
    """dub.py run with --keep-going: video 1 raises, video 2 still runs, and
    the process exits non-zero naming the failure."""
    import sys
    import pytest
    from dubadabidu import cli as dub
    seen: list[str] = []

    def stage(cfg, v, langs):
        seen.append(v)
        if v.endswith("bad.mp4"):
            raise RuntimeError("synth exploded")
    monkeypatch.setattr(dub, "STAGES", {n: stage for n in dub.ORDER})
    for n in ("bad", "good"):
        (tmp_path / f"{n}.mp4").write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "dubadabidu", "run", str(tmp_path / "bad.mp4"), str(tmp_path / "good.mp4"),
        "--langs", "en", "--from", "s4_synthesize", "--to", "s4_synthesize",
        "--keep-going"])
    with pytest.raises(SystemExit, match="1 video"):
        dub.main()
    assert seen == [str(tmp_path / "bad.mp4"), str(tmp_path / "good.mp4")]
