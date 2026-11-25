# SPDX-License-Identifier: AGPL-3.0-or-later
"""remote_run and provision: deadlines, reuse, holds, ledger rows."""
import time
from dubadabidu.infra.remote.orchestrator import _deadline
from dubadabidu.core.config.schema import Runpod, validate
from tests.paths import config, remote_source


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


def test_provisioning_logs_the_price_not_a_gpu_name():
    """REST v1 GET /pods/{id} returns an EMPTY machine object and no gpuTypeId
    (checked 2026-08-01), so the card's identity is not retrievable. Price is
    the better signal anyway — it is what the budget is spent in."""
    from dubadabidu.infra.remote.orchestrator import provision
    import inspect
    src = inspect.getsource(provision)
    assert "provider.active().price(info)" in src and "OVER the budgeted" in src


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
