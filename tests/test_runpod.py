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
    cmd = engine_install_cmd("~/dubadabidu", "voxcpm", "pip install -q voxcpm")
    # the snippet must run inside the ENGINE's venv, never the main .venv
    assert "venvs/voxcpm" in cmd
    assert ". venvs/voxcpm/bin/activate" in cmd
    assert ".venv/bin/activate" not in cmd.replace("venvs/voxcpm/bin/activate", "")
    assert cmd.rstrip().endswith("pip install -q voxcpm")


def test_engine_install_cmd_distinct_venvs():
    a = engine_install_cmd("~/d", "cosyvoice", "true")
    b = engine_install_cmd("~/d", "qwen", "true")
    assert "venvs/cosyvoice" in a and "venvs/cosyvoice" not in b
    assert "venvs/qwen" in b and "venvs/qwen" not in a
