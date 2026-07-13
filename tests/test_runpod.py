import time
from pipeline.runpod_infra import _deadline, ssh_target, DEFAULTS


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

def test_ssh_target_list_shape():
    pod = {"publicIp": "1.2.3.4",
           "portMappings": [{"privatePort": 22, "publicPort": 40022}]}
    assert ssh_target(pod) == ("1.2.3.4", 40022)


def test_ssh_target_dict_shape():
    pod = {"ports": {"22/tcp": {"ip": "5.6.7.8", "publicPort": 50022}}}
    assert ssh_target(pod) == ("5.6.7.8", 50022)


def test_ssh_target_runtime_nested():
    pod = {"ip": "9.9.9.9",
           "runtime": {"ports": [{"privatePort": 22, "publicPort": 12345}]}}
    assert ssh_target(pod) == ("9.9.9.9", 12345)


def test_ssh_target_not_ready():
    assert ssh_target({"desiredStatus": "PENDING", "ports": []}) is None
    assert ssh_target({"ports": [{"privatePort": 8888, "publicPort": 1}]}) is None
