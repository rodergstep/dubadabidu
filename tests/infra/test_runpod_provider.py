# SPDX-License-Identifier: AGPL-3.0-or-later
"""RunPod REST v2: pod payload, response shapes, gpu order, capacity, watchdog."""
from dubadabidu.core.config.schema import Runpod, derive
from dubadabidu.infra.remote.runpod import ssh_target
from tests.paths import config


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


def test_payload_sends_only_the_public_half_of_the_ssh_key(tmp_path):
    from dubadabidu.infra.remote.runpod import pod_payload
    key = tmp_path / "k"
    key.write_text("PRIVATE-NEVER-SENT")
    (tmp_path / "k.pub").write_text("ssh-ed25519 AAAA test\n")
    body = pod_payload(Runpod(ssh_key=str(key)), "x")
    assert body["env"]["PUBLIC_KEY"] == "ssh-ed25519 AAAA test"
    assert "PRIVATE-NEVER-SENT" not in str(body)


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
