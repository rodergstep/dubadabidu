# SPDX-License-Identifier: AGPL-3.0-or-later
"""RunPod: the REST v2 client, pod payload and response parsing, the
pod-side self-destruct, and RunPodProvider — everything that would change
for another GPU provider (docs/MIGRATION_VAST.md)."""
from __future__ import annotations
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dubadabidu.infra.remote import transport

log = logging.getLogger("dubadabidu.runpod")


# REST v2 since 2026-11-19. v1 (rest.runpod.io/v1) was RETIRED 2026-11-15 and
# its host no longer resolves; every `remote` command failed on DNS until this
# moved. v2 nests the create body (gpu/mounts), takes ONE gpu id per request,
# has no spot pods and reports state as `status`. Migration guide:
# https://docs.runpod.io/api-reference-v2/migrate-from-v1
BASE = "https://api.runpod.io/v2"


DEFAULTS = {
    # cheapest adequate community GPUs (>=16GB), in priority order; the API
    # picks the first available. Prices ~$0.16-0.22/hr (checked 2026-07).
    "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                     "NVIDIA RTX A4500", "NVIDIA RTX A4000"],
    "gpu_count": 1,
    # Ubuntu 24.04 => python3.12 (the old py3.10 image capped numpy at 2.2.6);
    # cu129 base image; the torch in the venv is cu130 and needs a 13.0 driver.
    # Verified present on Docker Hub 2026-08-02 (stable, not an -rc tag).
    "image": "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
    # Filters to hosts whose DRIVER supports these CUDA versions. Without it the
    # scheduler handed us a CUDA 12.4 driver, which caps torch at 2.6.0. Enum is
    # 11.8..13.0 (RunPod OpenAPI). Empty/absent => no filter, old behaviour.
    # TRADE-OFF: every entry removed here shrinks the host pool, and this project
    # has lost runs to capacity before. Widen this list first if provisioning
    # starts failing with "no instances currently available".
    "min_cuda_version": "13.0",   # driver floor >= torch's cu-build (cu130)
    "allowed_cuda_versions": [],  # v1-style exact list; mutually exclusive (400)
    "container_disk_gb": 30,
    "volume_gb": 0,               # ephemeral; all state rsync'd back
    "cloud_type": "SECURE",       # COMMUNITY had no on-demand capacity (FINDINGS §3)
    # REST v2 has NO spot pods. Kept as a key only so an old overlay that sets
    # it is reported instead of silently ignored (create_pod warns).
    "interruptible": False,
    "ports": ["22/tcp"],
    "budget_usd": 10.0,           # hard session cap (overridden by --budget / spec)
    "assumed_price_per_hr": 0.25, # conservative: deadline = budget / this
    "max_runtime_hours": 3.0,     # absolute ceiling regardless of budget
    "ssh_key": "~/.ssh/id_ed25519_runpod",
    "ssh_user": "root",
    "remote_dir": "~/dubadabidu",
    "provision_timeout_s": 420,   # wait this long for the pod to reach RUNNING+SSH
    # HOTFIX HOLD: when the task fails on a pod that bootstrapped, keep it up
    # this many minutes (pod-side watchdog re-armed to it) instead of
    # terminating, so the fix is tested on the same pod with no new bootstrap.
    # The next `remote` command reuses it automatically. 0 = terminate always.
    "hold_on_failure_min": 30,
    # no GPU type has capacity: re-sweep the list every capacity_retry_s for
    # up to capacity_wait_min before failing (creating nothing costs nothing)
    "capacity_wait_min": 15,
    "capacity_retry_s": 60,
    # per-engine install snippets for the bake-off (git-clone challengers —
    # THIRD_PARTY.md). Each runs on the pod in that engine's OWN venv
    # (installed into .venv by engine_install_cmd — the per-engine venvs were
    # untouchable by construction); empty => that engine is marked unavailable
    # and skipped. Fill with PINNED commands once validated.
    "engine_setup": {},           # e.g. {"qwen": "git clone ... && pip install ..."}
}


# ---------------- REST client ----------------

def _key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        raise SystemExit("RUNPOD_API_KEY not set (add it to .env). Needed only "
                         "for the `remote` commands.")
    return k


def _req(method: str, path: str, body: dict | None = None) -> object:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Authorization": "Bearer " + _key(),
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt.strip() else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod {method} {path} -> HTTP {e.code}: "
                           f"{e.read().decode()[:300]}") from e


def pod_payload(rp: dict, gpu_id: str) -> dict:
    """REST v2 create body for ONE gpu type. v2 is strict (unknown keys -> 422),
    so only documented fields are sent."""
    gpu: dict = {"id": gpu_id, "count": int(rp["gpu_count"])}
    # Only send it when set — an empty list would filter to nothing. The wheel
    # in uv.lock (cu130) must be <= min() of this list; tests assert it.
    # v2 rejects both together (400). The floor is preferred: an exact list
    # excludes every newer driver, which v2's catalog showed leaves no capacity.
    if rp.get("allowed_cuda_versions"):
        gpu["allowedCudaVersions"] = [str(v) for v in rp["allowed_cuda_versions"]]
    elif rp.get("min_cuda_version"):
        gpu["minCudaVersion"] = str(rp["min_cuda_version"])
    payload: dict = {
        "name": rp.get("name", "dubadabidu"),
        "image": rp["image"],
        "gpu": gpu,
        "cloud": rp["cloud_type"],
        "disk": int(rp["container_disk_gb"]),
        # v1 defaulted to "8888/http,22/tcp"; v2 has no default, and without
        # 22/tcp there is no ssh.direct mapping at all.
        "ports": list(rp["ports"]),
    }
    if rp.get("registry_auth_id"):
        # private image (P3): RunPod container-registry credential id
        payload["registry"] = {"authId": rp["registry_auth_id"]}
    if rp.get("network_volume_id"):
        # persistent volume at /workspace — remote_dir moves onto it
        payload["mounts"] = {"network": [{"volumeId": rp["network_volume_id"],
                                          "path": "/workspace"}]}
    elif rp.get("volume_gb"):
        payload["mounts"] = {"persistent": {"size": max(10, int(rp["volume_gb"])),
                                            "path": "/workspace"}}
    # inject ONLY the low-privilege translation key into the pod env (encrypted
    # in transit) so s3 can run on the pod. The RunPod full-access key is never
    # sent — it stays local, used only to drive this API.
    env: dict[str, str] = {}
    tk = os.environ.get("TRANSLATE_API_KEY")
    if tk:
        env["TRANSLATE_API_KEY"] = tk
    # SSH. v1 started sshd implicitly; v2 only injects PUBLIC_KEY when
    # startSsh is true AND keys are registered on the account — otherwise the
    # pod comes up RUNNING with a port mapping and sshd never answers (measured
    # 2026-11-19: 9 min, "SSH never became reachable"). Sending the PUBLIC half
    # of the key this client connects with makes it independent of account
    # state; RunPod images start sshd from PUBLIC_KEY.
    pub = Path(str(rp.get("ssh_key", "~/.ssh/id_ed25519_runpod")) + ".pub").expanduser()
    if pub.exists():
        env["PUBLIC_KEY"] = pub.read_text(encoding="utf-8").strip()
    payload["startSsh"] = True
    if env:
        payload["env"] = env
    return payload


def create_pod(rp: dict) -> dict:
    """Create a pod, trying gpu_type_ids IN ORDER (cheapest first).

    v1 took the whole list plus gpuTypePriority "custom" and did the fallback
    server-side; v2 takes exactly one gpu id, so the ordering now lives here —
    which also settles the old "does custom fall back?" doubt by construction.
    A 400 is capacity (or a cross-field rule) for that type: try the next one.
    422 is a schema error and 402 is no balance: never retry those."""
    if rp.get("interruptible"):
        log.warning("runpod.interruptible is set, but REST v2 has no spot pods "
                    "— creating an on-demand pod")
    ids = rp["gpu_type_ids"]
    ids = [ids] if isinstance(ids, str) else list(ids)
    # No type has capacity -> wait and sweep the list again instead of
    # failing: SECURE stock churns by the minute (2026-11-24: all ten types
    # empty at 15:38, after pods had been created on four of them in the hour
    # before). Creating nothing costs nothing, so waiting is free.
    wait_s = float(rp.get("capacity_wait_min", 15)) * 60
    every = float(rp.get("capacity_retry_s", 60))
    give_up = time.time() + wait_s
    last: Exception | None = None
    sweep = 0
    while True:
        sweep += 1
        for gid in ids:
            try:
                return _req("POST", "/pods", pod_payload(rp, gid))
            except RuntimeError as e:
                if "HTTP 400" in str(e) or "HTTP 409" in str(e):
                    log.info("no pod for %s (%s); trying next type", gid,
                             str(e)[:160])
                    last = e
                    continue
                raise
        if time.time() + every > give_up:
            break
        log.warning("no capacity for any GPU type (sweep %d) — retrying in "
                    "%.0f s (gives up in %.0f min)", sweep, every,
                    (give_up - time.time()) / 60)
        time.sleep(every)
    raise RuntimeError(f"no capacity for any of {ids}: {last}")


# ---- pod-object accessors: one place that knows the response shape ----
# v2 names first, v1 names as fallback so a cached/test object still reads.

def pod_status(p: dict) -> str:
    return str(p.get("status") or p.get("desiredStatus") or "").upper()


def container_started(p: dict) -> bool:
    """v2: `runtime` is null until the container is up (image pulled)."""
    return bool(p.get("runtime"))


def pod_price(p: dict) -> float | None:
    v = p.get("cost", p.get("costPerHr"))
    return None if v is None else float(v)


def pod_vcpu(p: dict) -> float | None:
    return (p.get("gpu") or p.get("cpu") or {}).get("vcpuCount", p.get("vcpuCount"))


def pod_ram_gb(p: dict) -> float | None:
    return (p.get("gpu") or p.get("cpu") or {}).get("memory", p.get("memoryInGb"))


def list_pods() -> list[dict]:
    """Every pod on the account, following v2 cursor pagination."""
    out: list[dict] = []
    cursor = None
    while True:
        page = _req("GET", "/pods" + (f"?cursor={cursor}" if cursor else ""))
        out.extend(page.get("pods", []) if isinstance(page, dict) else [])
        pg = page.get("pagination", {}) if isinstance(page, dict) else {}
        cursor = pg.get("nextCursor")
        if not (pg.get("hasNextPage") and cursor):
            return out


def get_pod(pid: str) -> dict:
    return _req("GET", f"/pods/{pid}")


# ---------------- SSH parsing / helpers ----------------

def ssh_target(pod: dict) -> tuple[str, int] | None:
    """Extract (public_ip, ssh_port) from a pod GET response. The live REST v1
    shape is publicIp + portMappings {"22": <publicPort>}; other shapes are
    tolerated as fallbacks. Returns None until SSH is exposed. Never raises on
    an unexpected shape (a crash here used to leak the pod)."""
    direct = (pod.get("ssh") or {}).get("direct") or {}
    if direct.get("host") and direct.get("port"):          # REST v2
        try:
            return str(direct["host"]), int(direct["port"])
        except (TypeError, ValueError):
            pass
    for rpt in (pod.get("runtime") or {}).get("ports") or []:   # v2 runtime
        if (isinstance(rpt, dict) and str(rpt.get("private")) == "22"
                and rpt.get("ip") and rpt.get("public")):
            try:
                return str(rpt["ip"]), int(rpt["public"])
            except (TypeError, ValueError):
                pass
    ip = pod.get("publicIp") or pod.get("ip")
    pm = pod.get("portMappings")
    if isinstance(pm, dict):                       # {"22": 10345}
        for k, v in pm.items():
            if str(k).startswith("22") and ip and v:
                try:
                    return ip, int(v)
                except (TypeError, ValueError):
                    pass
    # fallbacks: [{privatePort:22, publicPort:...}] or runtime.ports
    ports = pod.get("ports")
    if not isinstance(ports, list) or (ports and not isinstance(ports[0], dict)):
        ports = (pod.get("runtime") or {}).get("ports") or []
    for p in ports if isinstance(ports, list) else []:
        if isinstance(p, dict) and str(p.get("privatePort") or "").startswith("22"):
            host, pub = ip or p.get("ip"), p.get("publicPort")
            if host and pub:
                return host, int(pub)
    return None


WATCHDOG_PID = "/tmp/dubadabidu_watchdog.pid"


def arm_pod_watchdog(rp: dict, host: str, port: int, seconds: float) -> None:
    """Independent pod-side self-destruct: a detached process that, after
    `seconds`, makes the pod remove ITSELF (runpodctl is self-authenticated on
    RunPod pods). This is the backstop for the one gap the client-side design
    can't cover — the LOCAL orchestrator dying (SIGKILL, laptop sleep) before
    its finally runs. Best-effort: if runpodctl is absent it halts the container.
    The client-side terminate still runs on the normal path; this only matters
    when it never gets the chance.

    RE-ARMABLE, and that is load-bearing for --reuse. The watchdog used to be
    armed exactly once, at provision. course.py then attaches ONE pod for every
    video's s4 (--reuse --keep-alive), and each of those calls recomputes a
    FRESH deadline locally while the pod is still counting down from video 1's.
    A course longer than max_runtime_hours therefore had its pod self-destruct
    mid-run with the local side believing it had hours left — paying for
    everything up to that point and resuming from cache on the next attempt.
    Killing the previous sleeper (its pid is parked at WATCHDOG_PID) before
    starting a new one makes "the deadline this run computed" the only one in
    force. `kill` targets the `sh -c` parent, so the pending runpodctl never
    runs even though the orphaned `sleep` lives out its timer."""
    secs = max(60, int(seconds))
    cmd = (f"[ -f {WATCHDOG_PID} ] && kill \"$(cat {WATCHDOG_PID})\" "
           f"2>/dev/null; "
           f"nohup sh -c 'sleep {secs}; "
           f"runpodctl remove pod $RUNPOD_POD_ID || shutdown -h now || "
           f"kill -9 -1' </dev/null >/tmp/watchdog.log 2>&1 & "
           f"echo $! > {WATCHDOG_PID}")
    try:
        transport.ssh_exec(rp, host, port, cmd, timeout=30)
        log.info("pod self-destruct watchdog armed (~%.0f min)", secs / 60)
    except Exception as e:
        log.warning("could not arm pod watchdog (%s) — relying on client-side "
                    "termination only", e)


class RunPodProvider:
    """Provider for RunPod REST v2 (the module functions above)."""
    name = "runpod"
    key_env = "RUNPOD_API_KEY"

    def create(self, rp: dict) -> dict:
        return create_pod(rp)

    def get(self, pid: str) -> dict:
        return get_pod(pid)

    def destroy(self, pid: str) -> None:
        _req("DELETE", f"/pods/{pid}")

    def list_all(self) -> list[dict]:
        return list_pods()

    def status(self, pod: dict) -> str:
        return pod_status(pod)

    def container_started(self, pod: dict) -> bool:
        return container_started(pod)

    def ssh_target(self, pod: dict) -> tuple[str, int] | None:
        return ssh_target(pod)

    def price(self, pod: dict) -> float | None:
        return pod_price(pod)

    def vcpu(self, pod: dict) -> float | None:
        return pod_vcpu(pod)

    def ram_gb(self, pod: dict) -> float | None:
        return pod_ram_gb(pod)

    def arm_watchdog(self, rp: dict, host: str, port: int, seconds: float) -> None:
        arm_pod_watchdog(rp, host, port, seconds)
