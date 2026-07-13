"""M2 — RunPod lifecycle automation (AUTOPILOT.md infra agent).

Provision a spot GPU pod -> rsync the project up -> run a task remotely ->
rsync results back -> ALWAYS terminate. The pod is cattle; all state returns to
this repo. Uses the REST v1 API (rest.runpod.io/v1, Bearer auth).

SAFETY (a leaked pod burns money):
  - The pod id is persisted to work/.runpod_active.json the instant it is
    created, BEFORE anything else — so a crash is recoverable with `remote kill`.
  - remote_run() terminates in a finally block no matter how the run exits.
  - A wall-clock DEADLINE derived from the budget cap ($/hr -> max hours) bounds
    the run; the remote command is also wrapped in `timeout` so the job self-caps
    even if this process dies.
  - Every `remote` invocation first sweeps orphaned pods from the state file.
  - RUNPOD_API_KEY is read from the env/.env; it is NEVER sent to the pod
    (rsync excludes .env) and never logged.
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("dubadabidu.runpod")

BASE = "https://rest.runpod.io/v1"
STATE_FILE = Path("work/.runpod_active.json")

DEFAULTS = {
    # cheapest adequate community GPUs (>=16GB), in priority order; the API
    # picks the first available. Prices ~$0.16-0.22/hr (checked 2026-07).
    "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                     "NVIDIA RTX A4500", "NVIDIA RTX A4000"],
    "gpu_count": 1,
    "image": "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
    "container_disk_gb": 30,
    "volume_gb": 0,               # ephemeral; all state rsync'd back
    "cloud_type": "COMMUNITY",
    "interruptible": True,        # spot — cheapest, fine (resumable + cached)
    "ports": ["22/tcp"],
    "budget_usd": 10.0,           # hard session cap (overridden by --budget / spec)
    "assumed_price_per_hr": 0.25, # conservative: deadline = budget / this
    "max_runtime_hours": 3.0,     # absolute ceiling regardless of budget
    "ssh_key": "~/.ssh/id_ed25519_runpod",
    "ssh_user": "root",
    "remote_dir": "~/dubadabidu",
    "provision_timeout_s": 420,   # wait this long for the pod to reach RUNNING+SSH
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
                           f"{e.read().decode()[:300]}")


def create_pod(rp: dict) -> dict:
    payload = {
        "name": rp.get("name", "dubadabidu"),
        "imageName": rp["image"],
        "gpuTypeIds": rp["gpu_type_ids"],
        "gpuTypePriority": "availability",
        "gpuCount": rp["gpu_count"],
        "cloudType": rp["cloud_type"],
        "interruptible": rp["interruptible"],
        "containerDiskInGb": rp["container_disk_gb"],
        "ports": rp["ports"],
        "supportPublicIp": True,
    }
    if rp.get("volume_gb"):
        payload["volumeInGb"] = rp["volume_gb"]
        payload["volumeMountPath"] = "/workspace"
    # inject ONLY the low-privilege translation key into the pod env (encrypted
    # in transit) so s3 can run on the pod. The RunPod full-access key is never
    # sent — it stays local, used only to drive this API.
    tk = os.environ.get("TRANSLATE_API_KEY")
    if tk:
        payload["env"] = {"TRANSLATE_API_KEY": tk}
    return _req("POST", "/pods", payload)


def get_pod(pid: str) -> dict:
    return _req("GET", f"/pods/{pid}")


def terminate_pod(pid: str) -> None:
    _req("DELETE", f"/pods/{pid}")
    log.info("terminated pod %s", pid)


# ---------------- state file (leak protection) ----------------

def _save_state(d: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def _terminate_tracked() -> None:
    """THE single cleanup path: terminate + clear whatever pod is recorded in
    the state file. Idempotent and crash-safe — because the pod id is persisted
    on create, this cleans up even when provisioning dies mid-flight (the local
    pid variable is useless then). Every finally block and orphan-sweep calls
    this, so termination never depends on a return value."""
    if not STATE_FILE.exists():
        return
    try:
        pid = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("pod_id")
    except (OSError, json.JSONDecodeError):
        _clear_state(); return
    if pid:
        try:
            terminate_pod(pid)
        except Exception as e:
            log.error("TERMINATE FAILED for %s: %s — kill it in the RunPod "
                      "console NOW", pid, e)
    _clear_state()


def sweep_orphans() -> None:
    """Clean up a pod left in the state file by a prior run. Called at the start
    of every `remote` command and exposed as `remote kill`."""
    if STATE_FILE.exists():
        log.warning("state file present — terminating tracked pod before start")
    _terminate_tracked()


# ---------------- SSH parsing / helpers ----------------

def ssh_target(pod: dict) -> tuple[str, int] | None:
    """Extract (public_ip, ssh_port) from a pod GET response. The live REST v1
    shape is publicIp + portMappings {"22": <publicPort>}; other shapes are
    tolerated as fallbacks. Returns None until SSH is exposed. Never raises on
    an unexpected shape (a crash here used to leak the pod)."""
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


def _ssh_base(rp: dict, host: str, port: int) -> list[str]:
    key = os.path.expanduser(rp["ssh_key"])
    return ["ssh", "-i", key, "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15", f"{rp['ssh_user']}@{host}"]


def ssh_exec(rp: dict, host: str, port: int, cmd: str, timeout: int = 7200) -> int:
    full = _ssh_base(rp, host, port) + [cmd]
    return subprocess.run(full, timeout=timeout).returncode


def wait_ssh(rp: dict, host: str, port: int, deadline: float) -> bool:
    while time.time() < deadline:
        r = subprocess.run(_ssh_base(rp, host, port) + ["true"],
                           capture_output=True, timeout=30)
        if r.returncode == 0:
            return True
        time.sleep(8)
    return False


def rsync(rp: dict, src: str, dst: str) -> None:
    key = os.path.expanduser(rp["ssh_key"])
    subprocess.run([
        "rsync", "-az", "--delete-excluded",
        "--exclude", ".venv", "--exclude", "__pycache__", "--exclude", ".env",
        "--exclude", ".git",
        "-e", f"ssh -i {key} -o StrictHostKeyChecking=no "
              f"-o UserKnownHostsFile=/dev/null",
        src, dst], check=True)


# ---------------- provisioning + orchestration ----------------

def _deadline(rp: dict, budget_usd: float) -> float:
    hours = min(budget_usd / max(rp["assumed_price_per_hr"], 0.01),
                rp["max_runtime_hours"])
    return time.time() + hours * 3600


def provision(rp: dict, deadline: float) -> tuple[str, str, int]:
    """Create a pod and wait for SSH. Writes state BEFORE waiting so a hang is
    recoverable. Returns (pod_id, host, port). Terminates + raises on timeout."""
    pod = create_pod(rp)
    pid = pod.get("id") or pod.get("podId")
    if not pid:
        raise RuntimeError(f"no pod id in create response: {str(pod)[:200]}")
    _save_state({"pod_id": pid, "created_at": time.time()})
    log.info("pod %s created; waiting for RUNNING + SSH ...", pid)
    # any failure past this point terminates the pod (via the state file) so a
    # crash — including an unexpected pod shape — can never leak a live box
    try:
        end = min(deadline, time.time() + rp["provision_timeout_s"])
        target = None
        while time.time() < end:
            p = get_pod(pid)
            status = p.get("desiredStatus") or p.get("status")
            target = ssh_target(p)
            if target:
                break
            if status in ("EXITED", "TERMINATED", "FAILED"):
                raise RuntimeError(f"pod entered {status} before SSH came up")
            time.sleep(10)
        if not target:
            raise RuntimeError("pod did not expose SSH within provision_timeout_s")
        host, port = target
        if not wait_ssh(rp, host, port, min(deadline, time.time() + 240)):
            raise RuntimeError("SSH never became reachable")
    except BaseException:
        _terminate_tracked()   # crash-safe: reads pid from state, not a local var
        raise
    log.info("pod %s SSH ready at %s:%d", pid, host, port)
    return pid, host, port


REMOTE_SETUP = (
    "set -e; cd {dir}; apt-get install -y ffmpeg rsync >/dev/null 2>&1 || true; "
    "python3 -m venv .venv 2>/dev/null || true; . .venv/bin/activate; "
    "pip install -q chatterbox-tts==0.1.7 && pip install -q -e '.[dev]'; "
    "python -c \"import torch;print('CUDA:',torch.cuda.is_available())\"")

REMOTE_TASK = {
    "bakeoff": "dubadabidu bakeoff {video} --langs {langs} "
               "--overlay config.gpu.yaml",
    "autopilot": "dubadabidu autopilot {video} --langs {langs} "
                 "--overlay config.gpu.yaml --overlay config.deepseek.yaml",
    "run": "dubadabidu run {video} --langs {langs} "
           "--overlay config.gpu.yaml --overlay config.deepseek.yaml",
}


def remote_run(cfg: dict, video: str, langs: list[str], task: str,
               budget_usd: float | None = None) -> bool:
    """Full lifecycle: sweep -> provision -> sync up -> run -> sync back ->
    ALWAYS terminate. Returns True on remote task success."""
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    budget = float(budget_usd if budget_usd is not None else rp["budget_usd"])
    sweep_orphans()
    if task not in REMOTE_TASK:
        raise SystemExit(f"unknown remote task {task!r} (choose "
                         f"{list(REMOTE_TASK)})")
    deadline = _deadline(rp, budget)
    remote = rp["remote_dir"]
    hours = (deadline - time.time()) / 3600
    log.info("budget $%.2f -> auto-terminate in %.1f h", budget, hours)

    pid = None
    try:
        pid, host, port = provision(rp, deadline)
        # 1. project up (excludes .env/.venv/.git)
        rsync(rp, "./", f"{rp['ssh_user']}@{host}:{remote}/")
        # patch rsync -e for the port (host may use a nonstandard SSH port)
        key = os.path.expanduser(rp["ssh_key"])
        subprocess.run(["rsync", "-az", "--exclude", ".venv", "--exclude",
                        "__pycache__", "--exclude", ".env", "--exclude", ".git",
                        "-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=no "
                              f"-o UserKnownHostsFile=/dev/null",
                        "./", f"{rp['ssh_user']}@{host}:{remote}/"], check=True)
        # 2. install + verify CUDA
        if ssh_exec(rp, host, port, REMOTE_SETUP.format(dir=remote),
                    timeout=1800) != 0:
            raise RuntimeError("remote setup failed (see output above)")
        # 3. run the task, self-capped by remote `timeout` to the deadline
        secs = max(60, int(deadline - time.time()))
        cmd = REMOTE_TASK[task].format(video=video, langs=",".join(langs))
        full = (f"cd {remote}; . .venv/bin/activate; "
                f"export TRANSLATE_API_KEY=${{TRANSLATE_API_KEY:-x}}; "
                f"timeout {secs} {cmd}")
        rc = ssh_exec(rp, host, port, full, timeout=secs + 120)
        # 4. results back (work/ + output/) regardless of task rc
        for sub in ("work", "output"):
            subprocess.run(["rsync", "-az", "-e",
                            f"ssh -i {key} -p {port} -o StrictHostKeyChecking=no "
                            f"-o UserKnownHostsFile=/dev/null",
                            f"{rp['ssh_user']}@{host}:{remote}/{sub}/",
                            f"./{sub}/"], check=False)
        log.info("remote task %s exited rc=%d; results synced back", task, rc)
        return rc == 0
    finally:
        _terminate_tracked()   # state-file driven: fires even if provision crashed


def smoke_test(cfg: dict) -> bool:
    """Cheapest possible lifecycle validation (~$0.02): provision -> print the
    raw pod JSON (to learn field shapes) + nvidia-smi over SSH -> terminate.
    Run this ONCE before trusting the full remote_run path."""
    rp = {**DEFAULTS, **cfg.get("runpod", {})}
    sweep_orphans()
    deadline = time.time() + rp["provision_timeout_s"] + 120
    pid = None
    try:
        pid, host, port = provision(rp, deadline)
        print(f"[smoke] pod {pid} reachable at {host}:{port}")
        print("[smoke] raw pod object (field-shape reference):")
        print(json.dumps(get_pod(pid), indent=2)[:1500])
        rc = ssh_exec(rp, host, port,
                      "nvidia-smi --query-gpu=name,memory.total "
                      "--format=csv,noheader", timeout=60)
        print(f"[smoke] nvidia-smi rc={rc}")
        return rc == 0
    finally:
        _terminate_tracked()   # state-file driven: fires even if provision crashed
        print("[smoke] cleanup done (pod terminated, state cleared)")


def status() -> None:
    """Show the state file + all pods currently on the account (leak check)."""
    st = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None
    print("state file:", st or "(none)")
    try:
        pods = _req("GET", "/pods")
        print(f"pods on account: {len(pods)}")
        for p in (pods if isinstance(pods, list) else []):
            print(f"  {p.get('id')}  {p.get('name')}  "
                  f"{p.get('desiredStatus') or p.get('status')}")
    except Exception as e:
        print("could not list pods:", e)
