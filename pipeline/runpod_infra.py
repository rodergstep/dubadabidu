"""M2 — RunPod lifecycle automation (AUTOPILOT.md infra agent).

Provision a spot GPU pod -> rsync the project up -> run a task remotely ->
rsync results back -> ALWAYS terminate. The pod is cattle; all state returns to
this repo. Uses the REST v1 API (rest.runpod.io/v1, Bearer auth).

SAFETY — layered so no single failure leaks a billing pod:
  1. The pod id is persisted to work/.runpod_active.json the instant it is
     created, BEFORE anything else — the state file, not a return value, is the
     source of truth for what must be killed.
  2. remote_run() terminates in a finally block on every exit path; provision()
     self-terminates on any exception. Both go through terminate_pod(), which
     RETRIES the DELETE and VERIFIES the pod left RUNNING.
  3. The state file is cleared ONLY when termination is confirmed — a failed
     terminate keeps the id so the next `remote` call / `remote kill` retries.
  4. Every `remote` invocation first sweeps the state file (orphan from a prior
     run). `remote status` lists all account pods as an authoritative check.
  5. INDEPENDENT backstop: a pod-side self-destruct watchdog (arm_pod_watchdog)
     removes the pod after the deadline even if THIS process dies — the one gap
     the client-side design cannot cover.
  6. A wall-clock DEADLINE (budget/$per-hr, capped by max_runtime_hours) bounds
     the run; the remote command is also wrapped in `timeout`.
  - RUNPOD_API_KEY is read from env/.env; NEVER sent to the pod (rsync excludes
    .env) and never logged. Only the low-privilege TRANSLATE_API_KEY reaches it.

Note: RunPod's REST create has no native pod-TTL field (per its OpenAPI schema),
so the watchdog (5) is the server-independent auto-kill.
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

from . import manifest as M

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
    # per-engine install snippets for the bake-off (git-clone challengers —
    # THIRD_PARTY.md). Each runs on the pod in the venv; empty => that engine is
    # marked unavailable and skipped. Fill with PINNED commands once validated.
    "engine_setup": {},           # e.g. {"cosyvoice": "git clone ... && pip install ..."}
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
    if rp.get("network_volume_id"):
        # persistent volume mounted at /workspace — keeps .venv across runs so
        # the ~15min chatterbox/torch install happens once (REMOTE_SETUP skips
        # reinstall when chatterbox already imports). remote_dir moves onto it.
        payload["networkVolumeId"] = rp["network_volume_id"]
        payload["volumeMountPath"] = "/workspace"
    elif rp.get("volume_gb"):
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


def terminate_pod(pid: str, retries: int = 4) -> bool:
    """DELETE the pod and CONFIRM it's gone. Returns True only when termination
    is verified (or the pod is already absent). This is the one call that must
    not fail silently — a dropped DELETE leaks a billing pod — so it retries and
    then GETs the pod to check it actually left the RUNNING state."""
    for attempt in range(retries):
        try:
            _req("DELETE", f"/pods/{pid}")
        except RuntimeError as e:
            if "HTTP 404" in str(e):            # already gone
                log.info("pod %s already absent", pid)
                return True
            log.warning("terminate %s attempt %d/%d failed: %s",
                        pid, attempt + 1, retries, e)
            time.sleep(3)
            continue
        # DELETE returned OK — verify the pod really left RUNNING
        try:
            st = (get_pod(pid).get("desiredStatus")
                  or get_pod(pid).get("status") or "").upper()
        except RuntimeError as e:
            if "HTTP 404" in str(e):            # confirmed gone
                log.info("terminated pod %s (confirmed)", pid)
                return True
            st = "?"
        if st in ("TERMINATED", "EXITED", "", "?"):
            log.info("terminated pod %s", pid)
            return True
        log.warning("pod %s still %s after DELETE; retrying", pid, st)
        time.sleep(3)
    return False


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
    if not pid:
        _clear_state(); return
    # Clear the state file ONLY when termination is confirmed. If it failed,
    # KEEP the id so the next `remote` call (or `remote kill`) retries — clearing
    # it here would erase the only record of a still-running billing pod.
    if terminate_pod(pid):
        _clear_state()
    else:
        log.error("could NOT confirm termination of %s — state file KEPT for "
                  "retry. Run `dubadabidu remote kill` or delete it in the "
                  "RunPod console.", pid)


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


def arm_pod_watchdog(rp: dict, host: str, port: int, seconds: float) -> None:
    """Independent pod-side self-destruct: a detached process that, after
    `seconds`, makes the pod remove ITSELF (runpodctl is self-authenticated on
    RunPod pods). This is the backstop for the one gap the client-side design
    can't cover — the LOCAL orchestrator dying (SIGKILL, laptop sleep) before
    its finally runs. Best-effort: if runpodctl is absent it halts the container.
    The client-side terminate still runs on the normal path; this only matters
    when it never gets the chance."""
    secs = max(60, int(seconds))
    cmd = (f"nohup sh -c 'sleep {secs}; "
           f"runpodctl remove pod $RUNPOD_POD_ID || shutdown -h now || "
           f"kill -9 -1' </dev/null >/tmp/watchdog.log 2>&1 &")
    try:
        ssh_exec(rp, host, port, cmd, timeout=30)
        log.info("pod self-destruct watchdog armed (~%.0f min)", secs / 60)
    except Exception as e:
        log.warning("could not arm pod watchdog (%s) — relying on client-side "
                    "termination only", e)


def wait_ssh(rp: dict, host: str, port: int, deadline: float) -> bool:
    while time.time() < deadline:
        r = subprocess.run(_ssh_base(rp, host, port) + ["true"],
                           capture_output=True, timeout=30)
        if r.returncode == 0:
            return True
        time.sleep(8)
    return False


def rsync(rp: dict, port: int, src: str, dst: str, check: bool = True,
          extra_excludes: tuple[str, ...] = ()) -> None:
    """rsync over SSH on the pod's MAPPED port (RunPod exposes 22 on a random
    public port — omitting -p hits default 22 and fails auth)."""
    key = os.path.expanduser(rp["ssh_key"])
    args = ["rsync", "-az"]
    for e in (".venv", "__pycache__", ".env", ".git", *extra_excludes):
        args += ["--exclude", e]
    args += ["-e", f"ssh -i {key} -p {port} -o StrictHostKeyChecking=no "
                   f"-o UserKnownHostsFile=/dev/null", src, dst]
    subprocess.run(args, check=check)


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


REMOTE_SETUP = (  # rsync/ffmpeg already installed in step 0
    "set -e; cd {dir}; "
    "python3 -m venv .venv 2>/dev/null || true; . .venv/bin/activate; "
    # skip the ~15min reinstall when deps are already present (persistent-volume
    # reuse across runs — see runpod.network_volume_id)
    "if ! python -c 'import chatterbox' 2>/dev/null; then "
    "pip install -q chatterbox-tts==0.1.7 && pip install -q -e '.[dev]'; fi; "
    # FAIL if CUDA is missing — otherwise the run would silently synth on CPU,
    # which is uselessly slow and defeats the point of renting a GPU
    "python -c 'import torch,sys; ok=torch.cuda.is_available(); "
    "print(\"CUDA:\",ok); sys.exit(0 if ok else 1)'")

REMOTE_TASK = {
    "bakeoff": "dubadabidu bakeoff {video} --langs {langs} "
               "--overlay config.gpu.yaml",
    # --no-mux: stops at s7 like `run`; the mux happens locally after sync-back
    "autopilot": "dubadabidu autopilot {video} --langs {langs} --no-mux "
                 "--overlay config.gpu.yaml --overlay config.deepseek.yaml",
    # stops at s7: the pod produces dubbed audio + subs; the final mux (a video
    # stream-copy needing the 4K source) runs LOCALLY after sync-back, so the
    # source video never uploads.
    "run": "dubadabidu run {video} --langs {langs} --to s7_subtitles "
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
    # the pod gets work/ + ref/ + code, NOT the source video — so s1/s2 must be
    # done locally first (their cached stems live in work/). Fail fast with a
    # clear message instead of wasting a pod.
    if not M.manifest_path(cfg, video).exists():
        raise SystemExit(
            f"{video}: no cached manifest — run s1+s2 locally first "
            f"(`dubadabidu preamble {video}`). The GPU pod never receives the "
            f"source video; it works from work/<video>/ audio stems.")
    wd_v = M.video_workdir(cfg, video)
    if not (wd_v / "vocals.wav").exists():
        raise SystemExit(f"{video}: work/{Path(video).stem}/vocals.wav missing — "
                         f"run s1 locally first.")
    deadline = _deadline(rp, budget)
    # with a persistent volume, put the project (incl .venv) on it so deps survive
    remote = ("/workspace/dubadabidu" if rp.get("network_volume_id")
              else rp["remote_dir"])
    hours = (deadline - time.time()) / 3600
    log.info("budget $%.2f -> auto-terminate in %.1f h", budget, hours)

    pid = None
    try:
        pid, host, port = provision(rp, deadline)
        # arm the independent pod-side self-destruct FIRST — before the long
        # install — so a crash during setup can't leave a billing pod
        arm_pod_watchdog(rp, host, port, deadline - time.time())
        # 0. the base image (runpod/pytorch, Ubuntu) lacks rsync/ffmpeg — install
        # them BEFORE the project sync (rsync-over-ssh needs rsync on the pod).
        # Retry loop: the container runs its own apt at boot and can hold the
        # dpkg lock briefly, which failed an earlier single-shot attempt.
        apt = ("export DEBIAN_FRONTEND=noninteractive; "
               "for i in 1 2 3 4 5 6; do "
               "apt-get update && apt-get install -y rsync ffmpeg && break; "
               "echo \"[apt] attempt $i failed (lock?), retrying in 10s\"; "
               "sleep 10; done; which rsync && which ffmpeg")
        if ssh_exec(rp, host, port, apt, timeout=600) != 0:
            raise RuntimeError("could not install rsync/ffmpeg on the pod after "
                               "retries (see log for the apt error)")
        # 1. project up — EXCLUDE input/ (the 4K source video is never needed on
        # the pod) and output/. The pod works from the synced work/ audio stems.
        rsync(rp, port, "./", f"{rp['ssh_user']}@{host}:{remote}/",
              extra_excludes=("input", "output"))
        # 2. install + verify CUDA (fails fast if CUDA is unavailable)
        if ssh_exec(rp, host, port, REMOTE_SETUP.format(dir=remote),
                    timeout=1800) != 0:
            raise RuntimeError("remote setup failed: dependency install error "
                               "OR CUDA unavailable (check the log's CUDA: line)")
        # 2.5 bake-off challengers (cosyvoice/indextts) are git-clone installs;
        # run each configured engine_setup snippet. A failure is non-fatal — the
        # bake-off just marks that engine unavailable and compares the rest.
        if task == "bakeoff":
            for eng in cfg.get("bakeoff", {}).get("engines", []):
                snip = rp.get("engine_setup", {}).get(eng)
                if not snip:
                    continue
                log.info("installing bake-off engine on pod: %s", eng)
                if ssh_exec(rp, host, port,
                            f"cd {remote}; . .venv/bin/activate; {snip}",
                            timeout=2400) != 0:
                    log.warning("engine %s setup failed — bake-off will mark it "
                                "unavailable", eng)
        # 3. run the task, self-capped by remote `timeout` to the deadline.
        # Pass the (low-privilege) translation key inline — an SSH session may
        # not inherit the pod's container env, and s3 needs it. Not logged.
        secs = max(60, int(deadline - time.time()))
        cmd = REMOTE_TASK[task].format(video=video, langs=",".join(langs))
        import shlex
        tk = shlex.quote(os.environ.get("TRANSLATE_API_KEY", ""))
        full = (f"cd {remote}; . .venv/bin/activate; "
                f"export TRANSLATE_API_KEY={tk}; "
                f"timeout {secs} {cmd}")
        rc = ssh_exec(rp, host, port, full, timeout=secs + 120)
        if rc != 0:   # spot pods can be reclaimed mid-run — name it clearly
            try:
                st = (get_pod(pid).get("desiredStatus") or "").upper()
            except Exception:
                st = "UNREACHABLE"
            if st in ("EXITED", "TERMINATED", "UNREACHABLE", ""):
                log.warning("task rc=%d and pod is %s — likely SPOT PREEMPTION. "
                            "Re-run to resume from the content-hash cache (only "
                            "un-synthesized segments recompute).", rc, st or "gone")
        # 4. results back — only work/ (dubbed audio, subs, manifest, QC). No
        # output/ from the pod; the mux happens locally next. Always attempt it,
        # even on failure, so partial progress is cached for a resumed run.
        rsync(rp, port, f"{rp['ssh_user']}@{host}:{remote}/work/",
              "./work/", check=False)
        # 5. mux LOCALLY: the source video is here, muxing is a cheap stream-copy.
        # Both run and autopilot skip the mux on the pod (--to s7 / --no-mux).
        if task in ("run", "autopilot") and rc == 0:
            from . import s8_mux
            try:
                s8_mux.run(cfg, video, langs)
                log.info("local mux -> output/%s_multi.mp4",
                         Path(video).stem)
            except Exception as e:
                log.warning("local mux failed (%s); audio is in work/", e)
        log.info("remote task %s exited rc=%d; results synced back", task, rc)
        return rc == 0
    finally:
        _terminate_tracked()   # state-file driven: fires even if provision crashed


# ---------------- entry points / diagnostics ----------------

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
    if st and st.get("created_at"):
        age = (time.time() - st["created_at"]) / 60
        print(f"state file: pod {st.get('pod_id')} (tracked {age:.0f} min ago)"
              + ("  !! STALE — run `remote kill`" if age > 30 else ""))
    else:
        print("state file:", st or "(none)")
    try:
        pods = _req("GET", "/pods")
        print(f"pods on account: {len(pods)}")
        for p in (pods if isinstance(pods, list) else []):
            print(f"  {p.get('id')}  {p.get('name')}  "
                  f"{p.get('desiredStatus') or p.get('status')}")
    except Exception as e:
        print("could not list pods:", e)
