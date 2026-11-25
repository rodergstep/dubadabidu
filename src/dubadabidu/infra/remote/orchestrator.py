# SPDX-License-Identifier: AGPL-3.0-or-later
"""M2 — RunPod lifecycle automation (docs/AUTOPILOT.md infra agent).

Provision a spot GPU pod -> rsync the project up -> run a task remotely ->
rsync results back -> ALWAYS terminate. The pod is cattle; all state returns to
this repo. Uses the REST v2 API (api.runpod.io/v2, Bearer auth).

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
import shlex
import time
from pathlib import Path

from dubadabidu.core import manifest as M
from dubadabidu.infra.remote import bootstrap
from dubadabidu.infra.remote import provider
from dubadabidu.infra.remote import runpod
from dubadabidu.infra.remote import state
from dubadabidu.infra.remote import transport

log = logging.getLogger("dubadabidu.runpod")


# Filled by provision() with the pod's REAL price and shape, read by remote_run
# when it writes the run ledger. A module global rather than a return value so
# the ledger could be added without changing provision()'s signature — the
# safety-critical path stays untouched.
_LAST_POD: dict = {}


# ---------------- provisioning + orchestration ----------------

def _deadline(rp: dict, budget_usd: float) -> float:
    hours = min(budget_usd / max(rp["assumed_price_per_hr"], 0.01),
                rp["max_runtime_hours"])
    return time.time() + hours * 3600


def _provision_once(rp: dict, deadline: float) -> tuple[str, str, int, str]:
    """One create + wait. Returns (pod_id, host, port, gpu_id). Terminates the
    pod (via the state file) on ANY failure, so nothing can leak."""
    pod = provider.active().create(rp)
    pid = pod.get("id") or pod.get("podId")
    if not pid:
        raise RuntimeError(f"no pod id in create response: {str(pod)[:200]}")
    gpu_id = str((pod.get("gpu") or {}).get("id") or "")
    state._save_state({"pod_id": pid, "created_at": time.time()})
    log.info("pod %s created (%s); waiting for RUNNING + SSH ...", pid, gpu_id)
    try:
        end = min(deadline, time.time() + rp["provision_timeout_s"])
        direct_wait = float(rp.get("direct_ssh_wait_s", 150))
        running_since: float | None = None
        target = None
        while time.time() < end:
            p = provider.active().get(pid)
            status = provider.active().status(p)
            target = provider.active().ssh_target(p)
            if target:
                break
            if status in ("EXITED", "TERMINATED", "FAILED", "ERROR"):
                raise RuntimeError(f"pod entered {status} before SSH came up")
            # REST v2 has no way to REQUIRE a public IP (v1's supportPublicIp
            # was removed). Some hosts never assign one: the pod sits RUNNING
            # with only the ssh.runpod.io proxy, which cannot carry rsync.
            # Measured 2026-11-19: 7 min RUNNING, no mapping. Give up early.
            # BUT v2 says RUNNING from the first second, while the image is
            # still pulling: `runtime` stays null until the container starts,
            # and ssh.direct arrives WITH it (measured 2026-11-24: 182 s on an
            # L4). Clocking from status alone condemned 5 healthy pods in one
            # afternoon as "proxy-only". Start the clock at container start.
            if status == "RUNNING" and provider.active().container_started(p):
                running_since = running_since or time.time()
                if time.time() - running_since > direct_wait:
                    raise provider.NoDirectSsh(
                        f"pod {pid} ({gpu_id}) RUNNING {direct_wait:.0f}s with "
                        f"no public SSH mapping (proxy-only host)")
            time.sleep(10)
        if not target:
            raise RuntimeError("pod did not expose SSH within provision_timeout_s")
        host, port = target
        if not transport.wait_ssh(rp, host, port, min(deadline, time.time() + 240)):
            raise RuntimeError("SSH never became reachable")
    except BaseException:
        state._terminate_tracked()   # crash-safe: reads pid from state, not a local var
        raise
    return pid, host, port, gpu_id


def provision(rp: dict, deadline: float) -> tuple[str, str, int]:
    """Create a pod and wait for SSH. Returns (pod_id, host, port).

    A proxy-only host is retried on the NEXT gpu type (a different host pool),
    up to runpod.provision_attempts; every failed attempt's pod is terminated
    before the next is created."""
    attempts = max(1, int(rp.get("provision_attempts", 3)))
    ids = rp["gpu_type_ids"]
    ids = [ids] if isinstance(ids, str) else list(ids)
    for attempt in range(1, attempts + 1):
        try:
            pid, host, port, used = _provision_once({**rp, "gpu_type_ids": ids},
                                                    deadline)
            break
        except provider.NoDirectSsh as e:
            if attempt == attempts or time.time() >= deadline:
                raise
            # rotate past the type that landed on a proxy-only host
            if used_type := next((i for i in ids if f"({i})" in str(e)), None):
                ids = ids[ids.index(used_type) + 1:] + ids[:ids.index(used_type) + 1]
            else:
                ids = ids[1:] + ids[:1]
            log.warning("%s — retrying (%d/%d) starting from %s",
                        e, attempt + 1, attempts, ids[0])
    # WHAT ARE WE PAYING? Nothing recorded this before, so the only way to know
    # was the RunPod dashboard — which is how every run silently landing on a
    # 4090 (7% GPU / 7% CPU / 17% RAM at top rate) went unnoticed for a week
    # while gpuTypePriority was hardcoded to "availability" and the
    # cheapest-first ordering did nothing.
    # Flagged on PRICE, not on the GPU name: REST v1's GET /pods/{id} returns an
    # EMPTY `machine` object and no gpuTypeId anywhere (checked 2026-08-01), so
    # the card's identity is simply not retrievable. Price is the better signal
    # regardless — it is what the budget is spent in, and vcpu/RAM size with the
    # GPU tier, so an oversized instance shows up here too.
    try:
        info = provider.active().get(pid)
        price = provider.active().price(info)
        budgeted = float(rp.get("assumed_price_per_hr", 0) or 0)
        over = (price is not None and budgeted and float(price) > budgeted)
        log.info("pod %s SSH ready at %s:%d — $%s/h, %s vCPU, %s GB RAM%s",
                 pid, host, port, price, provider.active().vcpu(info),
                 provider.active().ram_gb(info),
                 f"  [OVER the budgeted ${budgeted}/h]" if over else "")
        # Stash the ACTUAL price/shape for the run ledger. This is the only
        # place the real rate is known — the API reports costPerHr per pod, and
        # rp["assumed_price_per_hr"] is deliberately the worst case (0.7), so
        # costing a run from config would overstate it ~2.5x. See runlog.py.
        _LAST_POD.update({"pod_id": pid, "price_per_hr": price,
                          "vcpu": provider.active().vcpu(info),
                          "ram_gb": provider.active().ram_gb(info)})
    except Exception as e:      # never fail a run over a log line
        log.info("pod %s SSH ready at %s:%d (price unknown: %s)",
                 pid, host, port, e)
    return pid, host, port


def _install_engines(rp: dict, host: str, port: int, remote: str,
                     needed: list[str], fail_note: str) -> None:
    """Run each engine's engine_setup snippet on the pod, in the main venv.
    edge has no snippet and is skipped. A failure is non-fatal — the caller
    decides what an unavailable engine means (bake-off skips it; a run's s4
    raises)."""
    for eng in needed:
        snip = rp.get("engine_setup", {}).get(eng)
        if not snip:
            continue
        log.info("installing engine on pod: %s (main venv) [%s]",
                 eng, eng)
        if transport.ssh_exec(rp, host, port, bootstrap.engine_install_cmd(remote, eng, snip),
                    timeout=2400) != 0:
            log.warning("engine %s setup failed — %s", eng, fail_note)


def remote_run(cfg: dict, video: str | list[str], langs: list[str], task: str,
               budget_usd: float | None = None,
               keep_alive: bool = False, reuse: bool = False,
               overlays: list[str] | None = None,
               stages: str | None = None) -> bool:
    """Full lifecycle: sweep -> provision -> sync up -> run -> sync back ->
    ALWAYS terminate. Returns True on remote task success.

    `overlays` are the --overlay paths the CALLER was invoked with. They must be
    forwarded into the pod-side command: REMOTE_TASK hardcodes
    `--overlay config.gpu.yaml`, so before this existed any extra overlay shaped
    the LOCAL config (provisioning decisions) and was silently dropped on the
    pod. An experiment overlay therefore ran as a plain re-run of the base
    config — measured 2026-08-01, ~10 min of pod time spent re-sweeping
    references while believing an ICL test was running."""
    rp = {**runpod.DEFAULTS, **cfg.get("runpod", {})}
    budget = float(budget_usd if budget_usd is not None else rp["budget_usd"])
    # config.gpu.yaml is already in REMOTE_TASK; forward only the extras, in
    # the caller's order, so "later overlay wins" holds on the pod as locally.
    extra_overlays = [o for o in (overlays or [])
                      if Path(o).name != "config.gpu.yaml"]
    # NOTE: the orphan sweep must run AFTER the reuse decision. sweep_orphans()
    # terminates whatever the state file tracks, which is exactly the pod a
    # previous --keep-alive run deliberately left up — so sweeping first made
    # --reuse impossible: it killed the pod ~30 lines before _live_pod() looked
    # for it (measured 2026-07-30, destroying a validated indextts install).
    # Without --reuse the behaviour is unchanged: sweep before anything starts.
    held = None if reuse else state._held_pod()
    if held:
        # a failed run left this pod up for exactly this: retry on it.
        # _live_pod() still proves it is RUNNING + reachable, else a fresh
        # pod is provisioned (and the held one is swept below).
        log.info("pod %s is held from a failed run — reusing it (hotfix)", held)
        reuse = True
    if not reuse:
        state.sweep_orphans()
    if task not in bootstrap.REMOTE_TASK:
        raise SystemExit(f"unknown remote task {task!r} (choose "
                         f"{list(bootstrap.REMOTE_TASK)})")
    # `run` takes several videos and does them in ONE pod-side process, so
    # the models load once (~30 s each time: qwen 7 s, the clone prompt 8.5 s,
    # ECAPA, MOS, whisper — measured 2026-11-24) instead of once per video.
    videos = [video] if isinstance(video, str) else list(video)
    if not videos:
        raise SystemExit("remote_run: no video given")
    if len(videos) > 1 and task != "run":
        raise SystemExit(f"remote {task} takes one video, got {len(videos)}")
    video = videos[0]
    # the pod gets work/ + ref/ + code, NOT the source video — so s1/s2 must be
    # done locally first (their cached stems live in work/). Fail fast with a
    # clear message instead of wasting a pod.
    for v in videos:
        if not M.manifest_path(cfg, v).exists():
            raise SystemExit(
                f"{v}: no cached manifest — run s1+s2 locally first "
                f"(`dubadabidu preamble {v}`). The GPU pod never receives the "
                f"source video; it works from work/<video>/ audio stems.")
        if not (M.video_workdir(cfg, v) / "vocals.wav").exists():
            raise SystemExit(f"{v}: work/{Path(v).stem}/vocals.wav missing — "
                             f"run s1 locally first.")
    # FAIL BEFORE PROVISIONING IF THE POD COULD NOT POSSIBLY DO THE WORK.
    # runpod.engine_setup lives in config.gpu.yaml, and the overlay stack is
    # asymmetric: REMOTE_TASK hardcodes `--overlay config.gpu.yaml` for the POD
    # command, so the pod always gets it — but the LOCAL cfg only has it if the
    # caller passed it too, and that half is what supplies engine_setup for the
    # installer. Omit it locally and everything looks fine: the pod provisions,
    # bootstraps, installs NOTHING, and the bake-off dutifully reports "engine
    # unavailable" over zero measurements. That exact run is already recorded in
    # this file's history ("two ICL arms provisioned their own pod and both
    # produced empty scorecards"), and it happened again on 2026-08-13 — ~10
    # minutes of billing for a result that could not exist.
    #
    # The check is cheap and it is BEFORE the money starts: a GPU engine with no
    # install snippet cannot be installed, so refuse rather than discover it
    # after the bootstrap.
    _needed_pre = bootstrap.engines_for_task(cfg, task, langs)
    _no_snippet = [e for e in _needed_pre
                   if e != "edge" and not (rp.get("engine_setup") or {}).get(e)]
    if _no_snippet:
        msg = (f"runpod.engine_setup has no install command for "
               f"{_no_snippet} — the pod cannot install {'them' if len(_no_snippet) > 1 else 'it'}, "
               f"so this run would provision, bootstrap and measure nothing.\n"
               f"Almost always the cause is a missing LOCAL overlay: pass\n"
               f"  --overlay config.gpu.yaml\n"
               f"in addition to any experiment overlay. (The pod-side command "
               f"already has it; the local side is what feeds the installer.)")
        if reuse:
            log.warning("%s\n  --reuse: continuing, since a live pod may "
                        "already have it installed.", msg)
        else:
            raise SystemExit(msg)

    deadline = _deadline(rp, budget)
    # with a persistent volume, put the project (incl .venv) on it so deps survive
    remote = ("/workspace/dubadabidu" if rp.get("network_volume_id")
              else rp["remote_dir"])
    hours = (deadline - time.time()) / 3600
    log.info("budget $%.2f -> auto-terminate in %.1f h", budget, hours)

    # WHICH ENGINES this pod must have — resolved ONCE, before the reuse
    # decision, and shared by the probe and the installer (see engines_for_task).
    needed = bootstrap.engines_for_task(cfg, task, langs)

    _t0 = time.time()
    _LAST_POD.clear()          # this run's pod, not the previous one's
    pid = None
    _boot_s = None   # None distinguishes "died before the task" from "0 s"
    rc = None   # bound before the finally can read it: a failure during
                # provision/bootstrap never reaches the task, and an unbound
                # local would raise UnboundLocalError inside the cleanup
    try:
        alive = state._live_pod() if reuse else None
        if reuse and not alive:
            # nothing reusable: do not leave a tracked-but-dead pod behind
            state.sweep_orphans()
        if alive:
            # REUSE a pod left up by a previous --keep-alive run: skip provision,
            # watchdog-arm and apt, all of which are already done. The saving is
            # the BOOTSTRAP, and it is far bigger than it first looks: the torch
            # download is 4-6 GB and its wall-clock depends entirely on the pod's
            # route to PyPI. Measured 2026-07-30: ~4 min on a good draw, but
            # ~1.5 MB/s (40-60 min) on a bad one, and one run died on a bare
            # ReadTimeoutError. Reuse converts that lottery into a one-off.
            pid, host, port = alive
            log.info("reusing pod %s at %s:%d — skipping provision/apt",
                     pid, host, port)
            # The ledger's price comes from provision(), which a reused run
            # skips — so every --reuse run recorded price_per_hr 0.00 and cost
            # $0.000. That is silent UNDER-reporting of exactly the runs
            # --keep-alive exists to encourage: two real A/B arms on 2026-08-09
            # billed ~20 min of pod time and landed in runs.jsonl as free.
            # cost_per_video_hour then averages the zeros in. Re-stamp it here.
            try:
                info = provider.active().get(pid)
                _LAST_POD.update({"pod_id": pid,
                                  "price_per_hr": provider.active().price(info),
                                  "vcpu": provider.active().vcpu(info),
                                  "ram_gb": provider.active().ram_gb(info),
                                  "reused": True})
            except Exception as e:   # never fail a run over a ledger field
                log.info("reused pod price unknown: %s", e)
            # RE-ARM against THIS run's deadline. The pod is still counting
            # down from the deadline of the run that created it; a course of
            # twenty videos on one pod would otherwise be killed partway
            # through by video 1's timer while this process believed it had
            # hours left. arm_pod_watchdog kills the old sleeper first.
            provider.active().arm_watchdog(rp, host, port, deadline - time.time())
        else:
            if reuse:
                log.info("--reuse: no live pod to attach to; sweeping then "
                         "provisioning")
                state.sweep_orphans()   # deferred from the top; a stale state-file
                                  # pod must still never be left billing
            pid, host, port = provision(rp, deadline)
            # RE-DERIVE THE DEADLINE FROM WHAT THIS POD ACTUALLY COSTS.
            # _deadline() divides the budget by runpod.assumed_price_per_hr
            # (0.25), so --budget only ever meant dollars if the guess was
            # right. The ledger says it often was not: prices actually paid are
            # 0.25, 0.26, 0.28 and 0.49. On the 0.49 pod a "$10 cap" bought
            # nearly twice the dollars it promised, because the cap is on TIME.
            # The real rate is known the moment the pod is up, so use it — and
            # only ever to SHORTEN, never to extend past what was asked for.
            actual = float(_LAST_POD.get("price_per_hr") or 0)
            if actual > 0:
                hrs = min(budget / actual, rp["max_runtime_hours"])
                tighter = time.time() + hrs * 3600
                if tighter < deadline:
                    log.warning(
                        "pod costs $%.2f/h, not the assumed $%.2f — deadline "
                        "cut from %.1f h to %.1f h so $%.2f stays $%.2f",
                        actual, rp["assumed_price_per_hr"],
                        (deadline - time.time()) / 3600, hrs, budget, budget)
                    deadline = tighter
            # arm the independent pod-side self-destruct FIRST — before the long
            # install — so a crash during setup can't leave a billing pod
            provider.active().arm_watchdog(rp, host, port, deadline - time.time())
        # 0. base image lacks rsync/ffmpeg — install BEFORE the project sync
        # (rsync-over-ssh needs rsync on the pod). A bake-off never invokes the
        # ffmpeg binary, so it skips ffmpeg and dodges the runtime-upgrade that
        # kills sshd; run/autopilot need s5/s6 and still take it. See apt_setup.
        # ffmpeg is needed only by stages that actually shell out to it — s5's
        # atempo/rubberband, s6's loudnorm, _synth_edge. If the pod is running a
        # NARROWED range that excludes them (e.g. s4 only, with fit/mix done
        # locally), skip it: the apt tree behind ffmpeg upgrades the C runtime
        # and has killed sshd mid-run before. Do not "work around" that by
        # retrying — just do not install what is not used.
        needs_ffmpeg = task != "bakeoff" and not (
            stages and "s4_synthesize" in stages and "s5" not in stages
            and "s6" not in stages and "s7" not in stages)
        if not alive and transport.ssh_exec(rp, host, port, bootstrap.apt_setup(needs_ffmpeg),
                                  timeout=600) != 0:
            raise RuntimeError("could not install rsync/ffmpeg on the pod after "
                               "retries (see log for the apt error)")
        # 1. project up — EXCLUDE input/ (the 4K source video is never needed on
        # the pod) and output/. The pod works from the synced work/ audio stems.
        #
        # A BAKE-OFF reads only three things out of work/<video>: manifest.json
        # (the translations), vocals.wav (the real-voice anchor) and qc_ua/ (the
        # side-by-side slices). Measured on sketch60 that is 3.5 MB against 238 MB
        # for the whole tree — the rest is old seg/, seg_old/, demucs/, tune/ and
        # previous dubs, none of which a bake-off opens. The full sync took ~8 min
        # of BILLED pod time, and testing engines one at a time multiplies that by
        # the number of runs, so send the code without work/ and then just the
        # three paths (-R keeps their layout).
        up = f"{rp['ssh_user']}@{host}:{remote}/"
        if task == "bakeoff":
            transport.rsync(rp, port, "./", up, extra_excludes=("input", "output", "work"))
            wd = M.video_workdir(cfg, video)
            # results_*.json carries the engines measured by EARLIER runs.
            # Without it the pod merges into an empty dict and the sync-back
            # overwrites the local scorecard — which silently discarded the
            # voxcpm+qwen comparison twice before this was spotted.
            # NOT `needed` — that name holds the ENGINE list computed at the top
            # of this function and used further down by _verify_ready and
            # _install_engines. Reusing it here clobbered the engines with file
            # paths, so `_install_engines` looked up engine_setup["…/vocals.wav"],
            # found nothing, and installed NOTHING — silently, because a missing
            # snippet is a legitimate skip (edge has none).
            # Every `remote bakeoff` on a FRESH pod therefore came up with
            # "qwen unavailable" and measured zero engines. It went unnoticed
            # because bake-offs are normally run with --reuse against a pod some
            # earlier `run` had already installed qwen on; on 2026-08-09 two ICL
            # arms provisioned their own pod and both produced empty scorecards.
            bake_paths = [str(wd / n) for n in ("manifest.json", "vocals.wav",
                                                "qc_ua", "bakeoff")
                          if (wd / n).exists()]
            transport._rsync_paths(rp, port, bake_paths, up)
        else:
            # code + THESE videos' work dirs only. Sending all of work/ uploaded
            # every other lesson too (2.3 GB for a 339 MB video) and was most
            # of a 22.6 min bootstrap on 2026-11-24.
            transport.rsync(rp, port, "./", up, extra_excludes=("input", "output", "work"))
            transport._rsync_paths(rp, port, [str(M.video_workdir(cfg, v))
                                    for v in videos], up)
        # 2. install + verify CUDA (fails fast if CUDA is unavailable).
        # SKIPPED on a reused pod that PROVES it is already set up. The
        # experiment queue runs 6 bake-offs back to back against one pod, and
        # re-running setup + engine install per experiment cost ~50 s each -
        # roughly a third of the queue's wall clock spent re-confirming what the
        # first run established. The probe is the proof, not an assumption: it
        # imports the engine module and asserts CUDA inside the engine's OWN
        # venv, which is exactly what setup exists to guarantee. Anything less
        # certain re-runs setup.
        setup_ok = False
        if alive:
            setup_ok = transport._verify_ready(rp, host, port, remote, needed)
            if setup_ok:
                log.info("reused pod already provisioned (venv + %s import + "
                         "CUDA all verified) — skipping setup and engine install",
                         "/".join(needed) or "engines")
        if not setup_ok and transport.ssh_exec(
                rp, host, port, bootstrap.REMOTE_SETUP.format(dir=remote),
                timeout=1800) != 0:
            raise RuntimeError("remote setup failed: dependency install error "
                               "OR CUDA unavailable (check the log's CUDA: line)")
        # 2.5 the git-clone engine (qwen) must be installed on the pod before
        # the task runs (`needed`, computed once above and shared with the reuse
        # probe). edge needs no snippet and is skipped. A failure is non-fatal:
        # the bake-off marks the engine unavailable, and for a run s4 raises an
        # actionable error for the langs that needed it.
        if not setup_ok:
            _install_engines(rp, host, port, remote, needed,
                             "bake-off will mark it unavailable"
                             if task == "bakeoff"
                             else "s4 will fail for langs routed to it")
        # 3. run the task, self-capped by remote `timeout` to the deadline.
        # Pass the (low-privilege) translation key inline — an SSH session may
        # not inherit the pod's container env, and s3 needs it. Not logged.
        secs = max(60, int(deadline - time.time()))
        # shlex.quote: the video path reaches a REMOTE SHELL, so a filename
        # with spaces ("Organising the colour palette and materials.mp4" —
        # exactly how a real lesson is named) would be split into several
        # arguments and the run would fail after provisioning. _rsync_paths is
        # already safe: it passes a list to subprocess, never a shell.
        cmd = bootstrap.REMOTE_TASK[task].format(
            video=" ".join(shlex.quote(v) for v in videos),
            langs=",".join(langs),
            # s5, not s7 — see dub._remote_stages. Only s4 needs the GPU; s6/s7
            # are CPU work on already-synced files, and s6 needs the [master]
            # extra that pods do not install (it failed a full run 2026-08-09).
            stages=stages or "--to s5_fit")
        # extras AFTER the hardcoded config.gpu.yaml so they win, matching the
        # local merge order. The files themselves ship with the project rsync.
        for ov in extra_overlays:
            cmd += f" --overlay {ov}"
        if extra_overlays:
            log.info("forwarding overlays to the pod: %s",
                     " ".join(extra_overlays))
        # (shlex is imported at module scope — a local `import shlex` HERE made
        # the name function-local for the whole of remote_run, so the earlier
        # shlex.quote(video) raised UnboundLocalError before this line ever ran.)
        tk = shlex.quote(os.environ.get("TRANSLATE_API_KEY", ""))
        full = (f"cd {remote}; . .venv/bin/activate; "
                f"export TRANSLATE_API_KEY={tk}; "
                f"timeout {secs} {cmd}")
        # Everything up to here is BOOTSTRAP: provision, apt, project rsync,
        # the ~2.5 GB torch install and the engine setup. runlog.py has always
        # subtracted `bootstrap_s` to separate fixed cost from real work — and
        # nothing ever wrote it, so 15 of 17 ledger rows carry 0 and
        # `usd_per_video_hour_per_language` has silently been billing setup as
        # synthesis. Measure it, so "is batching worth it" is answerable from
        # the ledger instead of from memory.
        _boot_s = round(time.time() - _t0, 1)
        log.info("bootstrap took %.1f min; task starts now", _boot_s / 60)
        rc = transport.ssh_exec(rp, host, port, full, timeout=secs + 120)
        if rc != 0:   # spot pods can be reclaimed mid-run — name it clearly
            try:
                st = provider.active().status(provider.active().get(pid))
            except Exception:
                st = "UNREACHABLE"
            if st in ("EXITED", "TERMINATED", "UNREACHABLE", ""):
                log.warning("task rc=%d and pod is %s — likely SPOT PREEMPTION. "
                            "Re-run to resume from the content-hash cache (only "
                            "un-synthesized segments recompute).", rc, st or "gone")
        # 4. results back — only work/ (dubbed audio, subs, manifest, QC). No
        # output/ from the pod; the mux happens locally next. Always attempt it,
        # even on failure, so partial progress is cached for a resumed run.
        transport.rsync(rp, port, f"{rp['ssh_user']}@{host}:{remote}/work/",
              "./work/", check=False)
        # 5. mux LOCALLY: the source video is here, muxing is a cheap stream-copy.
        # Both run and autopilot skip the mux on the pod (--to s7 / --no-mux).
        # `run` does s6..s8 locally in dub.py (_remote_plan): muxing here
        # would pack whatever stale dub_<lang>.m4a an older mix left behind.
        if task == "autopilot" and rc == 0:
            from dubadabidu.stages import s8_mux
            try:
                s8_mux.run(cfg, video, langs)
                log.info("local mux -> output/%s_multi.mp4",
                         Path(video).stem)
            except Exception as e:
                log.warning("local mux failed (%s); audio is in work/", e)
        log.info("remote task %s exited rc=%d; results synced back", task, rc)
        return rc == 0
    finally:
        # RECORD WHAT THIS COST, before anything can throw. Written on every
        # path including failure — a run that died after 30 min of bootstrap is
        # exactly the data point that never survives in a log file. See
        # runlog.py for why this is not hand-maintained.
        try:
            from dubadabidu.core import runlog
            vid_min = segs = None
            for v in videos:
                man_p = M.manifest_path(cfg, v)
                if man_p.exists():
                    _m = json.loads(man_p.read_text(encoding="utf-8"))
                    vid_min = (vid_min or 0) + float(_m.get("duration") or 0) / 60
                    segs = (segs or 0) + len(_m.get("utterances") or [])
            vid_min = round(vid_min, 3) if vid_min else None
            runlog.append({
                "video": ",".join(Path(v).stem for v in videos), "task": task, "langs": list(langs),
                "stages": stages or None, "reuse": bool(reuse),
                "ok": rc == 0, "rc": rc,
                "wall_s": round(time.time() - _t0, 1),
                "bootstrap_s": _boot_s,
                "video_minutes": vid_min, "utterances": segs,
                "engines": needed,
                **{k: v for k, v in _LAST_POD.items()},
            })
        except Exception as e:                               # noqa: BLE001
            log.debug("run ledger skipped (%s)", e)
        # keep_alive: leave the pod UP so an install recipe can be debugged IN
        # PLACE. Without it every attempt paid ~4 min of bootstrap (apt, rsync,
        # the torch install the qc metrics need) just to learn one error message,
        # then discarded the environment — so a build bug needing three tries cost
        # three full bootstraps and three pods. Iterating over SSH costs one.
        #
        # This is the ONLY path that skips the client-side terminate, and it is
        # still bounded: the pod-side self-destruct watchdog is armed before any
        # install and fires at the deadline even if this process dies, and the
        # state file is deliberately LEFT IN PLACE so `remote kill` (and the
        # sweep at the start of the next remote command) will collect it.
        hold_min = float(rp.get("hold_on_failure_min") or 0)
        st_hold = None
        if not keep_alive and pid and rc not in (None, 0) and hold_min > 0:
            try:
                st_hold = provider.active().ssh_target(provider.active().get(pid))
            except RuntimeError:
                st_hold = None
        if st_hold:
            # the task ran and failed on a working pod: HOLD it for a hotfix.
            # Bounded by the re-armed pod-side watchdog, so walking away costs
            # at most hold_on_failure_min of pod time.
            secs = min(hold_min * 60, max(deadline - time.time(), 60))
            provider.active().arm_watchdog(rp, st_hold[0], st_hold[1], secs)
            state._save_state({"pod_id": pid, "created_at": time.time(),
                         "held_for_hotfix": True})
            log.warning("task FAILED (rc=%s) — pod %s HELD for a hotfix, "
                        "self-destructs in ~%.0f min (billing meanwhile).",
                        rc, pid, secs / 60)
            log.warning("  fix the code locally, then re-run the SAME command: "
                        "it reuses this pod (rsyncs only changed files, skips "
                        "setup).")
            log.warning("  give up instead:  dubadabidu remote kill")
        elif keep_alive and pid:
            st = provider.active().ssh_target(provider.active().get(pid)) if pid else None
            # free this run's engine venv so a --reuse run does not inherit
            # a disk already full of the previous challenger's torch
            # Only reclaim disk when the run SUCCEEDED. On failure the whole
            # point of --keep-alive is to inspect the broken venv, and wiping
            # it here made the two features cancel out (measured: the indextts
            # venv was gone before it could be debugged).
            if st and task == "bakeoff" and rc == 0:
                for eng in (cfg.get("bakeoff", {}) or {}).get("engines", []):
                    transport.free_engine(rp, st[0], st[1], remote, eng)
            log.warning("keep-alive: pod %s LEFT RUNNING (billing!)", pid)
            if st:
                log.warning("  ssh -i %s -p %d %s@%s   # cd %s",
                            rp["ssh_key"], st[1], rp["ssh_user"], st[0], remote)
            log.warning("  dubadabidu remote kill --overlay config.gpu.yaml"
                        "   # when done")
            log.warning("  watchdog will self-destruct it in ~%.0f min regardless",
                        (deadline - time.time()) / 60)
        else:
            state._terminate_tracked()  # state-file driven: fires even if provision crashed
