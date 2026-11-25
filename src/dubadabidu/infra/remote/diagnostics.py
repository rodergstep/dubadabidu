# SPDX-License-Identifier: AGPL-3.0-or-later
"""`remote smoke | setup-check | status`."""
from __future__ import annotations
import json
import logging
import time

from dubadabidu.infra.remote import bootstrap
from dubadabidu.infra.remote import orchestrator
from dubadabidu.infra.remote import provider
from dubadabidu.infra.remote import runpod
from dubadabidu.infra.remote import state
from dubadabidu.infra.remote import transport

log = logging.getLogger("dubadabidu.runpod")


# ---------------- entry points / diagnostics ----------------

def smoke_test(cfg: dict) -> bool:
    """Cheapest possible lifecycle validation (~$0.02): provision -> print the
    raw pod JSON (to learn field shapes) + nvidia-smi over SSH -> terminate.
    Run this ONCE before trusting the full remote_run path."""
    rp = {**runpod.DEFAULTS, **cfg.get("runpod", {})}
    state.sweep_orphans()
    deadline = time.time() + rp["provision_timeout_s"] + 120
    pid = None
    try:
        pid, host, port = orchestrator.provision(rp, deadline)
        print(f"[smoke] pod {pid} reachable at {host}:{port}")
        print("[smoke] raw pod object (field-shape reference):")
        print(json.dumps(provider.active().get(pid), indent=2)[:1500])
        rc = transport.ssh_exec(rp, host, port,
                      "nvidia-smi --query-gpu=name,memory.total "
                      "--format=csv,noheader", timeout=60)
        print(f"[smoke] nvidia-smi rc={rc}")
        return rc == 0
    finally:
        state._terminate_tracked()   # state-file driven: fires even if provision crashed
        print("[smoke] cleanup done (pod terminated, state cleared)")


_OVERLAY_HINT = ("The bakeoff/runpod sections live in config.gpu.yaml, and "
                 "setup_check reads them from the LOCAL config — unlike "
                 "run/bakeoff/autopilot, whose REMOTE_TASK templates inject the "
                 "overlay into the pod-side command. Pass it explicitly:\n"
                 "  dubadabidu remote setup-check --overlay config.gpu.yaml")


def _require_something_to_validate(engines: list, probe_engines: list) -> None:
    """Refuse BEFORE provisioning when the config gives setup-check nothing to
    check. Without this, `remote setup-check` with no --overlay falls back to
    DEFAULTS (engine_setup: {}) and would provision a pod, install no
    challengers, probe none, and still return True because the incumbent
    imports — a paid run that teaches nothing and reports no error. Silent
    success is the failure mode worth spending three checks on."""
    if not engines:
        raise SystemExit("no bakeoff.engines configured — nothing for "
                         "setup-check to validate.\n" + _OVERLAY_HINT)
    if not probe_engines:
        raise SystemExit(
            f"bakeoff.engines is {engines} — only the incumbent, which "
            f"REMOTE_SETUP installs on every run anyway. setup-check exists to "
            f"validate the git-clone CHALLENGER (qwen); add it to "
            f"bakeoff.engines first.")
    if not any(has_snippet for _, _, has_snippet in probe_engines):
        missing = [e for e, _, has_snippet in probe_engines if not has_snippet]
        raise SystemExit(
            f"no runpod.engine_setup snippet for any challenger ({missing}) — "
            f"_install_engines would install nothing, so every probe would FAIL "
            f"for a reason that is not repo drift.\n" + _OVERLAY_HINT)


def setup_check(cfg: dict, budget_usd: float | None = None) -> bool:
    """Dry-run the install path on ONE cheap pod, then report which engines
    actually import — WITHOUT running a comparison. Provisions, installs the
    same base deps + every engine_setup snippet a real run would (into the ONE
    shared .venv since 2026-08-02), probes torch/CUDA and each engine module,
    and terminates. ~$0.30 and ~15 min to surface unpinned-repo drift, a wrong
    checkpoint id, or a dependency conflict BEFORE a full billing run hits them.
    Returns True iff CUDA is available AND every snippet-backed engine imports.
    Terminates on every path (state-file driven)."""
    rp = {**runpod.DEFAULTS, **cfg.get("runpod", {})}
    budget = float(budget_usd if budget_usd is not None else rp["budget_usd"])
    # No overlay forwarding here (remote_run has it): setup-check builds no
    # pod-side dub command — it runs REMOTE_SETUP + an import probe, both driven
    # by the LOCAL merged cfg. It read an undefined `overlays` until 2026-08-02.
    engines = list(dict.fromkeys(cfg.get("bakeoff", {}).get("engines", [])))
    # engines are probed by importing their module in the venv — the exact
    # thing a real synth call needs. (engine, module, has_snippet)
    probe_engines = [(e, bootstrap.ENGINE_MODULE[e], bool(rp.get("engine_setup", {}).get(e)))
                     for e in engines if e in bootstrap.ENGINE_MODULE]
    _require_something_to_validate(engines, probe_engines)
    state.sweep_orphans()
    remote = ("/workspace/dubadabidu" if rp.get("network_volume_id")
              else rp["remote_dir"])
    deadline = orchestrator._deadline(rp, budget)
    pid = None
    try:
        pid, host, port = orchestrator.provision(rp, deadline)
        provider.active().arm_watchdog(rp, host, port, deadline - time.time())
        # rsync only: setup-check probes imports, it never invokes ffmpeg
        if transport.ssh_exec(rp, host, port, bootstrap.apt_setup(False), timeout=600) != 0:
            raise RuntimeError("apt rsync/ffmpeg install failed")
        # code + ref only — no work/ needed for an install check (skip the upload)
        transport.rsync(rp, port, "./", f"{rp['ssh_user']}@{host}:{remote}/",
              extra_excludes=("input", "output", "work"))
        if transport.ssh_exec(rp, host, port, bootstrap.REMOTE_SETUP.format(dir=remote),
                    timeout=1800) != 0:
            raise RuntimeError("remote setup failed: dependency install error OR "
                               "CUDA unavailable (check the log's CUDA: line)")
        orchestrator._install_engines(rp, host, port, remote, engines,
                         "will report FAIL in the probe below")
        # import probe: one line per engine so the scorecard is skimmable.
        # ONE VENV since 2026-08-02 — engines install into .venv, so this
        # imports them directly instead of subprocessing into venvs/<engine>.
        # The chatterbox branch went with chatterbox; the exit code now comes
        # from CUDA plus every snippet-backed engine importing.
        probe = (
            "import sys\n"
            "import torch\n"
            "print('[probe] venv: torch', torch.__version__, '| cuda', "
            "torch.cuda.is_available())\n"
            "ok = torch.cuda.is_available()\n"
            "if not ok:\n"
            "    print('[probe] FAIL torch has no CUDA — the silent failure "
            "that cost a week of 126 s/take runs')\n"
            f"for eng, mod, has_snip in {probe_engines!r}:\n"
            "    if not has_snip:\n"
            "        print('[probe] SKIP', eng, '(no engine_setup snippet)')\n"
            "        continue\n"
            "    try:\n"
            "        __import__(mod)\n"
            "        print('[probe] OK  ', eng)\n"
            "    except Exception as e:\n"
            "        ok = False\n"
            "        print('[probe] FAIL', eng, '->', repr(e)[:140])\n"
            "sys.exit(0 if ok else 1)")
        cmd = (f"cd {remote}; . .venv/bin/activate; "
               f"python3 - <<'PYEOF'\n{probe}\nPYEOF")
        rc = transport.ssh_exec(rp, host, port, cmd, timeout=600)
        print(f"[setup-check] probe rc={rc} "
              f"({'all engines import + CUDA OK' if rc == 0 else 'BROKEN'})")
        print("[setup-check] read the [probe] lines above: an engine marked FAIL "
              "would be skipped (unavailable) in a real bake-off.")
        return rc == 0
    finally:
        state._terminate_tracked()
        print("[setup-check] cleanup done (pod terminated, state cleared)")


def status() -> None:
    """Show the state file + all pods currently on the account (leak check)."""
    st = json.loads(state.STATE_FILE.read_text()) if state.STATE_FILE.exists() else None
    if st and st.get("created_at"):
        age = (time.time() - st["created_at"]) / 60
        print(f"state file: pod {st.get('pod_id')} (tracked {age:.0f} min ago)"
              + ("  !! STALE — run `remote kill`" if age > 30 else ""))
    else:
        print("state file:", st or "(none)")
    try:
        pods = provider.active().list_all()
        print(f"pods on account: {len(pods)}")
        for p in pods:
            print(f"  {p.get('id')}  {p.get('name')}  {provider.active().status(p)}  "
                  f"${provider.active().price(p)}/h")
    except Exception as e:
        print("could not list pods:", e)
