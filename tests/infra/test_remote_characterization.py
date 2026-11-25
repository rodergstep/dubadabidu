# SPDX-License-Identifier: AGPL-3.0-or-later
"""What `remote run` DOES, recorded before the RunPod code is split up.

Each scenario drives the real lifecycle against fakes at seams the split
cannot move: the RunPod REST API (urllib.request.urlopen), every ssh / rsync
invocation (subprocess.run), the clock (each infra module's `time`), and the
files it leaves (the state file, the run ledger). The recorded call log —
every REST call with its body, every argv, every state-file version, the
ledger row — is compared with golden_remote_calls.json. A refactor that
changes one byte of a pod-side command, the order of the lifecycle, or what is
written where, fails here.

  fresh_ok          provision -> watchdog -> apt -> sync -> setup -> install
                    -> task ok -> sync back -> ledger -> terminate
  task_fail_hold    the task fails on a working pod: held for a hotfix
  keep_alive_bake   --keep-alive bake-off: the pod stays up, disk reclaimed
  reuse_live        --reuse attaches to a live, verified pod: no setup
  reuse_none        --reuse with nothing to attach to: provisions
  capacity_retry    the first GPU type has no capacity (HTTP 400)
  no_direct_ssh     a proxy-only host: terminated, next GPU type tried
  missing_snippet   no engine_setup: refused before anything is created

Regenerate ONLY for an intended change:  DUB_REGEN_GOLDEN=1 pytest tests/infra
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.cfgs import settings
from tests.infra import remote_api as api

GOLDEN = Path(__file__).resolve().parent / "golden_remote_calls.json"
STATE = Path("work/.runpod_active.json")


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def time(self) -> float:
        self.now += 0.5
        return self.now

    def sleep(self, s: float) -> None:
        self.now += float(s)

    def perf_counter(self) -> float:
        return self.time()


@dataclass
class World:
    """The fake RunPod account and pod for one scenario."""
    task_rc: int = 0
    no_capacity: set[str] = field(default_factory=set)   # gpu ids answering 400
    proxy_only: set[str] = field(default_factory=set)    # gpu ids with no direct ssh
    verify_rc: int = 0
    pods: dict[str, dict[str, object]] = field(default_factory=dict)
    log: list[object] = field(default_factory=list)
    n: int = 0
    last_state: str | None = None

    def _state(self) -> None:
        cur = STATE.read_text(encoding="utf-8") if STATE.exists() else None
        if cur != self.last_state:
            self.log.append(["state", json.loads(cur) if cur else None])
            self.last_state = cur

    def pod_view(self, pid: str) -> dict[str, object]:
        p = self.pods[pid]
        view: dict[str, object] = {"id": pid, "status": p["status"],
                                   "costPerHr": 0.31, "vcpuCount": 8,
                                   "memoryInGb": 31, "gpu": {"id": p["gpu"]}}
        if p["status"] == "RUNNING":
            view["runtime"] = {"uptimeInSeconds": 30}
            if p["gpu"] not in self.proxy_only:
                view["ssh"] = {"direct": {"host": "203.0.113.9", "port": 22122}}
        return view

    # --- the REST seam ---------------------------------------------------------
    def urlopen(self, req: object, timeout: float = 60) -> io.BytesIO:
        method = req.get_method()                              # type: ignore[attr-defined]
        path = req.full_url.split("/v2", 1)[1]                 # type: ignore[attr-defined]
        body = json.loads(req.data) if req.data else None      # type: ignore[attr-defined]
        self.log.append(["rest", method, path, body])
        out: object = {}
        if method == "POST" and path == "/pods":
            gpu = body["gpu"]["id"]
            if gpu in self.no_capacity:
                raise urllib.error.HTTPError(req.full_url, 400, "no capacity", {},  # type: ignore[arg-type]
                                             io.BytesIO(b"no instances available"))
            self.n += 1
            pid = f"pod{self.n}"
            self.pods[pid] = {"status": "RUNNING", "gpu": gpu}
            out = {"id": pid, "gpu": {"id": gpu}}
        elif method == "GET" and path == "/pods":
            out = {"pods": [self.pod_view(p) for p in self.pods
                            if self.pods[p]["status"] == "RUNNING"],
                   "pagination": {"hasNextPage": False}}
        elif method == "GET" and path.startswith("/pods/"):
            pid = path.split("/")[2]
            if pid not in self.pods or self.pods[pid]["status"] == "TERMINATED":
                raise urllib.error.HTTPError(req.full_url, 404, "gone", {},   # type: ignore[arg-type]
                                             io.BytesIO(b"not found"))
            out = self.pod_view(pid)
        elif method == "DELETE":
            self.pods[path.split("/")[2]]["status"] = "TERMINATED"
        self._state()
        return io.BytesIO(json.dumps(out).encode())

    # --- the process seam ------------------------------------------------------
    def run(self, args: list[str], *a: object, **k: object) -> subprocess.CompletedProcess[str]:
        argv = [str(x) for x in args]
        self.log.append(["run", argv])
        rc, out = 0, ""
        cmd = argv[-1]
        if argv[0] == "ssh":
            if "timeout " in cmd and ".venv/bin/activate; export TRANSLATE_API_KEY" in cmd:
                rc = self.task_rc
            elif cmd.startswith("test -x "):
                rc = self.verify_rc
            elif cmd.startswith("df --output=pcent"):
                out = "88"
        self._state()
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")


def _project(tmp: Path) -> list[str]:
    """Two videos with the stems remote_run checks for, in a throwaway project."""
    vids = []
    for name in ("lesson a", "lesson_b"):
        wd = tmp / "work" / name
        wd.mkdir(parents=True)
        (wd / "manifest.json").write_text(json.dumps({
            "video": f"input/{name}.mp4", "duration": 90.0, "stages": {},
            "utterances": [{"id": "u0001", "start": 0, "end": 2, "text_src": "x",
                            "tr": {}}]}), encoding="utf-8")
        (wd / "vocals.wav").write_bytes(b"RIFF")
        vids.append(f"input/{name}.mp4")
    return vids


SCENARIOS: dict[str, tuple[dict[str, object], dict[str, object], Callable[[World], None]]] = {
    # name: (World kwargs, remote_run kwargs, setup)
    "fresh_ok": ({}, {"task": "run"}, lambda w: None),
    "task_fail_hold": ({"task_rc": 1}, {"task": "run"}, lambda w: None),
    "keep_alive_bake": ({}, {"task": "bakeoff", "keep_alive": True, "one": True},
                        lambda w: None),
    "reuse_live": ({}, {"task": "run", "reuse": True}, lambda w: (
        w.pods.__setitem__("pod7", {"status": "RUNNING", "gpu": "NVIDIA RTX A4000"}),
        STATE.parent.mkdir(parents=True, exist_ok=True),
        STATE.write_text(json.dumps({"pod_id": "pod7", "created_at": 1.0}), encoding="utf-8"))),
    "reuse_none": ({}, {"task": "run", "reuse": True}, lambda w: None),
    "capacity_retry": ({"no_capacity": {"NVIDIA RTX A4000"}}, {"task": "run"}, lambda w: None),
    "no_direct_ssh": ({"proxy_only": {"NVIDIA RTX A4000"}}, {"task": "run"}, lambda w: None),
    "missing_snippet": ({}, {"task": "run", "no_snippet": True}, lambda w: None),
}


def _play(name: str, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    world_kw, run_kw, setup = SCENARIOS[name]
    monkeypatch.chdir(tmp)
    monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
    monkeypatch.setenv("TRANSLATE_API_KEY", "tk-test")
    w = World(**world_kw)                                     # type: ignore[arg-type]
    clock = Clock()
    monkeypatch.setattr("urllib.request.urlopen", w.urlopen)
    monkeypatch.setattr(subprocess, "run", w.run)
    for mod in api.time_users():
        monkeypatch.setattr(mod, "time", clock)
    vids = _project(tmp)
    setup(w)
    w._state()
    patch: dict[str, object] = {"work_dir": "work", "runpod": {"ssh_key": "~/.ssh/k"}}
    if run_kw.get("no_snippet"):
        # an EMPTY snippet: a {} patch would deep-merge into the overlay's
        patch["runpod"] = {"ssh_key": "~/.ssh/k", "engine_setup": {"qwen": ""}}
    cfg = settings(patch, "config.gpu.yaml")
    task = str(run_kw["task"])
    videos: str | list[str] = vids[0] if run_kw.get("one") or task != "run" else vids
    try:
        ok: object = api.remote_run(cfg, videos, ["en", "ru"], task, 2.0,
                                    keep_alive=bool(run_kw.get("keep_alive")),
                                    reuse=bool(run_kw.get("reuse")),
                                    overlays=["config.gpu.yaml", "config.exp.icl.yaml"],
                                    stages="--from s4_synthesize --to s5_fit")
    except SystemExit as e:
        ok = f"SystemExit: {str(e)[:120]}"
    w._state()      # whatever the lifecycle wrote after its last call
    ledger = Path("ledger/runs.jsonl")
    rows = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    for r in rows:
        r.pop("ts", None)
    home = os.path.expanduser("~")
    text = json.dumps({"result": ok, "log": w.log, "ledger": rows})
    return json.loads(text.replace(home, "~").replace(str(tmp), "<TMP>"))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_remote_lifecycle_is_unchanged(name: str, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    got = _play(name, tmp_path, monkeypatch)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    if os.environ.get("DUB_REGEN_GOLDEN") == "1":
        golden[name] = got
        GOLDEN.write_text(json.dumps(golden, indent=1, ensure_ascii=False, sort_keys=True)
                          + "\n", encoding="utf-8")
        return
    want = golden[name]
    if got != want:
        for i, (g, x) in enumerate(zip(got["log"], want["log"], strict=False)):
            if g != x:
                pytest.fail(f"{name}: call {i} moved\n  was: {str(x)[:400]}\n  now: {str(g)[:400]}")
        assert got == want
