# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neutrality harness: run the local pipeline end to end on a fixture, offline.

    python -m tests.neutrality.harness --dump OUT.json [--project DIR] [--full]

Builds a throwaway project directory (copies of config.yaml, the overlays,
glossary/, prompts/ and specs/, a synthetic reference clip, and the committed
fixture video), installs the boundary fakes (tests/neutrality/fakes.py),
forbids network access, then drives the REAL CLI in-process through:

  A. source uk -> en, ru with --engine edge
     run --to s2_transcribe | run --from s3_translate --to s5_fit |
     run --from s6_mix | run --from s3_translate (resume: nothing new may be
     synthesized or translated) | qc | review | batch
  B. source ru -> uk (uk routed to edge by config, the real dub path)
     run (whole pipeline)

and writes a snapshot of everything the code produced: the exact bytes of every
manifest (temp path replaced by <PROJECT>), the sha256 of every file under
work/ and output/, every LLM request, the argv of every subprocess call, and
what the commands printed. tests/neutrality/test_neutrality.py compares the
snapshot with golden_<mode>.json; scripts/neutrality_diff.py compares two git
revisions with each other.

translation.lang_workers is pinned to 1 by the harness overlay: with one
thread per language, the order of keys in tr{} (and so the manifest's bytes)
depends on which thread saves first. That is a property of production, not
of this harness — see ARCHITECTURE_PLAN.md §10.2.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "neutrality" / "clip.mp4"
OVERLAY = """\
# neutrality harness overlay (generated; see tests/neutrality/harness.py)
asr:
  backend: faster
  device: cpu
translation:
  lang_workers: 1
"""
ECAPA = "work/.models/ecapa"          # speechbrain's savedir (qc.metrics)
CALLS: list[list[str]] = []
PRINTED: list[str] = []


def _copy_project(project: Path) -> None:
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    for f in sorted(REPO.glob("config*.yaml")):
        shutil.copy2(f, project / f.name)
    for d in ("glossary", "prompts", "specs"):
        if (REPO / d).is_dir():
            shutil.copytree(REPO / d, project / d)
    (project / "neutrality.yaml").write_text(OVERLAY, encoding="utf-8")
    (project / "input").mkdir()
    shutil.copy2(FIXTURE, project / "input" / "clip.mp4")
    shutil.copy2(FIXTURE, project / "input" / "clip_ru.mp4")
    _reference(project / "ref" / "sketch_ref_08.wav")


def _reference(path: Path) -> None:
    """A deterministic 4 s 'voice' so ECAPA calibration has a reference."""
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True)
    t = np.arange(int(4.0 * 24000)) / 24000
    f = 140 + 25 * np.sin(2 * np.pi * 2.0 * t)
    y = 0.4 * np.sin(2 * np.pi * np.cumsum(f) / 24000) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t))
    sf.write(path, y.astype("float32"), 24000)


def _no_network() -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise OSError("network access is forbidden in the neutrality harness")
    socket.create_connection = refuse          # type: ignore[assignment]
    socket.socket.connect = refuse             # type: ignore[assignment]
    socket.getaddrinfo = refuse                # type: ignore[assignment]


def _record_subprocess() -> None:
    real_run, real_out = subprocess.run, subprocess.check_output

    def argv(a: object) -> list[str]:
        return [str(x) for x in a] if isinstance(a, (list, tuple)) else [str(a)]

    def run(args: Sequence[str] | str, *a: object, **k: object) -> subprocess.CompletedProcess[str]:
        CALLS.append(argv(args))
        return real_run(args, *a, **k)            # type: ignore[call-overload]

    def check_output(args: Sequence[str] | str, *a: object, **k: object) -> bytes | str:
        CALLS.append(argv(args))
        return real_out(args, *a, **k)            # type: ignore[call-overload]

    subprocess.run = run                          # type: ignore[assignment]
    subprocess.check_output = check_output        # type: ignore[assignment]


def _cli(project: Path) -> Callable[[list[str]], None]:
    """The CLI entry of THIS tree, pointed at the throwaway project."""
    sys.path.insert(0, str(REPO))
    import dub
    dub.ROOT = project            # pre-refactor: dub.py anchors on ROOT

    def call(args: list[str]) -> None:
        sys.argv = ["dubadabidu", *args]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                dub.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise RuntimeError(f"`dubadabidu {' '.join(args)}` exited "
                                   f"{e.code!r}\n{buf.getvalue()}") from e
        PRINTED.append(f"$ dubadabidu {' '.join(args)}\n{buf.getvalue()}")
    return call


def _norm(s: str, project: Path) -> str:
    # longest spelling first: on macOS /var is a symlink to /private/var, and
    # replacing the short form first would leave "/private<PROJECT>" behind
    for p in sorted({str(project), str(project.resolve())}, key=len, reverse=True):
        s = s.replace(p, "<PROJECT>")
    return s


def _snapshot(project: Path, calls_at: dict[str, int]) -> dict[str, object]:
    from tests.neutrality import fakes
    files: dict[str, str] = {}
    texts: dict[str, str] = {}
    for root in ("work", "output", "ledger"):
        base = project / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(project).as_posix()
            if not p.is_file() or rel.startswith(ECAPA):
                continue
            data = p.read_bytes()
            if p.name == ".audio_sig.json":
                # memo keyed on "name:size:mtime_ns"; the mtime is the run's,
                # the values (content hashes) are what must not change
                memo = json.loads(data)
                data = json.dumps({k.rsplit(":", 1)[0]: v for k, v in memo.items()},
                                  sort_keys=True).encode("utf-8")
            if p.suffix in (".json", ".srt", ".html", ".md", ".txt"):
                text = _norm(data.decode("utf-8"), project)
                files[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if p.name == "manifest.json" or p.suffix == ".srt":
                    texts[rel] = text
            else:
                files[rel] = hashlib.sha256(data).hexdigest()
    return {
        "files": files,
        "texts": texts,
        "llm_requests": [{k: (_norm(v, project) if isinstance(v, str) else v)
                          for k, v in r.items()} for r in fakes.request_digest()],
        "subprocess": [[_norm(a, project) for a in c] for c in CALLS],
        "calls_at": calls_at,
        "printed": [_norm(p, project) for p in PRINTED],
    }


def fingerprint() -> dict[str, str]:
    """Toolchain whose bytes the audio hashes depend on."""
    import importlib.metadata as md
    out = {"ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                    text=True).stdout.splitlines()[0]}
    for pkg in ("pedalboard", "torch", "torchaudio", "librosa", "numpy",
                "soundfile", "pydub"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def run(project: Path, full: bool = False) -> dict[str, object]:
    from tests.neutrality import fakes
    _copy_project(project)
    if full:   # real ECAPA weights from this checkout's cache (offline)
        (project / "work" / ".models").mkdir(parents=True)
        (project / ECAPA).symlink_to(REPO / ECAPA, target_is_directory=True)
    fakes.install(full=full)
    _no_network()
    _record_subprocess()
    os.environ["TRANSLATE_API_KEY"] = "fake-key"
    os.environ.pop("RUNPOD_API_KEY", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cwd = Path.cwd()
    os.chdir(project)
    cli = _cli(project)
    marks: dict[str, int] = {}
    a = ["--overlay", "neutrality.yaml"]
    va, langs = "input/clip.mp4", ["--engine", "edge", "--langs", "en,ru"]
    try:
        cli(["run", va, *langs, *a, "--to", "s2_transcribe"])
        cli(["run", va, *langs, *a, "--from", "s3_translate", "--to", "s5_fit"])
        cli(["run", va, *langs, *a, "--from", "s6_mix"])
        marks["before_resume"] = len(CALLS)
        llm_before = len(fakes.LLM_LOG)
        cli(["run", va, *langs, *a, "--from", "s3_translate"])
        marks["resume_llm_calls"] = len(fakes.LLM_LOG) - llm_before
        marks["after_resume"] = len(CALLS)
        cli(["qc", va, *langs, *a])
        cli(["review", va, "--langs", "en,ru", *a])
        cli(["batch", *a])
        cli(["run", "input/clip_ru.mp4", "--langs", "uk", *a,
             "--overlay", "config.source-ru.yaml"])
    finally:
        os.chdir(cwd)
    return {"fingerprint": fingerprint(), "mode": "full" if full else "fast",
            **_snapshot(project, marks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--project", default=None,
                    help="throwaway project dir (default: a new temp dir)")
    ap.add_argument("--full", action="store_true",
                    help="real ECAPA / Distill-MOS from local caches")
    a = ap.parse_args()
    dump = Path(a.dump).resolve()
    project = (Path(a.project).resolve() if a.project else
               Path(tempfile.mkdtemp(prefix="dub-neutrality-")) / "project")
    snap = run(project, full=a.full)
    dump.write_text(json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True),
                    encoding="utf-8")


if __name__ == "__main__":
    main()
