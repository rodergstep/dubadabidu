# SPDX-License-Identifier: AGPL-3.0-or-later
"""`dubadabidu doctor`: validate the environment before anything runs."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from dubadabidu.core.config.schema import Config, Translation
from dubadabidu.core.device import torch_device, whisper_device


def doctor(cfg: Config, project: str = "") -> int:
    import os
    ok = True

    def check(name: str, cond: object, hint: str = "") -> None:
        nonlocal ok
        print(f"  [{'OK' if cond else '!!'}] {name}" + ("" if cond else f" — {hint}"))
        ok = ok and bool(cond)

    def warn(name: str, hint: str) -> None:
        """A degraded-but-runnable setup: shown, never fails doctor."""
        print(f"  [w ] {name} — {hint}")

    print("dubadabidu doctor")
    if project:
        print(f"  [i ] project: {project}")
    check("ffmpeg", shutil.which("ffmpeg"), "install ffmpeg and add to PATH")
    check("ffprobe", shutil.which("ffprobe"), "comes with ffmpeg")
    dev = torch_device()
    from dubadabidu.engines.asr import resolve_backend
    backend = resolve_backend(cfg["asr"])
    if backend == "mlx":
        print(f"  [i ] torch device: {dev} | whisper: mlx-whisper (Metal/ANE)")
    else:
        wdev, wct = whisper_device(cfg["asr"].get("device", "auto"))
        print(f"  [i ] torch device: {dev} | whisper: faster-whisper {wdev}/{wct}")
        if dev == "mps":
            print("  [i ] ASR on CPU int8: `uv sync --extra mac` for mlx-whisper "
                  "(~4-5x on Apple Silicon).")
    if dev != "cuda":
        print("  [i ] no CUDA here: qwen synthesis runs on the GPU pod — "
              "`dubadabidu remote run <video> --overlay config.gpu.yaml` "
              "(translation, mix and mux stay local).")
    for mod in ["faster_whisper", "pydub", "soundfile", "srt", "yaml", "openai"]:
        try:
            __import__(mod); check(f"python: {mod}", True)
        except ImportError:
            check(f"python: {mod}", False, "uv sync --extra dev")
    sep = cfg.get("separation", {})
    if sep.get("enabled") and sep.get("backend", "roformer") == "roformer":
        try:
            __import__("audio_separator"); check("python: audio_separator", True)
        except ImportError:
            check("python: audio_separator", False,
                  'uv sync --extra sep (or separation.backend: demucs)')
    # s6 mastering chain: on in config.yaml, but pedalboard (GPLv3) is the
    # opt-in `.[master]` extra, so a plain `uv sync` stops at s6.
    if cfg.get("mix", {}).get("master", {}).get("enabled"):
        try:
            __import__("pedalboard"); check("python: pedalboard (mix.master)", True)
        except ImportError:
            warn("python: pedalboard missing",
                 "mix.master.enabled is on and s6 will stop without it: "
                 "uv sync --extra master (or set mix.master.enabled: false)")
    # s5 stretcher: rubberband when ffmpeg carries librubberband, else atempo
    if cfg.get("fit", {}).get("stretcher", "auto") != "atempo" \
            and shutil.which("ffmpeg"):
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
        if "rubberband" in filters:
            check("ffmpeg: rubberband filter (s5 stretcher)", True)
        else:
            warn("ffmpeg: no rubberband filter",
                 "s5 falls back to atempo (WSOLA, softer consonants); install "
                 "an ffmpeg built with librubberband")
    engines = {cfg["tts"]["engine"], *cfg["tts"].get("engine_by_lang", {}).values(),
               *cfg.get("bakeoff", {}).get("engines", [])}
    # engines are probed in THIS venv — the only one, since the per-engine
    # venvs merged on 2026-08-02. (The venvs/<engine>/bin/python fallback that
    # used to live here went with them; it could only ever find a tree this
    # repo no longer creates.)
    for eng, mod, hint in [
        ("qwen", "qwen_tts",
         "git clone QwenLM/Qwen3-TTS + pip install -e . (THIRD_PARTY.md)"),
    ]:
        if eng not in engines:
            continue
        try:
            __import__(mod); check(f"python: {mod}", True)
        except ImportError:
            # A GPU engine missing on a machine with no CUDA is CORRECT, not a
            # fault: the remote flow runs s1/s2/s5-s8 here and only s4 on the
            # pod, which installs qwen itself from runpod.engine_setup. Failing
            # doctor for it made every local run end in "doctor: fix items
            # above" over a thing nobody should fix — and a health check that
            # reports a non-problem is one people learn to ignore, which is
            # worse than not having it.
            # Still a hard failure on a CUDA box, and still a failure here if
            # the remote path is not configured, because then there is nowhere
            # for synthesis to happen at all.
            remote_ready = bool(os.environ.get("RUNPOD_API_KEY"))
            if dev != "cuda" and remote_ready:
                print(f"  [i ] python: {mod} not installed locally — fine: "
                      f"the pod installs it (runpod.engine_setup in "
                      f"config.gpu.yaml).")
            else:
                check(f"python: {mod}", False, hint)
    if engines - {"edge"}:
        check("reference_wav", Path(cfg["tts"]["reference_wav"]).exists(),
              f"put a 15-20s clean voice clip at {cfg['tts']['reference_wav']} "
              f"or run `dubadabidu prep <video>`")
    check(f"env {cfg['translation']['api_key_env']}",
          os.environ.get(cfg["translation"]["api_key_env"]),
          "export it, or point translation.base_url at a local Ollama")
    if os.environ.get(cfg["translation"]["api_key_env"]):
        _check_translation_models(cfg.translation, check)
    # remote (M2) readiness — informational; only needed for `dubadabidu remote`
    if os.environ.get("RUNPOD_API_KEY"):
        from dubadabidu.infra.remote.runpod_infra import DEFAULTS as RP_DEFAULTS
        rp = {**RP_DEFAULTS, **(cfg.get("runpod") or {})}
        key_path = Path(rp["ssh_key"]).expanduser()
        check("remote: ssh_key", key_path.exists(),
              f"private key {key_path} missing (ssh-keygen -t ed25519 -f "
              f"{key_path})")
        # its PUBLIC half is what the pod gets (PUBLIC_KEY env): without it
        # the pod comes up and sshd never lets this client in
        check("remote: ssh_key .pub", Path(f"{key_path}.pub").exists(),
              f"{key_path}.pub missing (ssh-keygen -y -f {key_path} > "
              f"{key_path}.pub)")
        hold = float(rp["hold_on_failure_min"] or 0)
        print(f"  [i ] remote ready: budget cap ${rp['budget_usd']} | "
              f"no capacity: retry every {rp['capacity_retry_s']:.0f}s for "
              f"{rp['capacity_wait_min']:.0f} min | on task failure: "
              + (f"pod held {hold:.0f} min for a hotfix"
                 if hold > 0 else "pod terminated (hold off)"))
        if "runpod" not in cfg:
            print("  [i ] runpod block not in this config: pass "
                  "`--overlay config.gpu.yaml` to see the pod settings")
    else:
        print("  [i ] remote (`dubadabidu remote`) off: set RUNPOD_API_KEY in "
              ".env to enable GPU pod automation")
    print("doctor:", "all good" if ok else "fix items above")
    return 0 if ok else 1


def _check_translation_models(tcfg: Translation,
                              check: Callable[[str, object, str], None]) -> None:
    """Ask the endpoint which models it serves. A provider can retire a name
    and keep routing it to a different model (DeepSeek did, 2026-09-10), so a
    config that "works" may be translating with a model nobody evaluated.
    Informational when the endpoint can't be reached — a laptop offline
    should not fail doctor over the network."""
    import os
    from openai import OpenAI
    from dubadabidu.stages.s3_translate import missing_models
    try:
        client = OpenAI(base_url=tcfg["base_url"],
                        api_key=os.environ[tcfg["api_key_env"]], timeout=15)
        missing = missing_models(client, tcfg)
    except Exception as e:   # offline, local server down, no /models route
        print(f"  [i ] translation models not checked ({type(e).__name__}: {e})")
        return
    check("translation models listed by the endpoint", not missing,
          f"{missing} not served by {tcfg['base_url']} — the name may be "
          f"retired or rerouted; check the provider changelog and update "
          f"translation.model")
