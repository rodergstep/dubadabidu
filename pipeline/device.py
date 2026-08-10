"""Device selection: cuda -> mps (Apple Silicon) -> cpu, with per-component overrides.

Notes:
- faster-whisper (CTranslate2) supports cuda/cpu only — MPS is NOT supported,
  so on a Mac ASR runs on CPU (int8 keeps it tolerable).
- Chatterbox (torch) can try MPS on Apple Silicon; falls back to CPU on error.
"""
from __future__ import annotations
import functools
import logging
import shutil
from pathlib import Path

log = logging.getLogger("dubadabidu.device")


@functools.lru_cache
def cuda_host() -> bool:
    """True if the MACHINE has an NVIDIA GPU — asked of the driver, not of torch.

    A venv that picked up a CPU-only wheel makes torch.cuda.is_available()
    return False on a perfectly good GPU box, and torch_device() then reports
    "cpu" with no error — which cost a week of 126 s/take runs before this check
    existed. (It was introduced when engines had their OWN venvs and the base
    CUDA preflight could not see into them; the venvs merged 2026-08-02 but a
    silent CPU fallback is worth refusing either way.)"""
    return (Path("/proc/driver/nvidia/version").exists()
            or shutil.which("nvidia-smi") is not None)


def require_gpu(engine: str, allow_cpu: bool = False) -> str:
    """-> the resolved torch device, refusing a silent CPU fallback on a GPU box.

    A CPU-only torch in an engine venv is a broken install, not a reason to run
    ~40x slower and bill for it. Measured 2026-07-30: Qwen3-TTS-1.7B took a
    median 126 s/take while voxcpm on the same pod took 2.9 s.

    Set tts.allow_cpu_fallback: true to downgrade this to a warning."""
    dev = torch_device()
    try:
        import torch
        ver = torch.__version__
    except Exception:
        ver = "?"
    log.info("%s: torch %s -> device=%s (host GPU: %s)",
             engine, ver, dev, cuda_host())
    if dev == "cuda" or not cuda_host():
        return dev
    msg = (f"{engine}: host has an NVIDIA GPU but this engine's venv resolved "
           f"torch {ver} -> device={dev}. That is a CPU-only wheel in "
           f"the venv, and synthesis would run ~40x slower at full GPU "
           f"price. Reinstall with an explicit CUDA index-url (see "
           f"runpod.engine_setup in config.gpu.yaml), or set "
           f"tts.allow_cpu_fallback: true to proceed anyway.")
    if not allow_cpu:
        raise RuntimeError(msg)
    log.warning("%s (allow_cpu_fallback)", msg)
    return dev


@functools.lru_cache
def torch_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@functools.lru_cache
def whisper_device(preferred: str = "auto") -> tuple[str, str]:
    """-> (device, compute_type) for faster-whisper/CTranslate2."""
    if preferred == "auto":
        dev = "cuda" if torch_device() == "cuda" else "cpu"
    else:
        dev = "cpu" if preferred == "mps" else preferred
    return dev, ("float16" if dev == "cuda" else "int8")
