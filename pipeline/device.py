"""Device selection: cuda -> mps (Apple Silicon) -> cpu, with per-component overrides.

Notes:
- faster-whisper (CTranslate2) supports cuda/cpu only — MPS is NOT supported,
  so on a Mac ASR runs on CPU (int8 keeps it tolerable).
- Chatterbox (torch) can try MPS on Apple Silicon; falls back to CPU on error.
"""
from __future__ import annotations
import functools


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
