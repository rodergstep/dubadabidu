"""ASR backend abstraction: faster-whisper (CUDA/CPU) or mlx-whisper (Apple Silicon).

Both backends yield the SAME normalized segment shape so s2 stays
backend-agnostic:

    [{"start": float, "end": float, "text": str,
      "words": [{"start": float, "end": float, "word": str}, ...]}, ...]

faster-whisper (CTranslate2) runs on CUDA/CPU and carries a Silero vad_filter.
mlx-whisper runs Whisper through Apple's MLX (Metal/ANE) — measured ~4-5x
faster than CPU int8 on Apple Silicon (voicebox) — closing the gap that forced
the "prototype on the Mac, clone on a rented GPU" split for transcription. It
has no VAD, but s1 already isolates vocals (BS-RoFormer) so non-speech is
largely gone before ASR — tolerable for the local path. The cloned CUDA batch
keeps faster-whisper + VAD unchanged (backend auto-resolves to `faster` there).
"""
from __future__ import annotations
import logging
from pathlib import Path
from .device import torch_device, whisper_device

log = logging.getLogger("dubadabidu.asr")

# faster-whisper short names -> mlx-community HF repos (Metal-ready conversions).
_MLX_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def resolve_backend(a: dict) -> str:
    """-> 'faster' | 'mlx'. `asr.backend` (default 'auto') picks mlx on Apple
    Silicon when mlx-whisper is importable, else faster-whisper. An explicit
    `asr.device` of cuda/cpu keeps faster-whisper (that's its territory)."""
    backend = a.get("backend", "auto")
    if backend != "auto":
        return backend
    if a.get("device", "auto") in ("cuda", "cpu"):
        return "faster"
    if torch_device() == "mps":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            log.info("mlx-whisper not installed; ASR falls back to "
                     "faster-whisper on CPU — `pip install mlx-whisper` for "
                     "~4-5x on Apple Silicon")
    return "faster"


def _mlx_repo(a: dict) -> str:
    if a.get("mlx_model"):
        return a["mlx_model"]
    model = a["model"]
    if "/" in model:   # already an HF repo id
        return model
    repo = _MLX_REPOS.get(model)
    if repo is None:
        raise ValueError(
            f"no known mlx-community repo for asr.model={model!r} — set "
            f"asr.mlx_model to an MLX Whisper HF repo (e.g. "
            f"mlx-community/whisper-large-v3-mlx) or asr.backend: faster")
    return repo


def _norm_words(raw) -> list[dict]:
    """Keep only fully-timestamped words, from either backend's shape (faster-
    whisper Word objects with attributes, or mlx-whisper dicts). A word missing
    start/end/text is DROPPED rather than crashing: mlx occasionally emits a word
    without timestamps, and a None would KeyError here or feed the pause splitter
    (s2 -> logic.split_at_pauses) arithmetic on None downstream."""
    out = []
    for w in (raw or []):
        if isinstance(w, dict):
            ws, we, wt = w.get("start"), w.get("end"), w.get("word")
        else:  # faster-whisper Word (attribute access)
            ws, we, wt = (getattr(w, "start", None), getattr(w, "end", None),
                          getattr(w, "word", None))
        if ws is None or we is None or wt is None:
            continue
        out.append({"start": ws, "end": we, "word": wt})
    return out


def _transcribe_faster(a: dict, audio: Path, language: str,
                       prompt: str | None) -> list[dict]:
    from faster_whisper import WhisperModel
    dev, ctype = whisper_device(a.get("device", "auto"))
    log.info("ASR %s via faster-whisper on %s/%s", a["model"], dev, ctype)
    model = WhisperModel(a["model"], device=dev, compute_type=ctype)
    segments, _ = model.transcribe(
        str(audio), language=language, vad_filter=a["vad_filter"],
        word_timestamps=True, initial_prompt=prompt,
        no_speech_threshold=a.get("no_speech_threshold", 0.6),
        temperature=0.0)  # no fallback sampling: segmentation must be reproducible
    out = []
    for s in segments:
        out.append({"start": s.start, "end": s.end, "text": s.text,
                    "words": _norm_words(s.words)})
    return out


def _transcribe_mlx(a: dict, audio: Path, language: str,
                    prompt: str | None) -> list[dict]:
    import mlx_whisper
    repo = _mlx_repo(a)
    log.info("ASR %s via mlx-whisper (Apple Silicon Metal/ANE)", repo)
    if a.get("vad_filter"):
        log.info("mlx-whisper has no VAD; relying on s1 vocal separation "
                 "(non-speech already removed)")
    # no VAD here, so lean on Whisper's own no-speech gate to suppress phantom
    # text in the residual the separator left behind (a known VAD-less failure
    # mode). condition_on_previous_text is exposed too: True keeps context but can
    # propagate a hallucination into a repetition loop — flip it off per-config if
    # a transcript shows runaway repeats.
    res = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=repo, language=language,
        word_timestamps=True, initial_prompt=prompt,
        temperature=0.0,   # disable fallback sampling -> reproducible segmentation
        no_speech_threshold=a.get("no_speech_threshold", 0.6),
        condition_on_previous_text=a.get("condition_on_previous_text", True))
    out = []
    for s in res.get("segments", []):
        out.append({"start": s["start"], "end": s["end"],
                    "text": s["text"], "words": _norm_words(s.get("words"))})
    return out


def transcribe(a: dict, audio: Path, language: str,
               prompt: str | None) -> list[dict]:
    """Normalized transcription; backend chosen by resolve_backend(a)."""
    fn = _transcribe_mlx if resolve_backend(a) == "mlx" else _transcribe_faster
    return fn(a, audio, language, prompt)
