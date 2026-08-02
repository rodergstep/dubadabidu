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
        # See _transcribe_mlx for the full account. temperature=0.0 alone
        # DISABLES Whisper's own repetition rescue; the fallback ladder only
        # engages when a decode fails compression_ratio_threshold, so ordinary
        # segments stay at 0.0 and reproducible.
        compression_ratio_threshold=a.get("compression_ratio_threshold", 2.4),
        condition_on_previous_text=a.get("condition_on_previous_text", False),
        temperature=tuple(a.get("temperature_fallback",
                                (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))))
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
    # THE REPETITION LOOP THIS DEFENDS AGAINST — it happened, on the first
    # production lesson (2026-08-02). The tail of an 8-minute video came back as
    # "Він практично не має запаху." SEVEN times, over audio that measured
    # 30-77% voiced: two real sentences of technique and the outro were replaced
    # by a repeated line, then translated into five languages and synthesized.
    #
    # Two causes, and BOTH had to be wrong for it to happen:
    #  1. condition_on_previous_text=True fed each (wrong) segment back as
    #     context, so once it repeated it kept repeating. The old comment here
    #     predicted exactly this and still defaulted it on.
    #  2. temperature=0.0 with no fallback ladder DISABLED Whisper's own rescue.
    #     Whisper detects a failed decode via compression_ratio_threshold —
    #     repetitive text compresses unusually well — and retries hotter. With a
    #     single temperature that safety net can never fire.
    #
    # Reproducibility (the reason 0.0 was pinned) survives: the ladder is only
    # consulted when a decode FAILS the thresholds, so healthy segments still
    # decode greedily at 0.0. Determinism where it works, recovery where it does
    # not, is the better trade — the old setting was deterministic AND wrong.
    #
    # NOTE it cannot be reproduced on a short clip: transcribing just the bad
    # region in isolation gives correct text under BOTH settings, because the
    # poisoned context never accumulates. Validate on the whole file.
    res = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=repo, language=language,
        word_timestamps=True, initial_prompt=prompt,
        no_speech_threshold=a.get("no_speech_threshold", 0.6),
        compression_ratio_threshold=a.get("compression_ratio_threshold", 2.4),
        condition_on_previous_text=a.get("condition_on_previous_text", False),
        temperature=tuple(a.get("temperature_fallback",
                                (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))))
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
