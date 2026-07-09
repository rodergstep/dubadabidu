"""Metric primitives for the eval layer.

Model-backed metrics (lazy singletons, CPU-friendly):
  - ECAPA-TDNN speaker embeddings (speechbrain, Apache-2.0) — discriminative
    similarity; resemblyzer GE2E saturates ~0.85+ on any clean same-gender voice.
  - Distill-MOS (distillmos, MIT) — neural MOS naturalness proxy, 1..5.
  - f0 variability (librosa pyin) — monotony indicator, semitone std.

Pure scoring helpers (no models, unit-tested in tests/test_metrics.py):
  - calibrate_sim(): map raw cosine into the [floor..ceiling] band measured on
    this exact reference (floor = ref vs a definitely-different TTS voice,
    ceiling = ref vs the speaker's real vocals). Raw cosines lie; calibrated
    ones are comparable across refs and runs.
  - composite_score(): the tune loop's objective function.
"""
from __future__ import annotations
import logging
from pathlib import Path

log = logging.getLogger("dubadabidu.qc.metrics")
SAMPLE_RATE = 16000

_ecapa = None
_sqa = None


def _load_audio_16k(path: str | Path):
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def ecapa_embed(path_or_wave) -> "torch.Tensor":  # noqa: F821
    """Speaker embedding of a file path or a [1, T] 16 kHz tensor."""
    global _ecapa
    if _ecapa is None:
        from speechbrain.inference.speaker import EncoderClassifier
        log.info("loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) ...")
        _ecapa = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb",
            savedir="work/.models/ecapa", run_opts={"device": "cpu"})
    wav = path_or_wave if hasattr(path_or_wave, "dim") else _load_audio_16k(path_or_wave)
    return _ecapa.encode_batch(wav).squeeze()


def cosine(a, b) -> float:
    import torch.nn.functional as F
    return float(F.cosine_similarity(a, b, dim=0))


def mos(path: str | Path) -> float:
    """Distill-MOS naturalness estimate, 1..5."""
    global _sqa
    import torch
    if _sqa is None:
        import distillmos
        log.info("loading Distill-MOS ...")
        _sqa = distillmos.ConvTransformerSQAModel()
        _sqa.eval()
    with torch.no_grad():
        return float(_sqa(_load_audio_16k(path)))


def mos_min_window(path: str | Path, win_s: float = 3.0, hop_s: float = 1.0,
                   at: list | None = None) -> float:
    """Minimum MOS over speech-bearing sliding windows — catches brief glitches
    that a whole-take average hides (0.3s of garble in a 17s take barely moves
    the mean but craters its window).

    Windows dominated by silence are skipped: MOS models rate silence and
    natural pauses as 'bad audio', which would flag every long take. Glitches
    are energetic, so energy gating keeps them scoreable.
    Pass `at=[]` to receive the offending window's start second (diagnostics).
    """
    global _sqa
    import torch
    mos(path) if _sqa is None else None  # ensure model loaded
    wav = _load_audio_16k(path)
    n = wav.shape[1]
    win, hop = int(win_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
    if n <= win:
        return mos(path)
    peak = float(wav.abs().max()) or 1.0
    best, best_at = None, 0.0
    with torch.no_grad():
        for s in range(0, n - win + 1, hop):
            w = wav[:, s:s + win]
            active = float((w.abs() > 0.05 * peak).float().mean())
            if active < 0.35:          # mostly pause — skip
                continue
            m = float(_sqa(w))
            if best is None or m < best:
                best, best_at = m, s / SAMPLE_RATE
    if at is not None:
        at.append(best_at)
    return best if best is not None else mos(path)


def f0_semitone_std(path: str | Path) -> float:
    """Std of voiced f0 in semitones around its median. < ~1.5 ≈ monotone."""
    import numpy as np
    import librosa
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SAMPLE_RATE)
    f0 = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
    if f0 is None or len(f0) < 8:
        return 0.0
    semis = 12.0 * np.log2(f0 / np.median(f0))
    return float(np.std(semis))


# ---------- pure helpers ----------

def calibrate_sim(raw: float, floor: float, ceiling: float) -> float:
    """Raw cosine -> 0..1 position inside this reference's measured band."""
    if ceiling <= floor:
        return 0.0
    return max(0.0, min(1.0, (raw - floor) / (ceiling - floor)))


def tempo_penalty(tempo: float, max_tempo: float) -> float:
    """0 at tempo<=1, 1 at tempo==max_tempo."""
    if max_tempo <= 1.0 or tempo <= 1.0:
        return 0.0
    return min(1.0, (tempo - 1.0) / (max_tempo - 1.0))


def composite_score(sim_cal: float, mos_1to5: float, tempo_pen: float,
                    weights: dict, f0st: float = 0.0) -> float:
    """Weighted 0..1 objective. weights: {sim, mos, f0, tempo} summing to ~1.

    Weights calibrated against human ratings (test clip, n=12, 2026-07-08):
    rating correlates with mos +0.63 and f0 variability +0.48; raw similarity
    went NEGATIVE (-0.30) — over-cloning a thin reference hurts naturalness —
    so sim is kept for identity but demoted below mos.
    """
    mos_n = max(0.0, min(1.0, (mos_1to5 - 1.0) / 4.0))
    f0_n = max(0.0, min(1.0, f0st / 4.0))  # ~4 semitones ≈ lively narration
    return round(weights["sim"] * sim_cal + weights["mos"] * mos_n
                 + weights.get("f0", 0.0) * f0_n
                 - weights["tempo"] * tempo_pen, 4)
