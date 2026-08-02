"""s1: ffmpeg audio extraction + vocal/background separation.

Backends (separation.backend):
  roformer — BS/Mel-RoFormer checkpoints via audio-separator (UVR registry).
             ~12.9 dB vocal SDR vs htdemucs' ~9 dB — audibly cleaner stems.
             Everything downstream consumes them: ASR input, prep's ref
             candidates, the background bed under the dub, emotion slices.
  demucs   — htdemucs, the previous default; kept as the fallback backend.

Stems are cached per video: switching backends does NOT re-separate an
existing work/<video>/ — delete vocals.wav + background.wav there first.
"""
from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path
from .manifest import video_workdir

log = logging.getLogger("dubadabidu.s1")


def sh(*args: str) -> None:
    subprocess.run(args, check=True)


def _separate_roformer(sep_cfg: dict, work_dir: str, full: Path,
                       wd: Path) -> tuple[Path, Path]:
    """RoFormer separation via audio-separator; returns raw (vocals, background)
    stem paths. Checkpoints auto-download to work/.models/audio-separator;
    device (cuda/mps/cpu) is auto-detected by the library."""
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise FileNotFoundError(
            'audio-separator not importable — pip install "audio-separator[cpu]" '
            "([gpu] only swaps onnxruntime; RoFormer ckpts run on torch and use "
            "CUDA/MPS either way), or set separation.backend: demucs. "
            f"({e})")
    model = sep_cfg.get("roformer_model",
                        "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    out_dir = wd / "roformer"
    # separation.device: cpu — force it off the GPU for THIS call only.
    # audio-separator picks its device by probing torch.backends.mps.is_available()
    # (separator.py), with no constructor override, so the only lever is the
    # probe. On an 8-minute lesson RoFormer's attention asked for 1.19 GB with
    # 26.3 GB already committed system-wide and MPS refused — twice, with an
    # identical figure, so it is standing pressure rather than a transient.
    # NOT raising PYTORCH_MPS_HIGH_WATERMARK_RATIO: torch's own error says that
    # "may cause system failure", which is not a trade worth making on someone's
    # working laptop to save a few minutes of CPU time.
    _restore = None
    if str(sep_cfg.get("device", "")).lower() == "cpu":
        import torch
        if hasattr(torch.backends, "mps"):
            _restore = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False
        log.info("separation forced to CPU (separation.device: cpu) — slower "
                 "but bounded by system RAM rather than the MPS ceiling")
    separator = Separator(
        output_dir=str(out_dir),
        model_file_dir=str(Path(work_dir) / ".models" / "audio-separator"),
        output_format="WAV")
    def _go():
        separator.load_model(model_filename=model)
        return separator.separate(str(full))

    try:
        try:
            outs = _go()
        except RuntimeError as e:
            # AUTOMATIC CPU FALLBACK. Measured on an 8-minute lesson: RoFormer
            # asked MPS for 1.19 GB with 26.3 GB already committed against a
            # 30.2 GB ceiling and was refused — twice, with an identical figure,
            # so standing system pressure rather than a transient. Without this
            # the course runner dies on the FIRST long video of every batch and
            # needs a hand-passed overlay to continue.
            if "out of memory" not in str(e).lower() or _restore is not None:
                raise
            log.warning("separation ran out of GPU memory (%s) — retrying on "
                        "CPU. Slower, but bounded by system RAM. Set "
                        "separation.device: cpu to skip the failed attempt.",
                        str(e).split("(")[0].strip())
            import torch
            if hasattr(torch.backends, "mps"):
                _restore = torch.backends.mps.is_available
                torch.backends.mps.is_available = lambda: False
            separator_cpu = Separator(
                output_dir=str(out_dir),
                model_file_dir=str(Path(work_dir) / ".models" / "audio-separator"),
                output_format="WAV")
            separator_cpu.load_model(model_filename=model)
            outs = separator_cpu.separate(str(full))
    finally:
        if _restore is not None:
            import torch
            torch.backends.mps.is_available = _restore

    def pick(tag: str) -> Path:
        for o in outs:  # names look like "audio_full_(Vocals)_<model>.wav";
            if tag.lower() in Path(o).name.lower():  # returned relative or absolute
                p = Path(o)                          # depending on the version
                return p if p.is_absolute() else out_dir / p.name
        raise RuntimeError(f"no '{tag}' stem among separator outputs: {outs}")

    return pick("(Vocals)"), pick("(Instrumental)")


def _separate_demucs(sep_cfg: dict, full: Path, wd: Path) -> tuple[Path, Path]:
    model = sep_cfg["demucs_model"]
    sh(sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", model,
       "-o", str(wd / "demucs"), str(full))
    stem_dir = wd / "demucs" / model / full.stem
    return stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"


def run(cfg: dict, video: str) -> None:
    wd = video_workdir(cfg, video)
    full = wd / "audio_full.wav"
    if not full.exists():
        sh("ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "44100", str(full))

    vocals = wd / "vocals.wav"
    bg = wd / "background.wav"
    if vocals.exists() and bg.exists():
        print(f"[s1] cached: {wd}")
        return

    sep = cfg["separation"]
    if sep["enabled"]:
        backend = sep.get("backend", "roformer")
        if backend == "roformer":
            raw_voc, raw_bg = _separate_roformer(sep, cfg["work_dir"], full, wd)
        elif backend == "demucs":
            raw_voc, raw_bg = _separate_demucs(sep, full, wd)
        else:
            raise SystemExit(f"separation.backend '{backend}' is invalid "
                             "(use roformer | demucs).")
        sh("ffmpeg", "-y", "-i", str(raw_voc),
           "-ac", "1", "-ar", "16000", str(vocals))          # 16k mono for ASR
        sh("ffmpeg", "-y", "-i", str(raw_bg), str(bg))
    else:
        # no music in source: light denoise instead of separation
        sh("ffmpeg", "-y", "-i", str(full), "-af", "afftdn=nf=-25",
           "-ac", "1", "-ar", "16000", str(vocals))
        sh("ffmpeg", "-y", "-f", "lavfi", "-t", "1", "-i",
           "anullsrc=r=44100:cl=mono", str(bg))               # silent bg placeholder
    print(f"[s1] done: {vocals}, {bg}")
