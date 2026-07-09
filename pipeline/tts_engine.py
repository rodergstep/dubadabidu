"""Unified TTS engines. Both s4 (initial synthesis) and s5 (variant retries)
go through synthesize() so caching and device handling live in one place.

Engines:
  chatterbox — chatterbox-tts 0.1.7, Multilingual v3 weights from HF (MIT).
               cfg_weight=0.0 mandatory for Ukrainian reference -> other targets.
  cosyvoice  — Fun-CosyVoice3-0.5B (Apache-2.0), zero-shot cloning; needs the
               reference transcript (tts.reference_text / refs.json) alongside
               the wav. Git-clone install, GPU-phase challenger — see
               IMPROVEMENT_PLAN.md Phase C.
  edge       — edge-tts 7.2.8, free MS neural voices, CPU-only, no cloning.

Per-language routing: tts.engine_by_lang maps a language to an engine,
falling back to tts.engine (manifest.resolve_engine).
"""
from __future__ import annotations
import asyncio, logging, subprocess
from pathlib import Path
from .device import torch_device
from .manifest import resolve_engine
from .text_norm import normalize_for_tts

log = logging.getLogger("dubadabidu.tts")
_chatterbox_model = None
_cosyvoice_model = None


def _load_chatterbox():
    global _chatterbox_model
    if _chatterbox_model is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        dev = torch_device()
        log.info("loading Chatterbox Multilingual v3 on %s ...", dev)
        try:
            _chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
        except Exception as e:  # e.g. an op unsupported on MPS
            if dev != "cpu":
                log.warning("load on %s failed (%s); retrying on cpu", dev, e)
                _chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            else:
                raise
    return _chatterbox_model


def _synth_chatterbox(text: str, lang: str, out: Path, t: dict) -> None:
    import torchaudio as ta
    m = _load_chatterbox()
    ref = t["reference_wav"]
    if not Path(ref).exists():
        raise FileNotFoundError(
            f"reference_wav not found: {ref} — put a clean 15-20s clip of your "
            f"voice there (see ref/README.txt) or switch tts.engine to 'edge'.")
    wav = m.generate(text, language_id=lang, audio_prompt_path=ref,
                     cfg_weight=t["cfg_weight"], exaggeration=t["exaggeration"])
    ta.save(str(out), wav, m.sr)


def _load_cosyvoice(model_dir: str):
    global _cosyvoice_model
    if _cosyvoice_model is None:
        try:  # class name differs across CosyVoice releases; try newest first
            from cosyvoice.cli.cosyvoice import CosyVoice3 as _CV  # type: ignore
        except ImportError:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as _CV  # type: ignore
        log.info("loading CosyVoice from %s ...", model_dir)
        _cosyvoice_model = _CV(model_dir)
    return _cosyvoice_model


def _synth_cosyvoice(text: str, lang: str, out: Path, t: dict) -> None:
    """Fun-CosyVoice3 zero-shot cloning. Requires reference_text — the exact
    transcript of reference_wav (video-donated refs get it from refs.json).
    GPU-first: untested on MPS; expected to run on the RunPod box."""
    import torch
    import torchaudio as ta
    try:
        from cosyvoice.utils.file_utils import load_wav  # type: ignore
    except ImportError as e:
        raise FileNotFoundError(
            "cosyvoice package not importable — git clone --recursive "
            "https://github.com/FunAudioLLM/CosyVoice (pin the commit), install "
            "its requirements, and add it to PYTHONPATH. See IMPROVEMENT_PLAN.md "
            f"Phase C. ({e})")
    ref, ref_text = t["reference_wav"], t.get("reference_text", "")
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference_wav not found: {ref}")
    if not ref_text:
        raise FileNotFoundError(
            "tts.reference_text is required for the cosyvoice engine (the exact "
            "transcript of reference_wav; `dubadabidu prep` writes refs.json "
            "with transcripts for video-extracted refs).")
    m = _load_cosyvoice(t.get("cosyvoice_model_dir",
                              "pretrained_models/Fun-CosyVoice3-0.5B"))
    prompt = load_wav(ref, 16000)
    chunks = [r["tts_speech"] for r in
              m.inference_zero_shot(text, ref_text, prompt, stream=False)]
    ta.save(str(out), torch.cat(chunks, dim=1), m.sample_rate)


def _synth_edge(text: str, lang: str, out: Path, t: dict) -> None:
    import edge_tts
    voice = t["edge_voices"][lang]
    mp3 = out.with_suffix(".tmp.mp3")
    asyncio.run(edge_tts.Communicate(text, voice).save(str(mp3)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", str(t["sample_rate"]), "-ac", "1", str(out)], check=True)
    mp3.unlink(missing_ok=True)


def synth_best_of(text: str, lang: str, out: Path, t: dict) -> float:
    """Synthesize with MOS-gated re-rolls; keep the best take at `out`.
    Gate on the worst 3s window, not the take average — brief garble in a
    long take hides from the mean but not from its window. Used by s4 AND
    s5 (variant retries): a variant that replaces the primary in the mix
    must clear the same gate as the primary."""
    from qc.metrics import mos_min_window
    best_of = int(t.get("best_of", 1))
    threshold = float(t.get("retake_mos_below", 0.0))
    best_score = None
    for take in range(best_of):
        cand = out.with_name(f"{out.stem}_take{take}.wav") if take else out
        synthesize(text, lang, cand, t)
        score = mos_min_window(cand)
        if best_score is None or score > best_score:
            best_score = score
            if take:
                cand.replace(out)
        elif take:
            cand.unlink(missing_ok=True)
        if best_score >= threshold:
            break
        if take + 1 < best_of:
            log.info("retake %d for %r (mos %.2f < %.2f)",
                     take + 1, text[:40], score, threshold)
    return best_score


def synthesize(text: str, lang: str, out: Path, tts_cfg: dict,
               retries: int = 2) -> None:
    """Synthesize with retry (autoregressive TTS occasionally glitches; a retry
    with the same input often fixes it — final quality gate is qc/backcheck)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = resolve_engine(tts_cfg, lang)
    fn = {"chatterbox": _synth_chatterbox, "cosyvoice": _synth_cosyvoice,
          "edge": _synth_edge}[engine]
    if engine == "chatterbox":
        # accent marks etc. exist only here — manifest/subs/QC keep clean text.
        # NOT applied to cosyvoice yet: whether it honors acute marks needs its
        # own A/B (IMPROVEMENT_PLAN.md Phase C).
        text = normalize_for_tts(text, lang)
    # write to a .part file and rename on success: a killed run must never
    # leave a truncated wav that the hash cache would accept as a good take
    tmp = out.with_name(out.stem + ".part.wav")
    last = None
    for attempt in range(1 + retries):
        try:
            fn(text, lang, tmp, tts_cfg)
            tmp.replace(out)
            return
        except FileNotFoundError:
            raise
        except Exception as e:
            last = e
            log.warning("synthesis attempt %d failed for %r: %s", attempt + 1, text[:40], e)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"TTS failed after {retries + 1} attempts: {last}")
