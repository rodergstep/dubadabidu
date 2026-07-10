"""Unified TTS engines. Both s4 (initial synthesis) and s5 (variant retries)
go through synthesize() so caching and device handling live in one place.

Engines:
  chatterbox — chatterbox-tts 0.1.7, Multilingual v3 weights from HF (MIT).
               cfg_weight=0.0 mandatory for Ukrainian reference -> other targets.
  cosyvoice  — CosyVoice 2/3 (Apache-2.0). Covers all 5 targets + cross-lingual
               cloning. cosyvoice_mode picks the inference path:
                 cross_lingual — audio prompt ONLY, no ref transcript. The path
                   for a Ukrainian ref (UA is not a CosyVoice-supported language,
                   so its transcript can't be tokenized — cross-lingual clones
                   timbre from the audio alone).
                 zero_shot     — ref audio + ref transcript (reference_text).
                 instruct      — ref audio + a natural-language style/emotion
                   instruction (instruct_text) — per-segment prosody without any
                   custom modulation code. Also transcript-free (UA-ref safe).
  indextts   — IndexTTS-2 (Apache-2.0). EN + Mandarin ONLY (guarded). Built for
               dubbing: DISENTANGLED emotion (a separate emotion_wav prompt — set
               it to the source UA slice for real per-segment prosody transfer)
               and duration control. English-track challenger.
  edge       — edge-tts 7.2.8, free MS neural voices, CPU-only, no cloning.

The cosyvoice/indextts backends are GPU-only git-clone installs — see
THIRD_PARTY.md. They fail with an actionable FileNotFoundError (which
synthesize() re-raises without retry) when their package isn't importable.

Per-language routing: tts.engine_by_lang maps a language to an engine,
falling back to tts.engine (manifest.resolve_engine). The winning assignment
is decided by the bake-off (qc/bakeoff.py), not by reputation.
"""
from __future__ import annotations
import asyncio, logging, subprocess
from pathlib import Path
from .device import torch_device
from .manifest import resolve_engine
from .text_norm import normalize_for_tts

log = logging.getLogger("dubadabidu.tts")
INDEXTTS_LANGS = {"en", "zh"}   # IndexTTS-2 native coverage; others need finetune
_chatterbox_model = None
_cosyvoice_model = None
_indextts_model = None


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


def _cosyvoice_mode(t: dict) -> str:
    """Which CosyVoice inference path. Default cross_lingual: the reference is
    Ukrainian (not a CosyVoice-supported language), so its transcript can't be
    tokenized — clone timbre from the audio alone. auto upgrades to instruct
    when an instruct_text is present, or zero_shot when a ref transcript is."""
    mode = t.get("cosyvoice_mode", "cross_lingual")
    if mode != "auto":
        return mode
    if t.get("instruct_text"):
        return "instruct"
    return "zero_shot" if t.get("reference_text") else "cross_lingual"


def _synth_cosyvoice(text: str, lang: str, out: Path, t: dict) -> None:
    """CosyVoice 2/3 cloning. Mode-dependent (see _cosyvoice_mode); the default
    cross_lingual path needs the ref audio only — the right choice for a
    Ukrainian reference. GPU-first: expected to run on the RunPod box."""
    import torch
    import torchaudio as ta
    try:
        from cosyvoice.utils.file_utils import load_wav  # type: ignore
    except ImportError as e:
        raise FileNotFoundError(
            "cosyvoice package not importable — git clone --recursive "
            "https://github.com/FunAudioLLM/CosyVoice (pin the commit), install "
            "its requirements, and add it to PYTHONPATH. See THIRD_PARTY.md. "
            f"({e})")
    ref = t["reference_wav"]
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference_wav not found: {ref}")
    m = _load_cosyvoice(t.get("cosyvoice_model_dir",
                              "pretrained_models/CosyVoice2-0.5B"))
    prompt = load_wav(ref, 16000)
    mode = _cosyvoice_mode(t)
    if mode == "zero_shot":
        ref_text = t.get("reference_text", "")
        if not ref_text:
            raise FileNotFoundError(
                "cosyvoice_mode=zero_shot needs tts.reference_text (the ref "
                "transcript; `dubadabidu prep` writes refs.json). A Ukrainian "
                "ref cannot be tokenized here — use cosyvoice_mode=cross_lingual.")
        gen = m.inference_zero_shot(text, ref_text, prompt, stream=False)
    elif mode == "instruct":
        instruct = t.get("instruct_text") or "Speak naturally."
        # instruct2: prompt speech + a style/emotion instruction, transcript-free
        gen = m.inference_instruct2(text, instruct, prompt, stream=False)
    else:  # cross_lingual — audio prompt only, UA-ref safe
        gen = m.inference_cross_lingual(text, prompt, stream=False)
    chunks = [r["tts_speech"] for r in gen]
    ta.save(str(out), torch.cat(chunks, dim=1), m.sample_rate)


def _load_indextts(model_dir: str):
    global _indextts_model
    if _indextts_model is None:
        try:
            from indextts.infer_v2 import IndexTTS2  # type: ignore
        except ImportError as e:
            raise FileNotFoundError(
                "indextts package not importable — git clone "
                "https://github.com/index-tts/index-tts (pin the commit), install "
                "its requirements + download checkpoints, add it to PYTHONPATH. "
                f"See THIRD_PARTY.md. ({e})")
        cfg_path = str(Path(model_dir) / "config.yaml")
        log.info("loading IndexTTS-2 from %s ...", model_dir)
        _indextts_model = IndexTTS2(cfg_path=cfg_path, model_dir=model_dir,
                                    use_fp16=True)
    return _indextts_model


def _synth_indextts(text: str, lang: str, out: Path, t: dict) -> None:
    """IndexTTS-2: zero-shot clone from reference_wav, with optional disentangled
    emotion prompt (emotion_wav — set it per segment to the source UA slice for
    real prosody transfer) and duration control. EN/Mandarin only."""
    if lang not in INDEXTTS_LANGS:
        raise FileNotFoundError(
            f"indextts engine does not support '{lang}' (native: "
            f"{sorted(INDEXTTS_LANGS)}). Route {lang} to chatterbox/cosyvoice "
            f"via tts.engine_by_lang.")
    ref = t["reference_wav"]
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference_wav not found: {ref}")
    m = _load_indextts(t.get("indextts_model_dir", "checkpoints"))
    kw = {"spk_audio_prompt": ref, "text": text, "output_path": str(out),
          "verbose": False}
    emo = t.get("emotion_wav")
    if emo and Path(emo).exists():   # disentangled per-segment emotion prompt
        kw["emo_audio_prompt"] = emo
        kw["emo_alpha"] = float(t.get("emo_alpha", 1.0))
    elif t.get("instruct_text"):     # or a text emotion description
        kw["use_emo_text"] = True
        kw["emo_text"] = t["instruct_text"]
    if t.get("indextts_duration_ratio"):   # global 0.75-1.25x pace control
        kw["duration_ratio"] = float(t["indextts_duration_ratio"])
    m.infer(**kw)   # writes output_path itself


def _synth_edge(text: str, lang: str, out: Path, t: dict) -> None:
    import edge_tts
    voice = t["edge_voices"][lang]
    mp3 = out.with_suffix(".tmp.mp3")
    asyncio.run(edge_tts.Communicate(text, voice).save(str(mp3)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", str(t["sample_rate"]), "-ac", "1", str(out)], check=True)
    mp3.unlink(missing_ok=True)


PACE_TOL = 0.35        # |dur/target - 1| that zeroes the pace term
FIT_SLACK = 1.10       # takes within target*this count as "fitting the slot"


def _take_rank(m: dict, target_dur: float | None = None) -> float:
    """Relative ranking of takes of the SAME text/ref. Weights follow the
    human-rating calibration (mos +.63, f0 +.48, sim kept for identity):
    windowed MOS first, prosody liveliness and raw ECAPA sim next.
    Raw (uncalibrated) cosine is fine here — all takes share one reference,
    only the ordering matters. Renormalized when sim is unavailable (edge
    engine / missing ref).

    Pace term (when target_dur is known): chatterbox pacing varies wildly
    between rolls of the same text (measured −18%..+45% on sketch60), so
    without this term the delivered speed is a dice roll — a fast-talking
    take can win on MOS alone and the dub's rhythm drifts run to run.
    Takes closest to the source slot's pace score highest; PACE_TOL away
    scores zero."""
    mos_n = max(0.0, min(1.0, (m["mos_min"] - 1.0) / 4.0))
    f0_n = max(0.0, min(1.0, m.get("f0st", 0.0) / 4.0))
    parts = [(0.5, mos_n), (0.25, f0_n)]
    if m.get("sim") is not None:
        parts.append((0.25, m["sim"]))
    if target_dur and m.get("dur"):
        pace = max(0.0, 1.0 - abs(m["dur"] / target_dur - 1.0) / PACE_TOL)
        parts.append((0.25, pace))
    total = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total


def synth_best_of(text: str, lang: str, out: Path, t: dict,
                  meta: list | None = None, verify_cfg: dict | None = None,
                  verify_text: str | None = None,
                  target_dur: float | None = None) -> float:
    """Synthesize best_of takes; keep the winner at `out`.

    Two modes (tts.rank_takes, default true):
      ranked — synthesize ALL best_of takes, score each on a composite of
        windowed MOS + f0 liveliness + ECAPA similarity (_take_rank), and pick
        the composite-best among takes clearing the MOS floor. The old
        MOS-only early-stop froze flat or off-voice takes into the cache
        forever the moment they cleared the naturalness gate — monotony and
        clone drift were invisible to take selection by design.
      legacy — early-stop on the first take with worst-3s-window MOS >=
        retake_mos_below (cheapest; set rank_takes: false).

    Optional WER veto (verify_cfg + verify_text, used by autopilot re-rolls of
    WER-flagged segments): takes whose back-transcription WER exceeds
    qc.wer_flag_threshold are disqualified unless every take fails — a re-roll
    triggered by hallucination must not be won by another hallucination.

    `target_dur` (the source slot in seconds, passed by s4/s5) adds a pacing
    term to the ranking and an eligibility gate: takes fitting the slot
    (dur <= target*FIT_SLACK) are preferred outright — a take that overflows
    forces a shortened variant downstream, losing content for no reason when
    a fitting roll exists. Relaxed only if every take overflows.

    `meta`, if given, receives one dict per take (mos_min/f0st/sim/dur/wer/
    rank/picked) — the per-take record the weight re-fit and FIXES.md
    diagnostics read. Used by s4 AND s5 (variant retries): a variant that
    replaces the primary in the mix must clear the same gate as the primary.
    """
    import soundfile as sf
    from qc.metrics import ecapa_embed, cosine, f0_semitone_std, mos_min_window
    best_of = int(t.get("best_of", 1))
    threshold = float(t.get("retake_mos_below", 0.0))
    ranked = bool(t.get("rank_takes", True)) and best_of > 1
    engine = resolve_engine(t, lang)
    ref = t.get("reference_wav")
    ref_emb = (ecapa_embed(ref) if ranked and engine != "edge"
               and ref and Path(ref).exists() else None)
    wer_max = (float(verify_cfg["qc"]["wer_flag_threshold"])
               if verify_cfg and verify_text else None)

    takes = []  # (path, metrics)
    for take in range(best_of):
        cand = out.with_name(f"{out.stem}_take{take}.wav") if take else out
        synthesize(text, lang, cand, t)
        info = sf.info(str(cand))
        m = {"mos_min": round(mos_min_window(cand), 2),
             "dur": round(info.frames / info.samplerate, 2)}
        if ranked:
            m["f0st"] = round(f0_semitone_std(cand), 2)
            if ref_emb is not None:
                m["sim"] = round(cosine(ref_emb, ecapa_embed(cand)), 3)
        if wer_max is not None:
            from qc.backcheck import segment_wer
            m["wer"] = round(segment_wer(verify_cfg, verify_text, cand, lang), 3)
        takes.append((cand, m))
        if not ranked and m["mos_min"] >= threshold \
                and (wer_max is None or m["wer"] <= wer_max):
            break
        if take + 1 < best_of and not ranked:
            log.info("retake %d for %r (mos %.2f < %.2f)",
                     take + 1, text[:40], m["mos_min"], threshold)

    # eligibility gates, relaxed only when they would eliminate every take
    pool = takes
    if wer_max is not None:
        ok = [tk for tk in pool if tk[1]["wer"] <= wer_max]
        pool = ok or pool
    if target_dur:
        fits = [tk for tk in pool if tk[1]["dur"] <= target_dur * FIT_SLACK]
        pool = fits or pool
    gated = [tk for tk in pool if tk[1]["mos_min"] >= threshold]
    pool = gated or pool
    key = (lambda m: _take_rank(m, target_dur)) if ranked \
        else (lambda m: m["mos_min"])
    winner = max(pool, key=lambda tk: key(tk[1]))
    for path, m in takes:
        m["rank"] = round(key(m), 4)
        m["picked"] = path is winner[0]
        if meta is not None:
            meta.append(m)
    if winner[0] is not out:
        winner[0].replace(out)
    for path, _ in takes:
        if path is not winner[0] and path is not out:
            path.unlink(missing_ok=True)
    if len(takes) > 1:
        log.info("picked take %d/%d for %r (rank %.3f, mos %.2f)",
                 takes.index(winner) + 1, len(takes), text[:40],
                 winner[1]["rank"], winner[1]["mos_min"])
    return winner[1]["mos_min"]


def synthesize(text: str, lang: str, out: Path, tts_cfg: dict,
               retries: int = 2) -> None:
    """Synthesize with retry (autoregressive TTS occasionally glitches; a retry
    with the same input often fixes it — final quality gate is qc/backcheck)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = resolve_engine(tts_cfg, lang)
    fn = {"chatterbox": _synth_chatterbox, "cosyvoice": _synth_cosyvoice,
          "indextts": _synth_indextts, "edge": _synth_edge}[engine]
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
