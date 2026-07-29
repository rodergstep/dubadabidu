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
  voxcpm     — VoxCPM2 (Apache-2.0, pip `voxcpm`). 2B, 30 languages auto-detected
               (covers all 5 targets; UA is NOT among them — clone from the ref
               audio alone, transcript-free). 48 kHz output, ~8 GB VRAM. Style
               control via instruct_text -> the "(...)" text prefix it parses.
  qwen       — Qwen3-TTS 12Hz Base (Apache-2.0, git-clone `qwen_tts`). 0.6B/1.7B,
               10 languages incl. all 5 targets, 3 s clone, ~4 GB VRAM. UA ref
               -> x_vector_only_mode (speaker embedding only, no ref transcript).
               The reusable clone prompt is built once per ref and cached.
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
from .text_norm import normalize_for_tts, localize_numbers

log = logging.getLogger("dubadabidu.tts")
INDEXTTS_LANGS = {"en", "zh"}   # IndexTTS-2 native coverage; others need finetune
_chatterbox_model = None
_cosyvoice_model = None
_indextts_model = None
_voxcpm_model = None
_qwen_model = None
_qwen_prompt = None   # (cache_key, prompt_items) — reused across all segments
QWEN_LANGS = {"en": "English", "fr": "French", "de": "German",
              "es": "Spanish", "ru": "Russian"}


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


def _load_voxcpm(model_dir: str):
    global _voxcpm_model
    if _voxcpm_model is None:
        try:
            from voxcpm import VoxCPM  # type: ignore
        except ImportError as e:
            raise FileNotFoundError(
                "voxcpm package not importable — on the torch-2.6 stack install "
                "it WITH pins: pip install voxcpm==2.0.3 torchcodec==0.2.1 "
                "torch==2.6.0 torchaudio==2.6.0 'numpy<2'. See THIRD_PARTY.md. "
                f"({e})")
        log.info("loading VoxCPM2 from %s ...", model_dir)
        _voxcpm_model = VoxCPM.from_pretrained(model_dir, load_denoiser=False)
    return _voxcpm_model


def _synth_voxcpm(text: str, lang: str, out: Path, t: dict) -> None:
    """VoxCPM2 cloning from the reference audio ALONE (reference_wav_path) —
    Ukrainian is not among its 30 languages, so the transcript-based 'ultimate
    cloning' mode (prompt_wav_path + prompt_text) is unusable with a UA ref.
    Language of `text` is auto-detected; no seed is passed so best_of takes
    vary. instruct_text becomes the "(...)" style prefix VoxCPM2 parses."""
    import soundfile as sf
    ref = t["reference_wav"]
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference_wav not found: {ref}")
    m = _load_voxcpm(t.get("voxcpm_model_dir", "openbmb/VoxCPM2"))
    if t.get("instruct_text"):
        text = f"({t['instruct_text']})" + text
    wav = m.generate(text=text, reference_wav_path=ref,
                     cfg_value=float(t.get("voxcpm_cfg_value", 2.0)),
                     inference_timesteps=int(t.get("voxcpm_timesteps", 10)))
    sf.write(str(out), wav, m.tts_model.sample_rate)


def _load_qwen(t: dict):
    global _qwen_model
    if _qwen_model is None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as e:
            raise FileNotFoundError(
                "qwen_tts not importable — git clone "
                "https://github.com/QwenLM/Qwen3-TTS (pin the commit), "
                "pip install -e . (add flash-attn only on CUDA). See "
                f"THIRD_PARTY.md. ({e})")
        dev = torch_device()
        model_id = t.get("qwen_model_dir", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        kw = {"device_map": "cuda:0" if dev == "cuda" else dev,
              "dtype": torch.bfloat16 if dev == "cuda" else torch.float32}
        # flash-attn is a heavy CUDA-only build; opt-in (qwen_flash_attn) so a
        # missing wheel can't fail the load on a fresh pod
        if dev == "cuda" and t.get("qwen_flash_attn", False):
            kw["attn_implementation"] = "flash_attention_2"
        log.info("loading Qwen3-TTS from %s on %s ...", model_id, dev)
        _qwen_model = Qwen3TTSModel.from_pretrained(model_id, **kw)
    return _qwen_model


def _qwen_clone_prompt(m, t: dict):
    """Reusable clone prompt for the current reference, built once and cached
    (a full video is hundreds of segments off ONE ref). x_vector_only_mode:
    speaker embedding only, no ref transcript — the path for a Ukrainian ref
    (its transcript can't be tokenized). Full mode needs a target-lang ref +
    reference_text and is opt-in via qwen_x_vector_only: false."""
    global _qwen_prompt
    ref = t["reference_wav"]
    ref_text = t.get("reference_text", "")
    x_only = bool(t.get("qwen_x_vector_only", True)) or not ref_text
    key = (ref, x_only, ref_text if not x_only else "")
    if _qwen_prompt is None or _qwen_prompt[0] != key:
        kw = {"ref_audio": ref, "x_vector_only_mode": x_only}
        if not x_only:
            kw["ref_text"] = ref_text
        _qwen_prompt = (key, m.create_voice_clone_prompt(**kw))
    return _qwen_prompt[1]


def _synth_qwen(text: str, lang: str, out: Path, t: dict) -> None:
    """Qwen3-TTS Base voice clone. Covers all 5 targets; UA ref cloned via the
    speaker-embedding-only path (see _qwen_clone_prompt). Returns (wavs, sr)."""
    import soundfile as sf
    ref = t["reference_wav"]
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference_wav not found: {ref}")
    m = _load_qwen(t)
    prompt = _qwen_clone_prompt(m, t)
    wavs, sr = m.generate_voice_clone(
        text=text, language=QWEN_LANGS.get(lang, "Auto"),
        voice_clone_prompt=prompt)
    sf.write(str(out), wavs[0], sr)


def release_models() -> None:
    """Drop every in-process engine singleton and flush the CUDA cache. The
    bake-off switches engines sequentially and pairs this with
    engine_client.shutdown(engine); without both, models accumulate in VRAM
    (chatterbox ~7 GB + voxcpm ~8 GB) and a 16 GB card OOMs mid-comparison.
    The next synthesize() through a released engine just reloads it."""
    global _chatterbox_model, _cosyvoice_model, _indextts_model, \
        _voxcpm_model, _qwen_model, _qwen_prompt
    _chatterbox_model = _cosyvoice_model = _indextts_model = None
    _voxcpm_model = _qwen_model = _qwen_prompt = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:   # torch absent (edge-only env) — nothing was resident
        pass


def _synth_edge(text: str, lang: str, out: Path, t: dict) -> None:
    import edge_tts
    voice = t["edge_voices"][lang]
    mp3 = out.with_suffix(".tmp.mp3")
    asyncio.run(edge_tts.Communicate(text, voice).save(str(mp3)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", str(t["sample_rate"]), "-ac", "1", str(out)], check=True)
    mp3.unlink(missing_ok=True)


MIN_EMO_SLICE_S = 1.0   # shorter source slices carry no usable prosody


def with_source_emotion(t: dict, wd: Path, u: dict, lang: str) -> dict:
    """Per-segment prosody transfer (tts.emotion_from_source, IndexTTS-2 only):
    the emotion prompt becomes THIS utterance's slice of the source vocals, so
    the dub carries the speaker's original delivery segment by segment —
    IndexTTS-2 disentangles it from timbre. Slices are cut once into
    work/<video>/emo/ and synth_hash keys on emotion_wav, so every segment
    caches under its own slice. No-op for other engines or when disabled."""
    if not t.get("emotion_from_source") or resolve_engine(t, lang) != "indextts":
        return t
    if u["end"] - u["start"] < MIN_EMO_SLICE_S:
        return t
    src = wd / "vocals.wav"
    if not src.exists():
        log.warning("emotion_from_source: %s missing — run s1 first; "
                    "synthesizing without emotion prompt", src)
        return t
    out = wd / "emo" / f"{u['id']}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(u["start"]), "-to", str(u["end"]),
                        "-i", str(src), str(out)], check=True)
    return {**t, "emotion_wav": str(out)}


PACE_TOL = 0.35        # |dur/target - 1| that zeroes the pace term
FIT_SLACK = 1.10       # takes within target*this count as "fitting the slot"


def _f0_delivered(cand: Path) -> float:
    """f0 liveliness of the take AS DELIVERED — i.e. after s6's edge trim.

    Take selection used to measure f0 on the raw take while the ear (and
    evaluate's qc_f0st) get the PLACED segment, which s6 trims at TRIM_DB
    before mixing. Trimming changes which frames librosa.pyin sees as voiced,
    and the gap is large and one-directional: measured on sketch60, raw-take
    f0st ran 0.1-1.4 semitones ABOVE the delivered value (ru/u0001: 3.45 raw
    vs 2.02 delivered, with no time-stretch involved at all).

    That made tts.min_f0st a floor in the wrong domain — takes clearing 2.2 at
    selection could land near 2.0 in the mix — and it fed the composite ranker
    an f0 term that partly measured leading silence. Measuring post-trim puts
    selection and evaluate in the same units, so min_f0st means one thing.

    Only f0 moves to the delivered domain. mos_min/sim/dur keep their raw-take
    thresholds (retake_mos_below, early_accept_*, FIT_SLACK) — those are
    calibrated against raw takes and s5 places on raw-take durations.
    """
    from qc.metrics import f0_semitone_std
    from .s6_mix import _clean               # the exact trim s6 applies
    from pydub import AudioSegment
    try:
        trimmed = _clean(AudioSegment.from_wav(cand).set_channels(1))
    except Exception as e:      # never let a measurement failure kill a synth
        log.warning("post-trim f0 measurement failed for %s (%s); "
                    "falling back to the raw take", cand.name, e)
        return f0_semitone_std(cand)
    tmp = cand.with_name(cand.stem + ".f0tmp.wav")
    try:
        trimmed.export(tmp, format="wav")
        return f0_semitone_std(tmp)
    finally:
        tmp.unlink(missing_ok=True)


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


def early_accept_ok(m: dict, mos_floor: float, sim_floor: float,
                    min_f0st: float = 0.0, target_dur: float | None = None,
                    wer_max: float | None = None) -> bool:
    """Is this take good enough to stop rolling (tts.best_of_early_accept)?

    Every quality floor the winner would later have to clear must appear here,
    because accepting early leaves a ONE-take pool and every downstream filter
    in synth_best_of degrades to a no-op on a one-element pool
    (`pool = filtered or pool`). f0 was missing from this gate, so early-accept
    silently bypassed tts.min_f0st entirely — worst on the GPU config, where
    best_of is 5 and early-accept fires most often.
    """
    if m["mos_min"] < mos_floor:
        return False
    if target_dur and m["dur"] > target_dur * FIT_SLACK:
        return False
    if m.get("sim") is not None and m["sim"] < sim_floor:
        return False
    if wer_max is not None and m.get("wer", 1.0) > wer_max:
        return False
    if min_f0st and m.get("f0st", 0.0) < min_f0st:
        return False
    return True


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

    Crash-safe by construction: takes roll to their own `_take{k}.wav` files and
    `out` (the hash-cache key) is created only by the winner's atomic rename. A
    run killed mid-best-of therefore caches NOTHING and re-rolls cleanly on the
    next pass — which is what makes preemptible spot pods safe to use here.
    """
    import soundfile as sf
    from qc.metrics import ecapa_embed, cosine, mos_min_window
    best_of = int(t.get("best_of", 1))
    threshold = float(t.get("retake_mos_below", 0.0))
    ranked = bool(t.get("rank_takes", True)) and best_of > 1
    engine = resolve_engine(t, lang)
    ref = t.get("reference_wav")
    ref_emb = (ecapa_embed(ref) if ranked and engine != "edge"
               and ref and Path(ref).exists() else None)
    wer_max = (float(verify_cfg["qc"]["wer_flag_threshold"])
               if verify_cfg and verify_text else None)
    # adaptive best_of (ranked mode): if an early take already clears a HIGH bar
    # AND fits the slot AND passes any WER veto, stop rolling — the extra takes
    # rarely beat an already-excellent one and each costs GPU time. Quality-safe
    # because the bar is well above the retake floor; disable with
    # tts.best_of_early_accept: false to always synthesize all best_of takes.
    early = ranked and bool(t.get("best_of_early_accept", True))
    early_mos = float(t.get("early_accept_mos", 3.6))   # worst-3s window scores low
    early_sim = float(t.get("early_accept_sim", 0.50))
    min_f0st = float(t.get("min_f0st", 0.0))

    def _early_ok(m: dict) -> bool:
        return early_accept_ok(m, early_mos, early_sim, min_f0st,
                               target_dur, wer_max)

    def _measure(cand: Path) -> dict:
        info = sf.info(str(cand))
        m = {"mos_min": round(mos_min_window(cand), 2),
             "dur": round(info.frames / info.samplerate, 2)}
        if ranked:
            # delivered domain, so this is comparable to min_f0st and to
            # evaluate's qc_f0st (see _f0_delivered)
            m["f0st"] = round(_f0_delivered(cand), 2)
            if ref_emb is not None:
                m["sim"] = round(cosine(ref_emb, ecapa_embed(cand)), 3)
        if wer_max is not None:
            from qc.backcheck import segment_wer
            m["wer"] = round(segment_wer(verify_cfg, verify_text, cand, lang), 3)
        return m

    # Sweep leftovers from a previous ATTEMPT at this exact segment (a run that
    # died mid-best-of, or one that rolled more takes than the current best_of).
    # They are never read — every take below is overwritten before it is scored —
    # but without this they accumulate on disk across crashed pod sessions.
    for junk in out.parent.glob(f"{out.stem}_take*.wav"):
        junk.unlink(missing_ok=True)
    for junk in out.parent.glob(f"{out.stem}_reroll*.wav"):
        junk.unlink(missing_ok=True)

    takes = []  # (path, metrics)
    for take in range(best_of):
        # EVERY take rolls to its own file — never straight to `out`. `out` is
        # the hash-cache key: s4 treats its existence as "this segment has an
        # accepted take" (fresh = not out.exists()). Writing take 0 there meant
        # a run killed before the ranking finished — spot preemption, the budget
        # deadline, the pod self-destruct watchdog, an OOM — left an UNRANKED,
        # ungated take cached as if it had won, with no tr.takes record and no
        # way to notice. Only --force cleared it. `out` is now created by exactly
        # one operation: the atomic rename of the winner, below.
        cand = out.with_name(f"{out.stem}_take{take}.wav")
        synthesize(text, lang, cand, t)
        m = _measure(cand)
        takes.append((cand, m))
        if not ranked and m["mos_min"] >= threshold \
                and (wer_max is None or m["wer"] <= wer_max):
            break
        if early and _early_ok(m):
            log.info("early-accept take %d/%d for %r (mos %.2f)",
                     take + 1, best_of, text[:40], m["mos_min"])
            break
        if take + 1 < best_of and not ranked:
            log.info("retake %d for %r (mos %.2f < %.2f)",
                     take + 1, text[:40], m["mos_min"], threshold)

    # monotony re-roll (ranked mode): if EVERY take so far is flatter than
    # tts.min_f0st, the ranker has no lively read to choose from — roll more.
    # Take-to-take f0 variance is real (~0.4 st on sketch60) so extra samples
    # actually find a livelier take; the exaggeration parameter does NOT reliably
    # move f0 (measured 2026-07-14), so we sample instead of twiddling. The
    # composite ranker still picks the winner (a livelier take must also hold up
    # on MOS/sim), so this only ADDS candidates — it can't force a bad take in.
    reroll_max = int(t.get("f0_reroll_max", 0))
    if ranked and min_f0st > 0 and reroll_max > 0:
        rr = 0
        while rr < reroll_max and max(
                (tk[1].get("f0st", 0.0) for tk in takes), default=0.0) < min_f0st:
            rr += 1
            cand = out.with_name(f"{out.stem}_reroll{rr}.wav")
            synthesize(text, lang, cand, t)
            takes.append((cand, _measure(cand)))
            best_f0 = max(tk[1].get("f0st", 0.0) for tk in takes)
            log.info("monotony re-roll %d/%d for %r (best f0st %.2f, floor %.2f)",
                     rr, reroll_max, text[:40], best_f0, min_f0st)

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
    if ranked and min_f0st > 0:
        # prefer takes that clear the monotony floor — otherwise a flat take can
        # win the composite on MOS alone and the re-roll's lively take is wasted.
        # Relaxed only if none qualify (then rank picks the least-flat via its
        # f0 term). Applied after the MOS gate so a lively pick still holds up.
        lively = [tk for tk in pool if tk[1].get("f0st", 0.0) >= min_f0st]
        pool = lively or pool
    key = (lambda m: _take_rank(m, target_dur)) if ranked \
        else (lambda m: m["mos_min"])
    winner = max(pool, key=lambda tk: key(tk[1]))
    for path, m in takes:
        m["rank"] = round(key(m), 4)
        m["picked"] = path is winner[0]
        if meta is not None:
            meta.append(m)
    # Publish the winner LAST, with a single rename. Path.replace is an atomic
    # POSIX rename within the directory, so `out` — the cache key — goes from
    # "absent" straight to "the complete, ranked winner". There is no window in
    # which it holds a take that has not been through the gates above.
    winner[0].replace(out)
    for path, _ in takes:
        if path is not winner[0]:
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
          "indextts": _synth_indextts, "voxcpm": _synth_voxcpm,
          "qwen": _synth_qwen, "edge": _synth_edge}[engine]
    # per-engine venv isolation (engine_client): when this engine has its own
    # venv (venvs/<engine> or tts.engine_venvs), synthesize through its worker
    # process instead of importing it here — its deps then cannot collide with
    # the incumbent's torch pin. A configured-but-missing venv raises the same
    # FileNotFoundError contract as a missing package (-> engine unavailable).
    # Everything below (normalization, .part atomicity, retries) is engine-
    # agnostic and stays in THIS process; only the raw synth call crosses over.
    from .engine_client import isolated_python, synth as _worker_synth
    worker_py = isolated_python(engine, tts_cfg)
    if worker_py is not None:
        def fn(text, lang, out, t, _py=worker_py):  # noqa: F811 — same signature
            _worker_synth(engine, _py, text, lang, out, t)
    # number/symbol localization is engine-agnostic (digits read wrong-language
    # otherwise) — apply to every engine; manifest/subs/QC keep clean digits.
    text = localize_numbers(text, lang)
    if engine == "chatterbox":
        # acute stress marks exist only here — chatterbox training quirk.
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
