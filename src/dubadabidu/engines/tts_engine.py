# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unified TTS engines. Both s4 (initial synthesis) and s5 (variant retries)
go through synthesize() so caching and device handling live in one place.

Engines:
  qwen       — Qwen3-TTS 12Hz Base (Apache-2.0, git-clone `qwen_tts`). 0.6B/1.7B,
               10 languages incl. all 5 targets, 3 s clone, ~4 GB VRAM. UA ref
               -> x_vector_only_mode (speaker embedding only, no ref transcript).
               The reusable clone prompt is built once per ref and cached.
               tts.qwen_fast routes through faster-qwen3-tts (CUDA graphs) —
               measured 14.0 -> 2.36 s/take, the PRODUCTION default.
  edge       — edge-tts 7.3.1, free MS neural voices, CPU-only, no cloning.

REMOVED 2026-07-31 (git history has them all): cosyvoice never produced audio
in seven attempts; voxcpm and indextts lost to qwen+fast once its kernel-launch
bottleneck was fixed. indextts also took its emotion_from_source machinery with
it — that was disentangled-emotion prompting, an IndexTTS-2-only capability.
Restoring any of them means reverting that commit, not rewriting the adapter.

qwen is a GPU-only git-clone install — see THIRD_PARTY.md. It fails with an
actionable FileNotFoundError (which synthesize() re-raises without retry) when
its package isn't importable.

Per-language routing: tts.engine_by_lang maps a language to an engine,
falling back to tts.engine (manifest.resolve_engine). The winning assignment
is decided by the bake-off (qc/bakeoff.py), not by reputation.
"""
from __future__ import annotations
import asyncio, logging, subprocess, time
from pathlib import Path
from dubadabidu.core.device import require_gpu
from dubadabidu.core.manifest import STRESS_TRAINED_MODELS, resolve_engine, resolve_qwen_model
from dubadabidu.core.text_norm import localize_numbers, normalize_for_tts

log = logging.getLogger("dubadabidu.tts")
_qwen_model = None
_qwen_model_id: str | None = None   # which checkpoint _qwen_model holds
_qwen_prompt = None   # (cache_key, prompt_items) — reused across all segments
_declared_sr: dict[str, int | None] = {}   # checkpoint -> its declared output rate


def declared_output_sr(model_id: str) -> int | None:
    """The output rate a Qwen3-TTS checkpoint DECLARES
    (speech_tokenizer/config.json output_sample_rate), or None if unreadable.

    faster-qwen3-tts logs "Could not infer sample rate from base model;
    defaulting to 24000 Hz" and returns 24000 whatever the checkpoint says.
    A wrong rate does not fail anything: the take is written with the wrong
    header and plays at the wrong pitch and speed. 1.7B-Base declares 24000
    (checked 2026-11-24); this keeps a future checkpoint honest."""
    if model_id in _declared_sr:
        return _declared_sr[model_id]
    import json
    sr: int | None = None
    try:
        local = Path(model_id) / "speech_tokenizer" / "config.json"
        if local.exists():
            cfg_path = local
        else:
            from huggingface_hub import hf_hub_download
            cfg_path = Path(hf_hub_download(model_id,
                                            "speech_tokenizer/config.json"))
        v = json.loads(cfg_path.read_text(encoding="utf-8")).get(
            "output_sample_rate")
        sr = int(v) if v else None
    except Exception as e:                                   # noqa: BLE001
        log.warning("could not read %s's declared sample rate (%s) — "
                    "sample-rate guard skipped", model_id, e)
    _declared_sr[model_id] = sr
    return sr


def check_output_sr(model_id: str, sr: int) -> None:
    want = declared_output_sr(model_id)
    if want and int(sr) != want:
        raise RuntimeError(
            f"qwen returned audio at {sr} Hz but {model_id} declares "
            f"{want} Hz — the take would play at the wrong pitch and speed")
# Qwen3-TTS language names. uk is absent because the model does not list
# Ukrainian ("Auto" uk was unusable by ear), so config routes uk to edge.
QWEN_LANGS = {"en": "English", "fr": "French", "de": "German",
              "es": "Spanish", "ru": "Russian"}


def _load_qwen(t: dict, lang: str | None = None):
    """The qwen model for `lang`'s checkpoint (tts.qwen_model_dir_by_lang).
    Keyed on the checkpoint: a run whose languages resolve to DIFFERENT
    checkpoints (ru -> the stress fine-tune) releases and reloads instead of
    silently synthesizing every language with whichever loaded first."""
    global _qwen_model, _qwen_model_id
    want = resolve_qwen_model(t, lang) if lang else (
        t.get("qwen_model_dir") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    if _qwen_model is not None and _qwen_model_id != want:
        log.info("qwen checkpoint %s -> %s: reloading", _qwen_model_id, want)
        release_models()
    if _qwen_model is None:
        # tts.qwen_fast -> faster-qwen3-tts (MIT), a drop-in that wraps the same
        # weights in CUDA Graphs + StaticCache. Qwen3-TTS is kernel-LAUNCH bound,
        # not compute bound: each decode step dispatches ~500 tiny GPU ops from a
        # Python loop and the card idles at 10-12% utilisation between launches.
        # That is why our probe found 0.6B and 1.7B equally slow (1.38 vs 1.42
        # wall/audio) and dtype/attention irrelevant — none of those touch
        # dispatch. Upstream measures 4.1x on a 4090.
        # NO silent fallback to the stock class: an opt-in that quietly does
        # nothing is how qwen ran on CPU for a week.
        fast = bool(t.get("qwen_fast", False))
        try:
            import torch
            if fast:
                from faster_qwen3_tts import FasterQwen3TTS as Qwen3TTSModel  # type: ignore
            else:
                from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as e:
            pkg = ("faster-qwen3-tts (tts.qwen_fast is on) — pip install "
                   "faster-qwen3-tts" if fast else
                   "qwen_tts — git clone https://github.com/QwenLM/Qwen3-TTS "
                   "(pin the commit), pip install -e .")
            raise FileNotFoundError(
                f"{pkg} not importable. See THIRD_PARTY.md. ({e})") from e
        # NOT torch_device() — a CPU-only torch in venvs/qwen used to degrade
        # silently to fp32-on-CPU at 126 s/take (vs 2.9 s for voxcpm on the same
        # pod). require_gpu turns that into a loud install error.
        dev = require_gpu("qwen", bool(t.get("allow_cpu_fallback", False)))
        model_id = want
        if fast:
            # faster-qwen3-tts documents from_pretrained(model_id) only — it
            # manages device/dtype itself, and an unknown kwarg would fail the
            # load. Do NOT add device_map/dtype here without checking upstream.
            kw = {}
        else:
            kw = {"device_map": "cuda:0" if dev == "cuda" else dev,
                  "dtype": torch.bfloat16 if dev == "cuda" else torch.float32}
            # flash-attn is a heavy CUDA-only build; opt-in (qwen_flash_attn) so
            # a missing wheel can't fail the load on a fresh pod. Note it is NOT
            # a fix for the speed problem: dispatch overhead dominates, which is
            # why eager vs sdpa measured only 1.57 vs 1.42 wall/audio.
            if dev == "cuda" and t.get("qwen_flash_attn", False):
                kw["attn_implementation"] = "flash_attention_2"
        log.info("loading Qwen3-TTS from %s on %s (fast=%s) ...",
                 model_id, dev, fast)
        _qwen_model = Qwen3TTSModel.from_pretrained(model_id, **kw)
        _qwen_model_id = model_id
        if fast:
            # captures the CUDA graphs up front. Without it the first calls pay
            # capture cost lazily, which is how a benchmark reads 1.49 wall/audio
            # and a warmed one reads 0.73 for the same setup.
            t0 = time.perf_counter()
            _qwen_model.warmup(prefill_len=int(t.get("qwen_warmup_len", 100)))
            log.info("qwen warmup (CUDA graph capture): %.1fs",
                     time.perf_counter() - t0)
    return _qwen_model


def trimmed_ref(ref: str, max_s: float | None) -> str:
    """Reference clipped to max_s, cached next to the original. Returns `ref`
    unchanged when no trim is configured or the clip is already short enough.

    Qwen3-TTS cloning quality scales with reference length from ~3 s to ~15 s,
    then PLATEAUS AND DEGRADES, and over-long references are the documented
    trigger for its worst failure mode (the decoder never emits EOS and loops).
    Every reference we own is 15-18 s and the configured one is the longest at
    18.1 s, i.e. all of them sit in the degrading zone.

    soundfile, not ffmpeg, on purpose: an engine venv on a bake-off pod has
    soundfile (the install recipe adds it) but NOT ffmpeg — installing ffmpeg
    there upgrades the C runtime and kills sshd. Trimming from the START keeps
    the speech onset; these refs are continuous speech, so the tail is what goes."""
    if not max_s:
        return ref
    import soundfile as sf
    src = Path(ref)
    info = sf.info(str(src))
    if info.duration <= max_s:
        return ref
    out = src.with_name(f"{src.stem}.trim{int(max_s)}s{src.suffix}")
    if not out.exists():
        data, sr = sf.read(str(src))
        sf.write(str(out), data[:int(max_s * sr)], sr)
        log.info("reference trimmed %.1fs -> %.1fs: %s",
                 info.duration, max_s, out.name)
    return str(out)


def _qwen_clone_prompt(m, t: dict):
    """Reusable clone prompt for the current reference, built once and cached
    (a full video is hundreds of segments off ONE ref). x_vector_only_mode:
    speaker embedding only, no ref transcript — the path for a ref in the
    source language (a Ukrainian transcript can't be tokenized). Full mode needs a target-lang ref +
    reference_text and is opt-in via qwen_x_vector_only: false."""
    global _qwen_prompt
    ref = trimmed_ref(t["reference_wav"], t.get("reference_max_s"))
    ref_text = t.get("reference_text", "")
    x_only = bool(t.get("qwen_x_vector_only", True)) or not ref_text
    key = (ref, x_only, ref_text if not x_only else "")
    if _qwen_prompt is None or _qwen_prompt[0] != key:
        kw = {"ref_audio": ref, "x_vector_only_mode": x_only}
        if not x_only:
            kw["ref_text"] = ref_text
        # faster-qwen3-tts is a wrapper: the stock model (and this method) sit
        # at .model. The stock class has no .model, so getattr picks itself.
        inner = getattr(m, "model", m)
        _qwen_prompt = (key, inner.create_voice_clone_prompt(**kw))
    return _qwen_prompt[1]


def _synth_qwen(text: str, lang: str, out: Path, t: dict) -> None:
    """Qwen3-TTS Base voice clone. Covers all 5 targets; UA ref cloned via the
    speaker-embedding-only path (see _qwen_clone_prompt). Returns (wavs, sr)."""
    import soundfile as sf
    ref = t["reference_wav"]
    if not Path(ref).exists():
        raise FileNotFoundError(f"reference_wav not found: {ref}")
    # attribute the wall-clock: load (once) / prompt (once per ref) / generate
    # (every call). The bake-off's s_take is a median over takes, so only
    # `gen` recurs — but the split is what says whether a slow run is a slow
    # model or a cold start.
    t0 = time.perf_counter()
    m = _load_qwen(t, lang)
    t1 = time.perf_counter()
    prompt = _qwen_clone_prompt(m, t)
    t2 = time.perf_counter()
    # tts.qwen_gen_kwargs — sampling passthrough, empty by default so the
    # shipped generation config (temperature 0.9 / top_k 50 / top_p 1.0 /
    # repetition_penalty 1.05, identical across all five checkpoints) is what
    # runs unless deliberately overridden. Worth overriding because take-to-take
    # VARIANCE is the dominant quality factor we have measured (mos± 0.24-0.49,
    # and best_of was still climbing at k=6): a lower temperature trades peak
    # quality for consistency, which could buy back the takes it costs.
    # An unsupported kwarg fails the call outright, so the error names the knob
    # rather than surfacing as a bare TypeError from inside the library.
    gen = dict(t.get("qwen_gen_kwargs") or {})
    try:
        wavs, sr = m.generate_voice_clone(
            text=text, language=QWEN_LANGS.get(lang, "Auto"),
            voice_clone_prompt=prompt, **gen)
    except TypeError as e:
        if not gen:
            raise
        raise RuntimeError(
            f"qwen rejected tts.qwen_gen_kwargs={gen} — this build's "
            f"generate_voice_clone does not accept those names ({e})") from e
    t3 = time.perf_counter()
    check_output_sr(_qwen_model_id or "", sr)
    dur = len(wavs[0]) / float(sr)
    # RUNAWAY GUARD. Qwen3-TTS's most common failure is the decoder never
    # emitting EOS: it fills its token budget with laughing/humming/babble.
    # there is no per-call timeout (a first call may legitimately spend minutes
    # downloading weights), so a hang is otherwise bounded only by the pod
    # watchdog — expensive, and a merely-long take is
    # worse: it is silently SHIPPED. Raising makes it a failed take that
    # synth_best_of re-rolls. Not passed as max_new_tokens: that kwarg is
    # undocumented for both implementations and an unknown kwarg would fail
    # the call outright.
    cap = float(t.get("qwen_max_audio_s", 60.0))
    if cap and dur > cap:
        raise RuntimeError(
            f"qwen produced {dur:.1f}s of audio for {len(text)} chars "
            f"(cap {cap:.0f}s) — runaway generation, take rejected. Raise "
            f"tts.qwen_max_audio_s if a segment is legitimately this long.")
    sf.write(str(out), wavs[0], sr)
    # RTF = generate seconds per second of audio produced. Device-independent,
    # so it compares across engines and pods where raw s/take cannot.
    log.info("qwen timing: load=%.2fs prompt=%.2fs gen=%.2fs audio=%.2fs "
             "RTF=%.2f chars=%d", t1 - t0, t2 - t1, t3 - t2, dur,
             (t3 - t2) / dur if dur else float("nan"), len(text))


def engine_available(engine: str, t: dict | None = None) -> bool:
    """Can `engine` synthesize HERE? Import check only — no weights, no CUDA.

    Exists so s5 can say "run this on a pod" instead of dying on a bare
    ModuleNotFoundError. Kept cheap and side-effect free: importlib.util
    .find_spec does not execute the module, so calling this on a laptop with no
    engine costs nothing and cannot half-initialise anything."""
    import importlib.util
    if engine == "edge":
        return True                      # network service, runs anywhere
    if engine == "qwen":
        if importlib.util.find_spec("qwen_tts") is None:
            return False
        # qwen_fast is a different package and the one that actually runs
        if (t or {}).get("qwen_fast", True):
            return importlib.util.find_spec("faster_qwen3_tts") is not None
        return True
    return False                         # unknown engine: assume unavailable


def release_models() -> None:
    """Drop every in-process engine singleton and flush the CUDA cache. The
    bake-off switches engines sequentially and calls this between them;
    without it, models accumulate in VRAM
    (qwen ~4 GB per worker) and a 16 GB card OOMs mid-comparison.
    The next synthesize() through a released engine just reloads it."""
    global _qwen_model, _qwen_prompt, _qwen_model_id
    _qwen_model = _qwen_prompt = _qwen_model_id = None
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


def _refuse_marks_on_untrained_model(tts_cfg: dict, lang: str, engine: str) -> None:
    """Stress marks on a checkpoint not trained on them destroy the audio
    (FINDINGS 2.1b: 97% unusable by ear). Refuse loudly — before any GPU time —
    unless tts.ru_stress_force keeps a refuted experiment reproducible."""
    if not (lang == "ru" and tts_cfg.get("ru_stress") and engine == "qwen"):
        return
    model = resolve_qwen_model(tts_cfg, lang)
    if model in STRESS_TRAINED_MODELS or tts_cfg.get("ru_stress_force"):
        return
    raise RuntimeError(
        f"tts.ru_stress is on but ru resolves to {model}, which was not trained "
        f"on stress marks (FINDINGS 2.1b). Route ru to one that was via "
        f"tts.qwen_model_dir_by_lang, or set tts.ru_stress_force: true to "
        f"reproduce the refuted experiment on purpose.")


def synthesize(text: str, lang: str, out: Path, tts_cfg: dict,
               retries: int = 2) -> None:
    """Synthesize with retry (autoregressive TTS occasionally glitches; a retry
    with the same input often fixes it — final quality gate is qc/backcheck)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = resolve_engine(tts_cfg, lang)
    fn = {"qwen": _synth_qwen, "edge": _synth_edge}[engine]
    # ONE VENV since 2026-08-02. Per-engine venvs existed because four engines
    # had colliding pins (chatterbox hard-pinned torch 2.6.0 + numpy<2, voxcpm
    # wanted its own torch, cosyvoice needed setuptools<80 and a .pth hack), so
    # isolation made the collisions structurally impossible. Three engines are
    # gone and chatterbox took its pin with it. The constraint sets now agree:
    # faster-qwen3-tts wants torch>=2.5.1 (base has 2.6.0), qwen-tts pins
    # transformers==4.57.3 which NOTHING in the qc stack requires, and
    # huggingface-hub<1.0 satisfies speechbrain>=0.8 and faster-whisper>=0.21.
    # The cost of isolation was a SECOND ~2.5 GB torch download on every fresh
    # pod — the single largest item in the bootstrap.
    # What this gives up: crash isolation (a segfaulting engine now takes the
    # run with it, not just its worker). Acceptable at one engine; if a second
    # cloning engine ever returns, revert this commit rather than reinventing it.
    # number/symbol localization is engine-agnostic (digits read wrong-language
    # otherwise) — apply to every engine; manifest/subs/QC keep clean digits.
    text = localize_numbers(text, lang)
    # RU lexical stress, RESTORED 2026-08-03 and no longer engine-gated. It was
    # deleted with chatterbox because its call site read `engine == "chatterbox"`,
    # and that commit wrote down the risk: "Whether qwen mis-stresses Russian is
    # UNTESTED — and a ru track is about to ship." It shipped; the listener
    # reported wrong stress. Off by default (tts.ru_stress) because chatterbox
    # was TRAINED on combining acutes and qwen was not — so this is a hypothesis
    # to A/B, not known-good behaviour being reinstated.
    text = normalize_for_tts(text, lang, tts_cfg)
    _refuse_marks_on_untrained_model(tts_cfg, lang, engine)
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
