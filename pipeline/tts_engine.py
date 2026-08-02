"""Unified TTS engines. Both s4 (initial synthesis) and s5 (variant retries)
go through synthesize() so caching and device handling live in one place.

Engines:
  qwen       — Qwen3-TTS 12Hz Base (Apache-2.0, git-clone `qwen_tts`). 0.6B/1.7B,
               10 languages incl. all 5 targets, 3 s clone, ~4 GB VRAM. UA ref
               -> x_vector_only_mode (speaker embedding only, no ref transcript).
               The reusable clone prompt is built once per ref and cached.
               tts.qwen_fast routes through faster-qwen3-tts (CUDA graphs) —
               measured 14.0 -> 2.36 s/take, the PRODUCTION default.
  edge       — edge-tts 7.2.8, free MS neural voices, CPU-only, no cloning.

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
from .device import torch_device, require_gpu
from .manifest import resolve_engine
from .text_norm import localize_numbers

log = logging.getLogger("dubadabidu.tts")
_qwen_model = None
_qwen_prompt = None   # (cache_key, prompt_items) — reused across all segments
QWEN_LANGS = {"en": "English", "fr": "French", "de": "German",
              "es": "Spanish", "ru": "Russian"}


def _load_qwen(t: dict):
    global _qwen_model
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
                f"{pkg} not importable. See THIRD_PARTY.md. ({e})")
        # NOT torch_device() — a CPU-only torch in venvs/qwen used to degrade
        # silently to fp32-on-CPU at 126 s/take (vs 2.9 s for voxcpm on the same
        # pod). require_gpu turns that into a loud install error.
        dev = require_gpu("qwen", bool(t.get("allow_cpu_fallback", False)))
        model_id = t.get("qwen_model_dir", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
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
    speaker embedding only, no ref transcript — the path for a Ukrainian ref
    (its transcript can't be tokenized). Full mode needs a target-lang ref +
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
    m = _load_qwen(t)
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
            f"generate_voice_clone does not accept those names ({e})")
    t3 = time.perf_counter()
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


def release_models() -> None:
    """Drop every in-process engine singleton and flush the CUDA cache. The
    bake-off switches engines sequentially and calls this between them;
    without it, models accumulate in VRAM
    (qwen ~4 GB per worker) and a 16 GB card OOMs mid-comparison.
    The next synthesize() through a released engine just reloads it."""
    global _qwen_model, _qwen_prompt
    _qwen_model = _qwen_prompt = None
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
    # normalize_for_tts (acute RU stress marks) was CHATTERBOX-ONLY — a quirk of
    # its training, never validated on any other engine. It left with chatterbox
    # on 2026-08-02. If qwen turns out to mis-stress Russian, that is an A/B to
    # run, not a call to re-apply marks it was never trained on.
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
