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

import logging
from pathlib import Path

from dubadabidu.core.config.schema import Tts
from dubadabidu.core.manifest import STRESS_TRAINED_MODELS, resolve_engine, resolve_qwen_model
from dubadabidu.core.text_norm import localize_numbers, normalize_for_tts
from dubadabidu.engines.tts import edge, qwen

log = logging.getLogger("dubadabidu.tts")


def engine_available(engine: str, t: Tts | None = None) -> bool:
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
        # BUG (ARCHITECTURE_PLAN.md §10.2 #1), preserved: with qwen_fast UNSET
        # the loader runs stock qwen, yet this probe has always demanded
        # faster_qwen3_tts. Kept until a separate, approved fix.
        if t is None or "qwen_fast" not in t.model_fields_set or t.qwen_fast:
            return importlib.util.find_spec("faster_qwen3_tts") is not None
        return True
    return False                         # unknown engine: assume unavailable


def release_models() -> None:
    """Drop every in-process engine singleton and flush the CUDA cache. The
    bake-off switches engines sequentially and calls this between them;
    without it, models accumulate in VRAM
    (qwen ~4 GB per worker) and a 16 GB card OOMs mid-comparison.
    The next synthesize() through a released engine just reloads it."""
    qwen.release()


def _refuse_marks_on_untrained_model(tts_cfg: Tts, lang: str, engine: str) -> None:
    """Stress marks on a checkpoint not trained on them destroy the audio
    (FINDINGS 2.1b: 97% unusable by ear). Refuse loudly — before any GPU time —
    unless tts.ru_stress_force keeps a refuted experiment reproducible."""
    if not (lang == "ru" and tts_cfg.ru_stress and engine == "qwen"):
        return
    model = resolve_qwen_model(tts_cfg, lang)
    if model in STRESS_TRAINED_MODELS or tts_cfg.ru_stress_force:
        return
    raise RuntimeError(
        f"tts.ru_stress is on but ru resolves to {model}, which was not trained "
        f"on stress marks (FINDINGS 2.1b). Route ru to one that was via "
        f"tts.qwen_model_dir_by_lang, or set tts.ru_stress_force: true to "
        f"reproduce the refuted experiment on purpose.")


def synthesize(text: str, lang: str, out: Path, tts_cfg: Tts,
               retries: int = 2) -> None:
    """Synthesize with retry (autoregressive TTS occasionally glitches; a retry
    with the same input often fixes it — final quality gate is qc/backcheck)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = resolve_engine(tts_cfg, lang)
    fn = {"qwen": qwen._synth_qwen, "edge": edge._synth_edge}[engine]
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
