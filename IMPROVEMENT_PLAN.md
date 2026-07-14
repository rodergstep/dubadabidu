# dubadabidu — GPU-Phase Plan (RunPod + DeepSeek)

Status (2026-07-08, v0.3.x): the local pipeline is validated end-to-end on
sketch60 for **en** (HeyGen-parity by blind metrics AND by ear) and **ru**
(stress solved via RUAccent). Everything the ear complained about became a fix:
video-donated refs, flow segmentation, best-of-N with windowed-MOS gating,
trim/fade/linear-loudnorm mix, 3-pass context-aware translation. Polish is
DROPPED. Remaining targets: en, fr, de, es, ru.

The next phase moves to RunPod + DeepSeek. Guiding rule carried over:
**the eval harness decides, not model reputation** — every candidate model runs
the same tune/evaluate/review loop and needs to beat the incumbent on sim_raw +
MOS + the ear before adoption.

## Phase A — GPU environment (first pod session, ~1 h)

- [ ] Pod: RTX 4090 (24 GB) is ample — Chatterbox ~7 GB, CosyVoice3-0.5B needs
      only ~2–4 GB, whisper large-v3 fp16 ~5 GB; they load sequentially anyway.
      Spot/community instances are fine (everything is resumable + cached).
- [ ] Env: python 3.11 venv from `requirements.txt` (torchcodec is already
      pinned — the ta.save trap is solved). `whisper_device()` auto-selects
      cuda/float16; no code changes expected.
- [ ] rsync recipe: project + `ref/` up; `work/` + `output/` down. Content-hash
      caching makes re-syncs incremental for free.
- [ ] Config deltas for GPU (`config.gpu.yaml` overlay or direct edits):
      `tts.best_of: 5` (takes are cheap now), translation → DeepSeek.

## Phase B — Translation: DeepSeek (do this FIRST — it propagates everywhere)

- [ ] Model: **`deepseek-v4-flash`** — NOT `deepseek-chat`/`deepseek-reasoner`,
      which are deprecated 2026-07-24. $0.14/M in, $0.28/M out; 1M context.
      `translation.response_format: json_object` (v4 supports json_object; keep
      json_schema only for LM Studio).
- [ ] Context caching is automatic and cache hits cost ~2% of normal input.
      Our prompt design (big shared system prompt = glossary + terms + full
      transcript, small per-batch user payload) is exactly the shape that
      benefits — a 1 h lecture's transcript context is effectively free after
      batch 1.
- [ ] Keep `passes: 3`. Optional A/B once: draft/adapt on v4-flash + reflect
      (critic) on v4 thinking mode — a smarter critic is the cheapest known
      quality lever. Judge on the review page, not vibes.
- [ ] Cost envelope: ~10k words source/video × 5 langs × 3 passes ≈ well under
      $1/video. Not a constraint; do not optimize further.
- [ ] Re-run en+ru sketch60 translations on DeepSeek before any TTS bake-off,
      so TTS comparisons sit on final-quality text.

## Phase C — TTS bake-off: Chatterbox v3 vs CosyVoice 3 (per language)

- [ ] Add a `cosyvoice` engine to `pipeline/tts_engine.py` behind the same
      `synthesize()` interface. CosyVoice3 zero-shot cloning needs reference
      audio **plus its transcript** — we have both for free (video-donated ref
      slices have known `text_uk` in the manifest). Install is git-clone (not
      PyPI): pin the repo commit in a `third_party/` note. Apache-2.0 ✓.
      Languages: en/de/es/fr/ru all covered (no pl — dropped anyway).
- [ ] `tts.engine_by_lang` config override (Phase-2 roadmap item, now real):
      e.g. `{fr: cosyvoice, de: cosyvoice, en: chatterbox, ru: chatterbox}` —
      whatever the bake-off says.
- [ ] Protocol per language: tune-lite (refs already known → params grid only,
      subset 5) on BOTH engines → evaluate (sim_raw + MOS + f0st + windowed
      min) → review page for the ear. CosyVoice3 notes: ru stress behavior must
      be re-tested (RUAccent marks may or may not help it — A/B again);
      Chatterbox keeps the acute-mark path.
- [ ] Also worth one cheap look while at it: **Chatterbox-Turbo** for en
      (license must be verified MIT before any use — blog benchmark claims are
      Resemble-adjacent, trust only our own harness).
- [ ] **VoxCPM2** added as a 4th lane (2026-07-13): pip-installable (pins in
      THIRD_PARTY.md), Apache-2.0, all 5 targets in one engine, 48 kHz out,
      ~8 GB VRAM, top published SIM scores. UA ref → audio-only cloning path
      (its transcript modes need one of its 30 languages; uk isn't one). Same
      rule applies: the harness decides, not the model card.
- [ ] **Qwen3-TTS** added as a 5th lane (2026-07-13): git-clone (THIRD_PARTY.md),
      Apache-2.0, 10 languages incl. all 5 targets, 3 s clone, ~4 GB VRAM (the
      lightest challenger — cheapest to run). UA ref → x_vector_only (embedding
      only, no ref transcript). Full bake-off roster is now chatterbox +
      cosyvoice + voxcpm + qwen (+ indextts for en/zh). The harness decides.
- [ ] Decision gate: an engine wins a language only if it beats the incumbent
      on BOTH raw similarity and MOS, or ties metrics and wins the ear test.
      French additionally needs the native-speaker listen (no self-QC there).

## Phase D — Scale validation (one full video before the batch)

- [ ] Run one complete ~1 h lesson end-to-end on the pod, all 5 languages.
      Watch: s3 batching with full-transcript context (fine on 1M context),
      manifest size, s6's 1 h pydub timeline in RAM (~600 MB — acceptable;
      chunked ffmpeg concat is the fallback if not), wall-clock + $ totals.
- [ ] Automate the per-video preamble (currently manual): extract 2–3 ref
      candidates from the video's own demucs vocals (longest clean utterance
      spans), tune-lite R1 picks one, human skims `text_uk` + `terms_*.json`.
      This is the last scripting task worth doing — it runs 20 times.
- [ ] batch_report.md (Phase-1 leftover): per-video × per-lang matrix of
      means (score/sim/mos) + flagged-segment count + links to review pages.

## Phase E — The 20-video batch

- [ ] Frozen recipe per video: preamble → s1–s8 all langs → evaluate → qc
      (backcheck WER now cheap on GPU — run it for every language) → review
      pages → human spot-fix loop (edit manifest → surgical re-run) → final mux
      + per-language .m4a for YouTube multi-audio.
- [ ] Human budget: ~15 min/video review using worst-first pages; native French
      pass on video 1 only, then spot checks.

## Deferred — each has an explicit trigger, none built preemptively

- ~~Number/abbreviation expansion in `text_norm.py`~~ **DONE 2026-07-13**:
  `localize_numbers` expands digits + %/° to target-language words for all 5
  langs (engine-agnostic, salted into synth_hash via `NUM_VERSIONS`). Letter
  units (ml/cm) deliberately left as abbreviations — language-correct inflection
  (esp. ru case) is error-prone; extend `_SYMBOLS`/add a unit map if the ear
  flags them.
- Enhancement/bandwidth pass (Resemble Enhance or similar; verify license) —
  trigger: 24 kHz dullness audible on full-video listen through real speakers.
  Note CosyVoice3 outputs 24 kHz too — this is engine-independent.
- RVC timbre training (30–60 min of clean speech) — trigger: voice-fidelity
  complaints return at scale on whichever engine wins.
- WhisperX re-cut of subtitles to dubbed audio timing — trigger: subtitle
  timing complaints from real viewers.
- Gradio UI / Dockerfile / multi-GPU s4 — trigger: productization decision.

## Standing risks

- `deepseek-chat` name dies 2026-07-24 — config must say `deepseek-v4-flash`.
- chatterbox-tts 0.1.7 PyPI vs HF v3 weights drift — pin HF revision if a
  re-download ever changes output hashes.
- CosyVoice is a git-clone dependency — pin the commit; its requirements may
  conflict with ours (install in the same venv ONLY after `pip check`).
- License hygiene: XTTS-v2 (CPML) and Fish-Speech weights (CC-BY-NC) stay out.
  Verify Chatterbox-Turbo's license before first use.
