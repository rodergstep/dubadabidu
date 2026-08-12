# dubadabidu — Ukrainian → multilingual AI dubbing tool

Batch pipeline: `input/*.mp4` (Ukrainian) → `output/*.mp4` with dubbed audio tracks
(en, fr, de, es, ru) + soft subtitle tracks, cloned to your own voice.

Architecture follows the pattern converged on by pyVideoTrans / Softcatala open-dubbing /
KrillinAI plus the "isometric translation + n-best shorter variants + mild prosodic
stretch" stack from the automatic-dubbing literature.

## Stages (each independently re-runnable, cached via manifest)

```
s1_extract     ffmpeg audio extraction + BS-RoFormer vocal/background separation
               (audio-separator; ~12.9 dB vocal SDR vs htdemucs' ~9 — Demucs
               stays as the fallback backend, separation.backend: demucs)
s2_transcribe  faster-whisper large-v3 (uk) → utterance manifest (EDIT THIS BY HAND!)
s3_translate   LLM translation, ±10% isometric constraint, glossary, per-segment
s4_synthesize  Qwen3-TTS 12Hz Base (tts.qwen_fast), cloned from your reference
               clip via the speaker-embedding-only path (UA ref, no transcript)
s5_fit         place segments on timeline; atempo ≤ 1.12; else request shorter variant
s6_mix         dubbed vocals + original background stem, loudnorm -16 LUFS
s7_subtitles   per-language SRT from fitted segments
s8_mux         one MP4: video copy + N audio tracks + N mov_text subtitle tracks
qc/backcheck   re-transcribe each dubbed track, WER vs. source text (hallucination catch)
qc/similarity  speaker-embedding cosine vs. your reference (clone drift catch)
```

The central data structure is `work/<video>/manifest.json` — one editable JSON per video
(idea borrowed from Softcatala/open-dubbing's utterance_metadata). Fix a mistranslation
there, re-run from s4, and only the changed segments are re-synthesized (content-hash cache).

## Verified versions (checked 2026-08-03; pyproject.toml is the source of truth)

| Component | Version | Notes |
|---|---|---|
| Python | 3.11.x | recommended |
| torch / torchaudio | **2.11.0 / 2.11.0** | installed by the pod bootstrap from the cu128 index, NOT pinned in pyproject. 2.11 is the ceiling: torchaudio stopped there and the two ship version-locked. Moves together with `runpod.allowed_cuda_versions` |
| faster-whisper | 1.2.1 | model `large-v3` |
| whisperx | 3.8.6 | optional, word-level alignment for tighter subs |
| audio-separator | 0.44.5 | BS-RoFormer separation, `.[sep]` extra (s1, LOCAL only). Install WITH the torch pins on one command line — it declares torch>=2.3 with no upper bound |
| demucs | 4.1.0 | fallback separation backend (`htdemucs`), `.[sep]` extra — never installed on a pod |
| faster-qwen3-tts | 0.3.2 | CUDA-graph decode for qwen (GPU); 5.9x measured (THIRD_PARTY.md) |
| qwen_tts | git-clone | Qwen3-TTS bake-off engine (GPU); Apache-2.0, ~4 GB VRAM (THIRD_PARTY.md) |
| edge-tts | 7.2.8 | free fallback voices (no cloning) |
| openai SDK | 2.52.0 | universal client → DeepSeek / OpenAI / LM Studio / Ollama via `base_url` |
| jiwer | latest | WER for QC |
| ffmpeg | 7.x system binary | required in PATH |

## Install (macOS or Linux)

```bash
cd /Users/diadumenoss/Documents/projects/dubadabidu
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sep]"          # tool + deps, console command `dubadabidu`
                                     # add ,mac on Apple Silicon (mlx-whisper)
                                     # drop ,sep on a GPU box — s1 is local-only
brew install ffmpeg                  # macOS; apt install ffmpeg on Linux
dubadabidu doctor                    # validates everything before you start
```

### macOS reality check
Your Mac has no CUDA. For ASR, `pip install .[mac]` pulls in **mlx-whisper**
(Whisper via Apple's Metal/ANE, ~4-5× vs CPU int8) and `asr.backend: auto`
picks it up automatically — makes local transcription of full lessons
tolerable. mlx-whisper has no VAD, but s1 already isolates vocals so that gap
is minor; the rented CUDA box auto-resolves back to faster-whisper + VAD.
Without `.[mac]`, faster-whisper runs CPU (int8) — OK but slow for 1h videos;
(Historical: Chatterbox ran on MPS, validated 2026-07-09 on test60, ~16 it/s —
fine for short clips, slow for full lessons). Recommended split:
**prototype on the Mac with `--engine edge`** (free MS voices, CPU, validates the
whole chain incl. translation/fit/mux), then **run the cloned batch on a rented
CUDA GPU** (RunPod/vast.ai, ~$0.3–0.5/h) — rsync the project folder, run, rsync
`work/` + `output/` back. The manifest/caching design makes this split painless.

## Running on CUDA (real cloned voice)

The Mac path above uses `--engine edge` (no cloning). For your actual cloned voice
you need an NVIDIA GPU: qwen is a CUDA-only install (`require_gpu` refuses a
CPU-only torch rather than degrading to 126 s/take silently), and Whisper runs
~10× faster there. Device selection is **automatic** — `pipeline/device.py`
picks `cuda`+`float16` whenever `torch.cuda.is_available()`, so `config.yaml`
needs no edits (`asr.device: auto`, `tts.engine: qwen` are already correct).
Requirements: NVIDIA 16 GB VRAM (measured peak ~12 GB), CUDA 12.8+ driver.
Stages run sequentially, so ASR and TTS never share VRAM.

Prefer `dubadabidu remote run` / `python -m pipeline.course` over the manual
recipe below — they provision, bootstrap, cap the spend and *always* terminate.

Rent a box (RunPod / vast.ai, ~$0.3–0.5/h; pick a "CUDA 12.x / PyTorch" template).

**1. Send the project up** (include `work/` so cached translations aren't redone,
`ref/` for your voice, `input/` for the video). **Exclude `.env`** — a rented pod
must never hold your RunPod full-access key; the only key the pod needs is
`TRANSLATE_API_KEY`, exported directly in step 3:

```bash
rsync -avz --exclude .venv --exclude __pycache__ --exclude .env \
  ./ user@GPU_HOST:~/dubadabidu/
```

**2. Install on the GPU box.** torch comes from the cu128 index, not plain PyPI:
PyPI serves a `+cu130` build that needs a 13.0 driver, and the CUDA filter also
admits 12.8/12.9 hosts. Pinning the LOWEST admitted version makes every host in
the pool valid. This mirrors `runpod_infra.REMOTE_SETUP` — keep them in sync.

```bash
sudo apt install -y --no-install-recommends ffmpeg   # --no-install-recommends
                                                     # or apt swaps the C runtime
                                                     # and kills sshd mid-run
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip                            # the old resolver backtracks
                                                     # for ~26 min on this graph
pip install torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"                              # no .[sep]: s1 stays local
# qwen: git-clone engine, pinned by SHA — see runpod.engine_setup in config.gpu.yaml
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
dubadabidu doctor                       # should report 'torch device: cuda'
```

**3. Run.** Translations are cached in `work/<video>/manifest.json`, so start at s4 to
skip the LLM entirely (no LM Studio / API key needed on the GPU box):

```bash
dubadabidu run input/sketch60/sketch60.mp4 --from s4_synthesize   # cloned voice, all langs
```

If you instead run from scratch (`run ...` without `--from`), stage s3 still validates
`TRANSLATE_API_KEY` even when cached — either `export TRANSLATE_API_KEY=x` (any value)
or run LM Studio / a remote provider on that box.

**4. Bring results back:**

```bash
rsync -avz user@GPU_HOST:~/dubadabidu/work/   ./work/
rsync -avz user@GPU_HOST:~/dubadabidu/output/ ./output/
```

## Old install section (reference)

```bash
# system
sudo apt install ffmpeg
# env
uv venv --python 3.11 && source .venv/bin/activate
# torch on the SAME line as the separation extra: audio-separator declares
# torch>=2.3 with no upper bound and will otherwise pull 2.13, breaking the
# torchaudio pairing (torchaudio stopped at 2.11).
uv pip install -e '.[dev,sep]' torch==2.11.0 torchaudio==2.11.0
```

GPU: NVIDIA 8–12 GB VRAM (CUDA 12.x). Stages run sequentially so ASR and TTS
never share VRAM.

## Reference voice

Put 2–4 clean 15–20 s WAV clips of your voice (no music, typical teaching tone) in
`glossary/../ref/` — actually anywhere; set `tts.reference_wav` in `config.yaml`.
The Ukrainian reference is cloned through Qwen3-TTS's `x_vector_only_mode`
(speaker embedding only — a Ukrainian transcript cannot be tokenized for a
target-language generation). `tts.reference_max_s` trims it: cloning quality
climbs to ~15 s and then degrades, and an over-long reference is the documented
trigger for Qwen's never-emits-EOS failure.
(`tts.cfg_weight` survives at 0.0 only to keep old cache hashes valid — it was a
Chatterbox knob and no current engine reads it.)

## Translation backend (local or remote)

Stage `s3_translate` talks to any **OpenAI-compatible** chat endpoint through the
`openai` SDK, so the provider is a pure config choice in `config.yaml` under
`translation:` — no code changes. In all cases set the key env var first:

```bash
export TRANSLATE_API_KEY=...   # real key for remote; any non-empty string for local
```

| Provider | `base_url` | `model` | `response_format` | `TRANSLATE_API_KEY` |
|---|---|---|---|---|
| **DeepSeek (remote, DEFAULT)** | `https://api.deepseek.com` | `deepseek-v4-flash` | `json_object` | real key |
| OpenAI (remote) | `https://api.openai.com/v1` | `gpt-5-mini` | `json_object` | real key |
| LM Studio (local) | `http://127.0.0.1:1234/v1` | e.g. `google/gemma-4-e4b` | `json_schema` | any non-empty string |
| Ollama (local) | `http://localhost:11434/v1` | e.g. `qwen3:32b` | `json_object` | any non-empty string |

DeepSeek is the default: every downstream stage inherits translation quality,
it costs <$1/video with automatic context caching, and small local models were
the weakest link of the whole chain. Local endpoints remain a pure config swap
(restore a `base_url`/`model`/`response_format` triple above) for offline work.
Optional `translation.reflect_model` runs only the reflect (critic) pass of the
3-pass scheme on a stronger model — the cheapest known quality lever.

**`response_format`** — providers diverge on how they enforce JSON output:
`json_object` (DeepSeek/OpenAI/Ollama), `json_schema` (LM Studio — strict schema,
best for small local models), or `text` (no server enforcement; ```json fences
are stripped automatically). Using the wrong one yields errors like
`'response_format.type' must be 'json_schema' or 'text'`.

### LM Studio setup
1. Developer tab (`>_`) → load `google/gemma-4-e4b` → **Start Server**. Default port
   is `1234` (pin it in Developer > Settings if you don't want it to change); use
   the exact model id from `GET /v1/models` as `model`.
2. If you enabled *"Require API Key"* in the server settings, `export
   TRANSLATE_API_KEY=<that-key>`; otherwise any non-empty value works, e.g.
   `export TRANSLATE_API_KEY=lm-studio`.
3. Set `response_format: json_schema` in `config.yaml` (LM Studio rejects
   `json_object`).

Small local models can still drop or mangle segments; if you hit
`LLM dropped segments … re-run s3`, lower `translation.batch_size` (e.g. 20 → 5).

For local endpoints (base_url on 127.0.0.1/localhost) the full-transcript
context in the system prompt is automatically capped at
`translation.max_context_chars` (default 12000) so long lessons don't
silently overflow a small model's context window. Remote providers always
get the full transcript — on DeepSeek its context caching makes that
effectively free after the first batch.

## Run

```bash
dubadabidu run input/lesson01.mp4                 # all stages, all languages
dubadabidu run input/*.mp4 --langs en,de          # subset
dubadabidu run input/a.mp4 --to s7_subtitles      # stop before mux (audio+subs only)
dubadabidu stage s3_translate input/lesson01.mp4  # single stage re-run
dubadabidu qc input/lesson01.mp4                  # back-transcription WER + similarity report
dubadabidu bakeoff input/a.mp4 --langs en         # TTS engine head-to-head (Phase C)
dubadabidu autopilot input/a.mp4                  # spec-driven accept/fix loop
dubadabidu verdicts input/a.mp4 ratings_a_en_x.json  # ingest review-page verdicts
```

**GPU on RunPod (M2 — automated lifecycle):** `dubadabidu remote <task> <video>`
provisions a spot pod, runs the task, syncs results back, and ALWAYS terminates
(hard budget cap + independent pod-side self-destruct watchdog). The source
video never uploads — s1/s2 run locally, the pod works from `work/` audio stems,
and the mux runs locally after sync-back. Requires `RUNPOD_API_KEY` in `.env` and
your SSH key registered with RunPod.

```bash
dubadabidu remote smoke                            # ~$0.02 lifecycle check
dubadabidu remote setup-check                      # ~$0.30: dry-run the bake-off
                                                   #   installs, report which
                                                   #   engines import (do this
                                                   #   BEFORE a real bakeoff)
dubadabidu remote run  input/a.mp4 --langs fr --budget 10   # translate+synth on GPU
dubadabidu remote bakeoff input/a.mp4 --langs en   # engine bake-off on GPU
dubadabidu remote autopilot input/a.mp4            # accept/fix loop on GPU
dubadabidu remote status                           # list account pods (leak check)
dubadabidu remote kill                             # terminate any tracked pod
```

Recommended workflow per video:
1. `dubadabidu preamble input/lesson01.mp4` — runs s1+s2+prep (extracts 2–3 ref
   candidates from the video's own vocals), then pauses. **Hand-review the
   Ukrainian text in the manifest** (art terminology — one fix here propagates
   to all languages).
2. `dubadabidu preamble input/lesson01.mp4` again — translates the tune language,
   runs tune-lite R1 over this video's refs, and stores the winning ref in the
   manifest's `tts_overrides` (s4/s5/evaluate pick it up automatically; no
   config.yaml edit per video).
3. `run --from s3_translate` the rest. 4. `qc`. 5. Fix flagged segments in the
   manifest, re-`run` from s4.

## Licensing notes

- Qwen3-TTS (the only cloning engine since 2026-08-02): Apache-2.0 — OK for your
  commercial courses, and no watermark. (Chatterbox, which used to sit here, was
  MIT but stamped every take with an inaudible PerTh watermark.)
- `pedalboard` (s6 mastering) is GPLv3, kept OPT-IN as the `.[master]` extra: it
  only processes audio, so the WAV it emits is not a derivative, but the default
  install stays permissive.
- Do NOT ship XTTS-v2 output commercially (Coqui CPML is non-commercial), and keep
  Fish-Speech weights (CC-BY-NC) out; neither is wired in here, on purpose.

## New in v0.4 (quality + flywheel release)

- **Soft-anchor timeline (s5/s6/s7)**: dubs may drift up to `fit.drift_max_s`
  after their source start, eating pause time instead of being stretched or
  shortened; drift resets at real source pauses. Mix overlap is impossible by
  construction (`drift_exceeded` replaces the old `overrun_s` flag); subtitles
  follow the placed dub timing. Replaces `fit.borrow_gap_s`.
- **Composite take ranking (s4/s5)**: with `tts.rank_takes` (default on), all
  `best_of` takes are scored on windowed MOS + f0 liveliness + ECAPA speaker
  similarity and the composite-best wins — take selection now sees monotony
  and clone drift, not just naturalness. Per-take metrics land in the
  manifest (`tr.takes`).
- **Verdict flywheel (AUTOPILOT M3)**: review pages grew accept/reject
  buttons; `dubadabidu verdicts <video> <exported.json>` writes
  `human_rating`/`human_verdict` into the manifest and accumulates rows in
  `ratings_<lang>.json` for the periodic qc-weight re-fit. The autopilot
  never re-rolls an accepted segment and always re-rolls a rejected one;
  autopilot re-rolls of WER-flagged segments back-transcribe every fresh
  take and veto hallucinations.
- **Number-safe backcheck WER**: digits are expanded to words (num2words) on
  both sides before WER, so "25" vs "twenty-five" no longer false-flags
  measurement-heavy segments. Whisper is a singleton across QC rounds.
- s3: TTS pace measurement is filtered by engine; transcript context is
  capped for local (LM Studio/Ollama) endpoints. Polish dropped from config.

## New in v0.2.0 (real-tool release)

- console command `dubadabidu` (pyproject packaging, `pip install -e .`)
- `doctor` — environment validation with actionable hints
- `report` — per-language fit/QC table; `qc` now prints it too
- device auto-detection (cuda -> mps -> cpu) + faster-whisper CPU/int8 fallback
- unified TTS engine module with retries; `--engine edge` flag for free CPU runs
- LLM translation: retry with exponential backoff, env-key validation,
  drop-detection, per-batch checkpointing, stage progress in manifest
- s6 frame-rate normalization bugfix (24k segments onto 44.1k timeline)
- unit tests (10) for segment merging, fit ladder, cache hashing: `pytest`
- IMPROVEMENT_PLAN.md — prioritized roadmap
