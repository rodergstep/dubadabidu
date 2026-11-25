# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 on.

Measured results and refuted ideas live in `ledger/FINDINGS.md`; per-run costs
live in `ledger/runs.jsonl`. This file records what changed in the software.

## [Unreleased]

### Added
- Russian input: `source_language: uk | ru` (the schema rejects anything else)
  and the `config.source-ru.yaml` overlay. The translation prompts, the terms
  extraction, the shortening step and the adequacy judge name the configured
  source instead of assuming Ukrainian.
- Ukrainian as a target language: voiced by edge-tts `uk-UA-OstapNeural` via
  `tts.engine_by_lang: {uk: edge}`. qwen's Ukrainian (language `Auto`; Qwen3-TTS
  lists no Ukrainian) was unusable by ear (FINDINGS §2.4).
- A video is never dubbed into its own language.
- `dubadabidu migrate [video...] [--apply]` rewrites manifests, `refs.json` and
  `terms_*.json` that still use `text_uk` / `uk`. Dry run by default; never
  touches `ledger/`. `doctor` mentions files still waiting for it.
- `--project DIR` and `$DUBADABIDU_HOME` choose the project directory; without
  them it is the nearest directory above the cwd that holds `config.yaml`, so
  an installed wheel works outside the clone. `doctor` prints which was used.
  `python -m dubadabidu` runs the CLI.
- `docs/ARCHITECTURE.md`: the layers, where things live, and the checks that
  prove a refactor changes no output.

### Changed
- torch 2.11.0 → 2.14.1 and torchaudio 2.11.0 → 2.11.1. torchaudio is no longer
  version-locked to torch, but its Linux wheel is a CUDA 13 build, so Linux
  torch now comes from the cu130 index and `runpod.min_cuda_version` rises from
  12.8 to 13.0. ECAPA and Distill-MOS scores are bit-identical to 2.11 on CPU;
  CUDA graphs (faster-qwen3-tts) and the CUDA qc metrics were checked on a pod.
  The lock moves only torch, torchaudio, torchvision, triton and the CUDA
  runtime wheels (nvidia-*-cu12 → their CUDA 13 counterparts).
- Restructured into one package, `src/dubadabidu/`, in layers (cli, workflows,
  lab, infra, review, stages, qc, engines, core); `tests/test_layers.py`
  rejects upward imports. No output changes: manifests, audio, subtitles, QC
  scores, cache keys, prompts and pod-side commands are byte-identical, checked
  by the neutrality harness and golden tests added for the purpose. `dub.py`
  is now `dubadabidu.cli`; the research tools moved to `research/`.
- Files moved; every reference was updated, and `--overlay` still takes a bare
  file name: overlays to `configs/` and `configs/experiments/`; the evidence
  ledger (`FINDINGS.md`, `runs.jsonl`, ratings, comparisons, judge validation,
  stress lexicon) to `ledger/`; `experiments.yaml` to `specs/`; `AUTOPILOT.md`
  and `MIGRATION_VAST.md` to `docs/`; the prompts into the package.
- Config is a typed, frozen object read by attribute; the schema is the only
  place a default lives (each code default that disagreed was resolved as
  recorded in `docs/ARCHITECTURE_PLAN.md` §3.3). The RunPod defaults moved into the schema, so
  `remote status`/`kill` and doctor read the same values without the gpu
  overlay. A partial `runpod:` block in an overlay now validates.
- The remote code is split by concern under `infra/remote/`, with RunPod behind
  a `Provider` protocol; the lifecycle is pinned by eight recorded scenarios.
- One language table (`core/languages.py`) replaces four per-language tables.
- uk → edge is a property of the language table, no longer a line in
  `config.yaml`, so an overlay cannot drop it by replacing
  `tts.engine_by_lang`. An explicit `engine_by_lang` entry still wins, and a
  bake-off still measures the engine it names on every language.
- The manifest transcript field is `text_src` (was `text_uk`). Manifests,
  `refs.json` and cached `terms_<lang>.json` written before are renamed on
  read, so existing work and ratings still load.
- Glossaries moved to `glossary/<source>/<target>.csv`; the ASR initial prompt
  reads only the source language's glossary.
- Edge output keeps the `_EDGE-PLUMBING-ONLY` name and the review banner only
  when edge stands in for the configured engine; a language routed to edge on
  purpose (uk) is the real dub. Verdict ingestion still refuses every edge
  language.

## [1.0.0] — 2026-11-25

Release cleanup. No pipeline logic, default config values or model choices
changed; the only locked-version changes are PyAV 18.1.0 → 18.1.1 and edge-tts
7.2.8 → 7.3.1 (below).

### Added
- `doctor` warns, without failing, when `pedalboard` is missing while
  `mix.master.enabled` is on (s6 would stop) and when ffmpeg has no
  `rubberband` filter (s5 falls back to atempo).
- `CHANGELOG.md`; a Roadmap section in the README.
- `SPDX-License-Identifier: AGPL-3.0-or-later` header on every Python file.
- Type annotations on every public function (no `Any`).

### Changed
- `AUTOPILOT.md` trimmed to the current design.
- `config.gemini.yaml`: the "hybrid" block now describes the existing
  `translation.reflect_model`; the Gemini model id is marked unverified.
- `qc.stress_mfa` requires `DUB_MFA_SCRATCH`; `qc.stress_detect` and
  `qc.stress_mfa` take the compare export path as an argument;
  `qc.axis_check` / `qc.stress_var` find the repo relative to themselves.
  (Research tools only; they previously had one machine's paths baked in.)
- The s8 EDGE-fallback warning goes through logging; the pedalboard install
  hint says `uv sync --extra master`.
- Config schema: `translation.avoid_mis_stressed` declares the default the code
  actually uses (`true`); the docstring no longer claims stages receive
  `model_dump()`.

### Removed
- `IMPROVEMENT_PLAN.md`, `PLAN_v0.5.md` (open items moved to the README
  Roadmap; `git show d7ffdc6:<file>` for the originals).
- `qc/stress_wer.py` (no entry point or importer; `git show d7ffdc6:qc/stress_wer.py`).
- Dead code: `text_norm.marked_words`, `stress_consensus.status_quo`, and tune's
  unreachable chatterbox-era R2 parameter sweep (`tune.rounds: [R2]` is still
  accepted and logged as skipped).
- `FIXES.md` is no longer tracked: it is the autopilot's runtime log.

### Fixed
- `av<19` is now also a `[project]` dependency, so an install from the wheel
  carries it: `[tool.uv]` constraints do not travel in package metadata, and an
  unlocked install resolved PyAV 19.0.2 on Linux/Python 3.12 (the 0.5.0 ASR
  crash).
- PyAV locked at 18.1.1: 18.1.0 was yanked upstream (double free when decoding
  audio with custom channel layouts). 18.1.1 is newer than the global
  `exclude-newer` date, so `[tool.uv] exclude-newer-package` exempts av alone.
- edge-tts 7.3.1: the service now answers 7.2.8 with HTTP 403 for every voice,
  which broke `--engine edge` and the s8 EDGE fallback.
- Stale comments: chatterbox/CosyVoice references in `device.py`, `prep.py`,
  `tune.py`, `s3_translate.py`; the Python range in the CI workflow; FINDINGS §5
  item 1 marked closed, duplicate paragraph removed.

## [0.5.0] — 2026-11-24

Review release. `pyproject.toml` still said 0.4.0 at the time.

### Added
- `uv.lock` pins the whole stack (torch/torchaudio 2.11.0, `+cu128` on Linux);
  `requirements.txt` is generated from it; ruff and `uv lock --check` in CI.
- Typed config (`pipeline/config_schema.py`, pydantic, `extra="forbid"`) and
  manifest TypedDicts.
- Prebuilt pod image (`docker/Dockerfile`, `.github/workflows/image.yml`), used
  only when its lock stamp matches.
- `doctor` checks that the translation endpoint still serves the configured
  models; s3 records the model that answered.
- `remote run` over several videos shares one pod and one pod-side process.
- Per-language qwen checkpoint (`tts.qwen_model_dir_by_lang`) and the ru-stress
  fine-tune A/B arms (`experiments.yaml` `ru-ft-*`), refuted (FINDINGS §2.1r).
- `qc/judge.py`: Audiobox Aesthetics validated against 72 rated groups,
  refuted (FINDINGS §2.1s).

### Changed
- RunPod client moved to REST v2 (v1 retired 2026-11-15).
- Translation model `deepseek-flash` (`deepseek-v4-flash` retired).
- Licensed under AGPL-3.0-or-later.

### Fixed
- `remote run` works end to end as a Mac → RunPod → Mac loop.
- `av<19` constraint: PyAV 19 broke faster-whisper on the pod's Python 3.12.

## [0.4.x] — 2026-07-10 to 2026-08-13

Unreleased series (version stayed 0.4.0). Highlights, oldest first:

### Added
- Soft-anchor timeline (s5/s6/s7): dubs may drift up to `fit.drift_max_s`.
- Composite take ranking (windowed MOS + f0 + speaker similarity), verdict
  flywheel (`dubadabidu verdicts`, `ratings_<lang>.json`), `dubadabidu refit`.
- RunPod lifecycle automation (`dubadabidu remote`): provision → run → always
  terminate, budget cap, pod-side watchdog, `--reuse` / `--keep-alive`.
- Stackable `--overlay` configs and provider overlays.
- Engine bake-off (`dubadabidu bakeoff`), tune-lite, listening, blind-rating
  and same-sentence comparison pages.
- `python -m pipeline.course` (many videos, one pod) and
  `python -m pipeline.experiments` (queue experiments on one pod).
- qwen fast path (`tts.qwen_fast`, CUDA graphs), reference trim and runaway guard.
- BS-RoFormer separation (demucs kept as the fallback, `.[sep]` extra);
  mlx-whisper ASR on Apple Silicon (`.[mac]`); s6 mastering chain (`.[master]`).
- `FINDINGS.md` + `runs.jsonl` knowledge ledger; CI with a config/stage audit.

### Changed
- Engine roster cut to qwen (Qwen3-TTS 1.7B Base) for all five languages;
  per-engine venvs merged into one.
- Python 3.12 pod image, torch 2.11, numpy 2.

### Removed
- chatterbox, cosyvoice, voxcpm, indextts engines; the 0.6B qwen variant.
- Russian stress marks in production, reduction respelling, and lexical
  avoidance (each refuted, FINDINGS CLOSED table).

### Fixed
- Many correctness fixes found by audit: production routed to the wrong engine,
  a config key that never existed, every dub track encoded at 96 kHz, six audio
  tracks flagged default, Whisper repetition loops, WER normalisation of numbers
  and ё, and remote-run bugs (shell quoting, port mapping, budget units).

## [0.3.0] — 2026-07-09

Initial commit: Ukrainian → multilingual dubbing pipeline (s1–s8 + QC), plus
autopilot M1 (spec-driven loop, batch report, `.env` contract).

## [0.2.0] — before the repository

Console command `dubadabidu`, `doctor`, `report`, device auto-detection, unified
TTS module with retries, `--engine edge`, LLM translation with retry and
checkpointing, first unit tests.

[1.0.0]: https://github.com/rodergstep/dubadabidu/compare/d7ffdc6...v1.0.0
