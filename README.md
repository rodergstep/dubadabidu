# dubadabidu — Ukrainian/Russian → multilingual AI dubbing

Turns a Ukrainian or Russian lesson video (`source_language: uk | ru`; stack
`--overlay config.source-ru.yaml` for Russian) into one MP4 with dubbed audio
tracks in English, French, German, Spanish, Russian and Ukrainian (never the
source itself), plus soft subtitles, in **your own cloned voice**. Ukrainian is
the exception: Qwen3-TTS has no Ukrainian, so uk is voiced by edge-tts
`uk-UA-OstapNeural` (`tts.engine_by_lang`, FINDINGS §2.4).

`input/lesson01.mp4` → `output/lesson01_multi.mp4` (video copy + N audio
tracks + N subtitle tracks), and optionally one MP4 per language.

Voice-over content with loose sync: dubs may drift a little against the source
instead of being squeezed (`fit.drift_max_s`). Russian ships with a known
lexical-stress defect (~29% of takes carry one wrong stress); everything tried
against it is recorded in `ledger/FINDINGS.md` (CLOSED table).

## How it runs

The Mac drives everything. Only synthesis needs an NVIDIA GPU, so that part
runs on a rented RunPod pod that is created, used and **always terminated** by
`dubadabidu remote run`. The source video never leaves your machine.

```
s1_extract     ffmpeg audio + BS-RoFormer vocal/background separation  local
               (demucs fallback: separation.backend: demucs)
s2_transcribe  Whisper large-v3 (mlx-whisper on Apple Silicon,           local
               faster-whisper elsewhere) → utterance manifest
s3_translate   LLM translation (OpenAI-compatible endpoint), ±10%       local
               isometric constraint, glossary, 3-pass draft/reflect/adapt
s4_synthesize  Qwen3-TTS 1.7B Base, cloned from your reference clip      GPU pod
               (speaker-embedding path), best-of-N takes
s5_fit         soft-anchor timeline; rubberband stretch ≤ 1.12          GPU pod
               (atempo fallback), else a shorter translation variant
s6_mix         per-segment level match + mastering + ducked background, local
               two-pass loudnorm to -16 LUFS
s7_subtitles   per-language SRT from the placed dub timing               local
s8_mux         one MP4: video copy + audio tracks + mov_text subtitles   local
qc             back-transcription WER (hallucination catch) + speaker
               similarity (clone-drift catch)
```

The central data structure is `work/<video>/manifest.json`, one editable JSON
per video. Fix a transcript or translation there and re-run from s4: only the
changed segments are re-synthesized (content-hash cache).

## Requirements

- Apple-Silicon Mac (the driver) or Linux x86_64. Python 3.11 or 3.12.
- [uv](https://docs.astral.sh/uv/) and `ffmpeg` 7.x on `PATH`. An ffmpeg built
  with librubberband gives cleaner time-stretching; without it s5 uses atempo
  (`doctor` tells you which).
- A translation API key (DeepSeek by default; any OpenAI-compatible endpoint,
  including a local LM Studio or Ollama, works).
- For cloned voices: a RunPod account and API key, plus an SSH key pair. A
  pod needs an NVIDIA GPU with ≥ 16 GB VRAM (measured peak ~12 GB) and a
  CUDA ≥ 13.0 driver; `config.gpu.yaml` picks the cards.
- 2–4 clean 15–20 s WAV clips of your voice (see "Reference voice").

dubadabidu runs **from a clone of this repository**. Config, prompts, glossary,
`.env`, `work/` and the ledgers are resolved relative to the checkout, so a
wheel installed elsewhere only supports `dubadabidu --help`.

## Quickstart

### 1. Install

```bash
git clone https://github.com/rodergstep/dubadabidu && cd dubadabidu
```

```bash
uv sync --extra dev --extra sep --extra mac --extra master
```

```bash
brew install ffmpeg
```

```bash
cp .env.example .env
```

Fill in `TRANSLATE_API_KEY` (and `RUNPOD_API_KEY` for remote runs) in `.env`.
The extras: `sep` = RoFormer/demucs separation, `mac` = mlx-whisper,
`master` = the s6 mastering chain (pedalboard, GPLv3, on by default in
`config.yaml`; without it set `mix.master.enabled: false`). On Linux drop
`--extra mac` and use `apt install --no-install-recommends ffmpeg`.

```bash
uv run dubadabidu doctor
```

`doctor` checks ffmpeg, devices, imports, the reference clip, the translation
key and the models the endpoint serves, and (with `RUNPOD_API_KEY`) the SSH key.

### 2. Local run — plumbing check, no GPU, no cloning

Put a video at `input/lesson01.mp4`, then:

```bash
uv run dubadabidu run input/lesson01.mp4 --engine edge --langs en
```

This runs every stage with free Microsoft Edge voices. The output is named
`output/lesson01_multi_EDGE-PLUMBING-ONLY.mp4` on purpose: it validates the
chain (separation, transcription, translation, fit, mix, mux), not the voice.

### 3. Remote run — your cloned voice on a RunPod GPU

Create the SSH key the pod is reached with (its `.pub` is sent to the pod):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod
```

Transcribe locally, then **hand-review the source text** (`text_src`) in
`work/lesson01/manifest.json`; one fix there propagates to every language:

```bash
uv run dubadabidu run input/lesson01.mp4 --to s2_transcribe
```

Translate locally, synthesize on a pod, then mix and mux locally:

```bash
uv run dubadabidu remote run input/lesson01.mp4 --overlay config.gpu.yaml --budget 5
```

`--budget` is a hard dollar cap: the pod is terminated when it is reached, and a
pod-side watchdog self-destructs independently. Several videos in one command
share one pod. Check that nothing is left running:

```bash
uv run dubadabidu remote status
```

Then score the result and fix what it flags:

```bash
uv run dubadabidu qc input/lesson01.mp4
```

Edit flagged segments in the manifest and re-run (`--from s4_synthesize`).
`dubadabidu evaluate` scores every segment, `dubadabidu review` then builds
worst-first review pages (`work/<video>/review_<lang>.html`), and
`dubadabidu verdicts <video> <export.json>` writes your accept/reject calls back.

## Commands

```
dubadabidu doctor                          validate the environment
dubadabidu run <video...>                  all stages (--from/--to, --langs, --engine edge, --force)
dubadabidu stage <stage> <video...>        one stage
dubadabidu preamble <video>                s1+s2+reference candidates, then a pause for review
dubadabidu qc | report | evaluate | review <video>
dubadabidu verdicts <video> <export.json>  ingest review-page verdicts
dubadabidu batch [video...]                per-video × per-language matrix
dubadabidu autopilot <video...>            spec-driven accept/fix loop (docs/AUTOPILOT.md)
dubadabidu refit                           propose qc weights from your ratings (never applies)
dubadabidu migrate [video...] [--apply]    rewrite pre-text_src work files (dry run by default)
dubadabidu bakeoff | tune | prep <video>   engine / reference experiments
dubadabidu remote status | kill | smoke | setup-check
dubadabidu remote run | bakeoff | autopilot <video...>
python -m pipeline.course ...              many videos, one pod (from a clone)
python -m pipeline.experiments ...         queue experiments (specs/experiments.yaml) on one pod
```

Every command works on a project directory (`config.yaml`, `glossary/`, `ref/`,
`work/`, `.env`): `--project DIR`, else `$DUBADABIDU_HOME`, else the nearest
directory above the current one that holds `config.yaml`. `doctor` prints which
one it used. `python -m dubadabidu` is the same CLI.

`remote` flags: `--budget USD`, `--keep-alive` (leave the pod up to debug over
SSH; the watchdog still fires), `--reuse` (attach to that pod).

## Configuration

`config.yaml` is the base; overlays are deep-merged over it, left to right.
`--overlay NAME` looks for NAME in the project root, then `configs/`, then
`configs/experiments/`, so overlays are named by file name alone:

| File | What it sets |
|---|---|
| `config.yaml` | everything; defaults for a Mac |
| `configs/config.gpu.yaml` | the pod profile: qwen + fast path, takes, RunPod GPU list, image, budget, runtime cap |
| `configs/config.deepseek.yaml` | translation via DeepSeek (also forwarded to every pod run) |
| `configs/config.gemini.yaml` | translation via Google Gemini's OpenAI-compatible endpoint. Its model id is **unverified**: run `doctor` before using it |
| `configs/experiments/config.exp.*.yaml` | experiment arms; the record behind FINDINGS.md |

```bash
uv run dubadabidu run input/lesson01.mp4 --overlay config.gpu.yaml --overlay config.gemini.yaml
```

Every key is validated against `src/dubadabidu/core/config/schema.py`, the
only place a default lives, so a typo fails
with its dotted path instead of being silently ignored.

### Translation backend

s3 talks to any OpenAI-compatible chat endpoint; the provider is a config
choice under `translation:`. The key is read from the variable named by
`translation.api_key_env` (`TRANSLATE_API_KEY`).

| Provider | `base_url` | `model` | `response_format` | key |
|---|---|---|---|---|
| **DeepSeek (default)** | `https://api.deepseek.com` | `deepseek-flash` | `json_object` | real key |
| OpenAI | `https://api.openai.com/v1` | e.g. `gpt-5-mini` | `json_object` | real key |
| LM Studio (local) | `http://127.0.0.1:1234/v1` | e.g. `google/gemma-4-e4b` | `json_schema` | any non-empty string |
| Ollama (local) | `http://localhost:11434/v1` | e.g. `qwen3:32b` | `json_object` | any non-empty string |

`translation.reflect_model` runs only the critic pass of the 3-pass scheme on a
stronger model. Small local models can drop segments; if you see
`LLM dropped segments … re-run s3`, lower `translation.batch_size`. For local
endpoints the transcript context is capped at `translation.max_context_chars`.
A wrong `response_format` produces errors like
`'response_format.type' must be 'json_schema' or 'text'`.

### Reference voice

Record 2–4 clean 15–20 s WAV clips (quiet room, no music, natural teaching
tone; `ref/README.txt` has the checklist), put them in `ref/` (git-ignored) and
set `tts.reference_wav`. The reference is cloned through Qwen3-TTS's
speaker-embedding path (`qwen_x_vector_only`). `tts.reference_max_s` trims it:
quality climbs to ~15 s and then degrades. The reference is a curated asset;
per-video references measured worse (FINDINGS §2.2).

## Cost

Measured on RunPod, from `ledger/runs.jsonl`: about **$1.10–1.22 per hour of video for
all 5 languages** (~$0.22 per language), plus a fixed bootstrap of a few
minutes per pod. RAM and vCPU are bundled with the GPU and not billed
separately. Translation with DeepSeek costs well under $1 per video. Re-derive
the number from your own runs instead of quoting it:
`python -c "from dubadabidu.core.runlog import cost_per_video_hour as c; print(c())"`.
Always cap a run with `--budget` and check `remote status` afterwards.

## Development

```bash
uv run pytest -q
```

```bash
uvx ruff check .
```

The code is one package, `src/dubadabidu/`, in layers (cli → workflows →
lab | infra → review → stages → qc → engines → core); `tests/` mirrors them and
`tests/test_layers.py` rejects an upward import. `docs/ARCHITECTURE.md` has the
map, the config rules and the checks that prove a refactor changes no output.

To use a built wheel instead of the clone, install it into any environment
and point it at a project directory:

```bash
uv build && uv pip install dist/dubadabidu-1.0.0-py3-none-any.whl
```

```bash
dubadabidu --project ~/dubbing doctor
```

After changing a dependency in `pyproject.toml`, run `uv lock` and
`scripts/export_requirements.sh` (`requirements.txt` is the pod's generated
Linux export, `+cu130` torch; CI checks it is current).

Pinned versions that matter, and why, are commented in `pyproject.toml`
(`transformers==4.57.3`, `av<19`, torch 2.14.1 from the cu130 index on Linux,
torchaudio 2.11.1). Glossaries live in `glossary/<source>/<target>.csv`. Measured results and closed questions are in `ledger/FINDINGS.md`; read
its CLOSED table before spending a pod.

## Roadmap

Open work, in order. Everything else from the earlier planning docs is done
or closed by evidence (`ledger/FINDINGS.md`, CLOSED table).

**Next**
- One full ~1 h lesson end-to-end on the pod, all 5 languages
  (`python -m pipeline.course`, `--reuse --keep-alive`), wall clock and $
  recorded in `ledger/runs.jsonl`. Re-test `tts.best_of: 2` on it: how often is the
  shipped take marked unusable?
- The 20-video batch (~15 min/video of review on worst-first pages), then
  publish per-language audio (YouTube multi-language audio; `per_language`
  MP4s as the fallback).
- Point `runpod.image` at the GHCR tag once `.github/workflows/image.yml` has
  pushed one, then `remote setup-check` (never trust an image tag).
- Widen `specs/batch.yaml` `policy.langs` from `[en]` to all five (a human edit).

**Planned**
- Vast.ai as the second provider — docs/MIGRATION_VAST.md.

**Deferred, each with its trigger**
- Enhancement / bandwidth pass — 24 kHz dullness audible through real speakers.
- RVC timbre training — voice-fidelity complaints return at scale.
- WhisperX re-cut of subtitles to dubbed timing — subtitle-timing complaints.
- Gradio UI / multi-GPU s4 — a productization decision.
- Cross-model ASR consensus (catches silent omissions) and f0/wer scoring on
  GPU — FINDINGS §5.
- `pyright --basic` in CI on `src/`.

Comments tagged `PLAN_v0.5 Pn` or `IMPROVEMENT_PLAN Phase X` refer to planning
docs retired in 1.0.0: `git show d7ffdc6:PLAN_v0.5.md` /
`git show d7ffdc6:IMPROVEMENT_PLAN.md`.

## License

dubadabidu is licensed under the **GNU Affero General Public License v3.0 or
later** (AGPL-3.0-or-later); see [`LICENSE`](LICENSE). If you run a modified
version as a network service, you must offer its source to that service's users.

The licence covers this code only. Model weights and third-party engines keep
their own licences, and the audio this pipeline produces is not covered by the
AGPL. See `THIRD_PARTY.md` for the engine stack and its obligations.

- Qwen3-TTS (the cloning engine): Apache-2.0, no watermark.
- `pedalboard` (s6 mastering) is GPLv3 and kept opt-in (`.[master]`); it only
  processes audio, so its output is not a derivative.
- XTTS-v2 (Coqui CPML, non-commercial) and Fish-Speech weights (CC-BY-NC) are
  deliberately not wired in.

Release history: `CHANGELOG.md`.
