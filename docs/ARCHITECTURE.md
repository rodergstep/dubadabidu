# Architecture

How the code is laid out and the rules that keep it that way. The plan that
produced this layout, with the reasoning behind each decision, is
[`ARCHITECTURE_PLAN.md`](ARCHITECTURE_PLAN.md).

## Layers

Everything that ships is one package, `src/dubadabidu/`. A module may import
from its own layer and the layers below it, never above:

```
cli          argument parsing, doctor, report, the remote commands
workflows    multi-step flows: preamble, autopilot, course, migrate
lab          experiments: bakeoff, tune, prep, the experiment queue
infra        the remote GPU lifecycle (infra/remote/)
review       human-facing pages and verdict ingestion, refit
stages       s1_extract … s8_mux, one module per stage
qc           metrics, back-transcription, take selection, evaluation
engines      asr, llm (the chat client), tts/ (qwen, edge)
core         config, manifest, languages, project root, ledger paths, logic
```

`lab` and `infra` sit side by side: neither imports the other.

`research/` holds one-off analysis tools. They may import the product;
the product never imports them.

`tests/test_layers.py` builds the import graph from the source, lazy imports
inside functions included, and fails on any upward edge. Its allowlist of
exceptions is empty.

## Where things live

| Path | What |
|---|---|
| `config.yaml` | the base config (project root) |
| `configs/` | provider and profile overlays (`config.gpu.yaml`, `config.deepseek.yaml`, …) |
| `configs/experiments/` | experiment arms (`config.exp.*.yaml`) |
| `ledger/` | the evidence: `FINDINGS.md`, `runs.jsonl`, ratings, comparisons, judge validation, the stress lexicon. Append-only by convention; code reaches it through `core/ledger.py` |
| `specs/` | `experiments.yaml` (the experiment queue), autopilot specs |
| `glossary/<source>/<target>.csv` | translation glossaries |
| `src/dubadabidu/prompts/` | the s3 prompt templates (package data) |
| `pipeline/`, `qc/` (root) | clone-only shims so `python -m pipeline.course` and `python -m qc.blind` keep working; not in the wheel |

## Project root

The CLI works on a project directory (config, glossary, ref/, work/, .env).
It is found in this order: `--project DIR`, then `$DUBADABIDU_HOME`, then the
nearest directory at or above the cwd that holds `config.yaml`, then the
checkout the package was imported from (`core/project.py`). `doctor` prints
which one was used. An installed wheel therefore runs anywhere with
`dubadabidu --project <dir> …`.

`--overlay NAME` looks in the project root, then `configs/`, then
`configs/experiments/`, so overlays are named by file name alone
(`core/config/loader.py`). A file name may exist in only one of those
directories (`tests/core/test_config_layout.py`).

## Config

`core/config/schema.py` is the only place a config default lives. The loader
deep-merges `config.yaml` and the overlays, then validates the merged result
into a frozen, typed `Config`. Every block forbids unknown keys, so a typo
fails at load time with its dotted path. Code reads attributes
(`cfg.tts.best_of`), never dict keys. `derive(block, **updates)` produces a
modified copy, for manifest overrides and bake-off grid points.

`scripts/scan_config_reads.py` lists every raw dict read of a config. The
checked-in inventory (`tests/config/reads_inventory.json`) is empty, and
`tests/config/` fails if a raw read or a default that disagrees with the
schema comes back.

## Languages

`core/languages.py` has one row per language: prompt name, Whisper code, Qwen
name, edge voice, subtitle tag, roles, forced engine, glossary lead-in, stress
marking. Adding a language is one row, plus a glossary file if wanted.

## Remote GPU (`infra/remote/`)

| Module | Holds |
|---|---|
| `provider.py` | the `Provider` protocol: create, get, destroy, list, status, ssh target, price, watchdog |
| `runpod.py` | `RunPodProvider` and the RunPod REST v2 client; the only RunPod-specific module |
| `orchestrator.py` | `provision` and `remote_run`: deadline, reuse, hotfix hold, ledger row, terminate |
| `state.py` | the state file: what must be killed is what it records |
| `transport.py` | ssh and rsync |
| `bootstrap.py` | what runs on the pod: setup script, apt step, task commands, engine installs |
| `diagnostics.py` | `remote smoke`, `setup-check`, `status` |

The provider-neutral modules reach RunPod only through `provider.active()`.
A second provider (`docs/MIGRATION_VAST.md`) is one more class.

## What proves a change is neutral

| Check | Pins |
|---|---|
| `tests/neutrality/` | the local pipeline end to end on a committed 6-second clip, with fakes at the third-party boundaries: manifest, audio and subtitle bytes, recorded LLM requests. The full mode, which runs when the ECAPA weights are in `work/.models/ecapa`, uses the real ECAPA and Distill-MOS models, so QC scores must match bit for bit. `scripts/neutrality_diff.py BASE [--full]` compares two revisions |
| `tests/golden/` | every `synth_hash` cache key across the config matrix, rendered prompts, pod-side commands and payload, CLI parsing and `--help` |
| `tests/infra/test_remote_characterization.py` | eight `remote run` scenarios against fake REST, ssh and clock: every call, state-file version and ledger row |
| `tests/config/` | config round trip, raw-read inventory, drift ledger |
| `tests/test_comment_census.py` | load-bearing comments (dates, FINDINGS references, measurements) survive moves |
