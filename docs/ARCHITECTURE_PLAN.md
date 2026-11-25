# ARCHITECTURE_PLAN: restructuring dubadabidu without changing what it produces

> **Status (2026-11-25):** executed, with the answers to Q1–Q17 as
> recommended. Not done, and why: F2 (uk→edge routing, edge voices and
> subtitle tags from the language table): the bake-off clears
> `engine_by_lang` on purpose to test any engine on any language, and a
> registry-forced engine would silently override that, so it needs its own
> decision. G3–G5's renames (SshEndpoint, RunRecord) and the terminate
> retry-and-verify change: the split landed without them, and the last one
> changes behaviour. H3 (bakeoff split, optional) and H4 (refit split, deferred
> by recommendation). The current layout is described in `ARCHITECTURE.md`;
> paths below are the ones at the time of writing.

Phase 1 (design only). Written 2026-11-25 against `main` @ `5108227` (Release
1.0.0) and the open branch `chore/torch-2.14.1` @ `2c31393`. **No file was
moved or edited**; this document is the only new file.

Baseline on `main`: `pytest -q`: **472 passed** (1 pydub warning).
`uvx ruff@0.17.5 check .`: **clean**.

How the facts below were obtained (all reproducible, nothing guessed):

- **Import graph:** an AST walk over `dub.py`, `pipeline/*.py` and `qc/*.py`.
  It records every `import` and marks the ones inside a function as lazy.
- **Config reads:** a second AST walk that resolves aliases
  (`tts = cfg["tts"]`, `tcfg`, `rp = {**DEFAULTS, **cfg["runpod"]}`, …) to
  dotted schema paths. It then compares each inline default with the default
  pydantic declares, by importing `pipeline/config_schema.py`.
- **References:** `git grep` for every root file that would move.
- **Branch status:** `git rev-list` and `git diff main...chore/torch-2.14.1`.

The scanners become tests in Phase 2 (§8, commits A4 and A5), so the numbers in
this plan stay checkable.

---

## 0. Summary

| Problem from the brief | Verified? | What I found |
|---|---|---|
| Root is a dumping ground | **Yes** | 42 tracked top-level files. There are **17** `config.exp.*.yaml`, not 18. The branch adds `config.source-ru.yaml`, which makes 43. |
| `dub.py` god-script, `os.chdir` | **Yes** | 697 lines. It imports 7 `qc` modules at top level and `qc.metrics` lazily. `os.chdir(ROOT)` is at line 517, and `.env` is read from `ROOT`. |
| Config validated then discarded | **Yes** | `_validate_config(cfg)` at `dub.py:538`; the return value is dropped. My scan finds about 270 raw reads in 26 modules (`dub.py` 71, `runpod_infra` 58, `s3_translate` 41, …). |
| `qwen_fast` drift | **Yes** | `tts_engine.py:104` defaults it to `False` (the loader); `tts_engine.py:280` defaults it to `True` (`engine_available`, called by s5). |
| Inverted layering | **Partly** | Only **two** of the edges are true inversions: the engine module importing `qc` (take selection lives in `tts_engine`), and `tts_engine → s6_mix._clean`. `s3 → qc.stress_words`, `prep`, `tune` and `autopilot → qc` look inverted only because packages are split by *name* (`pipeline` vs `qc`) and not by *layer*. Under the layering in §2 they point downward and are fine. |
| Oversized modules | **Yes** | `runpod_infra` 1,525 · `qc/bakeoff` 812 · `qc/refit` 771 · `tts_engine` 671 · `dub.py` 697. |
| Hard-coded source language | **Mostly fixed on the branch** | `chore/torch-2.14.1` already renames `text_uk` to `text_src` (with a read shim), adds `{SRC}` to the prompts, `glossary/<src>/` and `source_language: Literal["uk","ru"]`. Still left: **five** per-language tables in five places (§4). |
| Tests mirror the mess | **Yes** | `test_runpod.py` is 1,020 lines with 62 tests. Six test files compute `ROOT = Path(__file__).parents[1]`, and about 18 depend on `cwd == repo root`. |

**The open branch decides the sequencing.** `chore/torch-2.14.1` is 8 commits
ahead of `main` and 0 behind, so it fast-forwards. It touches 41 files: almost
every module this plan moves, the configs, the prompts and the tests. The plan
therefore **starts after that branch is merged** (decision Q1). Every commit
below is written against the post-merge tree. Once the branch is merged, nothing
conflicts with it.

---

## 1. Current-state map

### 1.1 Packages and responsibilities

| Module | Lines | Responsibility | Runs on |
|---|---:|---|---|
| `dub.py` | 697 | argparse tree, `doctor`, `report`, `preamble`, `_pick_reference`, the remote run plan (`_remote_plan` / `_remote_run_batch`), the run loop, `.env` loading, config load, merge and validate | Mac + pod (the pod runs `dubadabidu run …`) |
| `pipeline/s1_extract` | 157 | ffmpeg + RoFormer/demucs separation | Mac |
| `pipeline/s2_transcribe` | 133 | Whisper → manifest utterances | Mac |
| `pipeline/asr` | 166 | faster-whisper / mlx-whisper backends | Mac (s2); pod (via `backcheck`) |
| `pipeline/s3_translate` | 517 | LLM client, prompts, terms, 3-pass translation, adequacy judge, `shorten` | Mac; pod (s5's `llm_shorten`, remote `autopilot`) |
| `pipeline/s4_synthesize` | 133 | best-of-N synthesis per segment | **pod** |
| `pipeline/s5_fit` | 213 | soft-anchor timeline, stretch, shorter variants | **pod** |
| `pipeline/s6_mix` | 268 | segment normalisation, ducking, mastering, loudnorm | Mac (needs pedalboard) |
| `pipeline/s7_subtitles` | 79 | SRT | Mac |
| `pipeline/s8_mux` | 117 | MP4 mux | Mac (needs the source video) |
| `pipeline/tts_engine` | 671 | qwen loader and adapter, edge adapter, ref trimming, **take ranking, early-accept, best-of selection, QC scoring** | **pod** |
| `pipeline/manifest` | 369 | manifest IO, TypedDicts, `synth_hash`, engine/model resolution, staleness | everywhere |
| `pipeline/text_norm` | 219 | number localisation, ru stress normalisation (salts hashes) | everywhere |
| `pipeline/logic` | 139 | `deep_merge`, segment merge/split (pure) | everywhere |
| `pipeline/device` | 85 | torch / whisper device choice | everywhere |
| `pipeline/config_schema` | 284 | pydantic schema, validation only | everywhere |
| `pipeline/runpod_infra` | 1,525 | REST v2 client, payload, provisioning loop, state file, ssh/rsync, watchdog, `REMOTE_SETUP`/`apt_setup`/`REMOTE_TASK`, engine install, `remote_run` lifecycle, smoke/setup-check/status | Mac |
| `pipeline/runlog` | 86 | `runs.jsonl` ledger | Mac |
| `pipeline/autopilot` | 333 | spec-driven accept/fix loop | Mac; pod (`remote autopilot`) |
| `pipeline/prep` / `tune` | 213 / 252 | reference extraction / reference sweep | Mac; pod (`remote preamble`) |
| `pipeline/course` / `experiments` | 171 / 132 | many videos or experiments on one pod (shells out to `python dub.py`) | Mac |
| `qc/metrics` | 221 | ECAPA, Distill-MOS, f0 (**bit-identity critical**) | everywhere |
| `qc/backcheck` | 167 | back-transcription WER | pod + Mac |
| `qc/evaluate` | 168 | per-segment `qc_score` | pod + Mac |
| `qc/review_page`, `verdicts`, `refit`, `batch_report` | 258 / 243 / 771 / 102 | verdict flywheel (production CLI) | Mac |
| `qc/stress_words` | 189 | lexicon IO, avoid list, word marking (production: s3, review, verdicts) | Mac |
| `qc/blind`, `compare`, `listen`, `tune_page` | 261 / 371 / 94 / 119 | `python -m qc.*` rating pages that write `ratings_*` and `comparisons_*` (inputs to `refit`) | Mac |
| `qc/bakeoff` | 812 | engine/variant A/B harness (`dubadabidu bakeoff`, `remote bakeoff`) | pod |
| `qc/stress_mfa`, `stress_detect`, `stress_var`, `stress_variants`, `stress_consensus`, `axis_check`, `avoidance_ab`, `judge` | 290 / 294 / 180 / 108 / 104 / 129 / 150 / 164 | **research only**; most carry REFUTED verdicts (FINDINGS §2.1g/h/i/q/s) | Mac |

### 1.2 The real import graph (generated; `*` = lazy, inside a function)

```
dub            → pipeline.{s1..s8, prep, tune, autopilot, manifest, device, logic}
                 qc.{backcheck, bakeoff, batch_report, evaluate, refit, review_page, verdicts}
                 *pipeline.{asr, config_schema, runpod_infra}   *qc.metrics
pipeline.asr            → device
pipeline.autopilot      → manifest, s1..s8        *qc.{backcheck, batch_report, evaluate}
pipeline.manifest       → text_norm
pipeline.prep           → manifest                *qc.metrics
pipeline.runpod_infra   → manifest                *runlog  *s8_mux
pipeline.s1/s6/s7/s8    → manifest
pipeline.s2_transcribe  → asr, logic, manifest
pipeline.s3_translate   → manifest                *qc.stress_words
pipeline.s4_synthesize  → manifest, tts_engine
pipeline.s5_fit         → logic, manifest, s3_translate, tts_engine
pipeline.tts_engine     → device, manifest, text_norm   *s6_mix  *qc.metrics  *qc.backcheck
pipeline.tune           → manifest, tts_engine    *qc.evaluate  *qc.metrics
pipeline.{config_schema, course, device, experiments, logic, runlog, text_norm} → (none internal)
qc.backcheck    → pipeline.{device, manifest, text_norm}
qc.bakeoff      → pipeline.manifest   *pipeline.{tts_engine, tune}  *qc.{backcheck, evaluate, metrics, review_page}
qc.batch_report → pipeline.manifest
qc.blind        → *pipeline.tts_engine  *qc.{evaluate, metrics}
qc.compare      → *qc.{evaluate, metrics}
qc.evaluate     → pipeline.manifest, qc.metrics   *pipeline.tts_engine
qc.refit        → qc.metrics
qc.review_page  → pipeline.manifest   *qc.stress_words
qc.verdicts     → pipeline.manifest, qc.stress_words
qc.tune_page    → qc.bakeoff
research:  avoidance_ab → pipeline.{manifest, s3_translate}, qc.stress_words
           axis_check, judge → qc.refit
           stress_consensus → stress_variants, *stress_mfa
           stress_mfa → stress_variants, *stress_detect
```

Other facts about how modules find each other and their files:

- `pipeline.course` and `pipeline.experiments` import nothing internal. They
  shell out to `[sys.executable, "dub.py", …]` with `cwd=ROOT`.
- Six `qc` modules (`backcheck`, `bakeoff`, `batch_report`, `review_page`,
  `axis_check`, `avoidance_ab`) do `sys.path.insert(0, parents[1])`.
- No production module imports a research module today. `avoidance_ab`,
  `axis_check` and `judge` import production code, which is the allowed
  direction.

### 1.3 Pod vs Mac vs research

The pod runs `dubadabidu run --from s4_synthesize --to s5_fit` (the default
`remote run` slice), `bakeoff`, `autopilot --no-mux` and `preamble`
(`REMOTE_TASK`). Because `dub.py` imports eagerly, a pod `run` today imports
**every** module listed at the top of `dub.py`, including the ones it never
calls.

| Needed on the pod | Imported there but never used | Mac only | Research only |
|---|---|---|---|
| s3 (`shorten`), s4, s5, `tts_engine`, `manifest`, `text_norm`, `logic`, `device`, `config_schema`, `qc.metrics`, `qc.backcheck`, `qc.evaluate`. For `remote bakeoff` / `preamble` / `autopilot` also: `bakeoff`, `tune`, `prep`, `review_page`, `stress_words`, `autopilot`, `batch_report`, s6, s7 | s1, s2 (+`asr` mlx path), s8, `refit`, `verdicts`, `prep` on a plain `run` | s1, s2, s6 (pedalboard), s7, s8, `runpod_infra`, `runlog`, `course`, `experiments`, `verdicts`, `refit`, `blind`, `compare`, `listen`, `tune_page`, `doctor` | `stress_mfa`, `stress_detect`, `stress_var`, `stress_variants`, `stress_consensus`, `axis_check`, `avoidance_ab`, `judge` |

### 1.4 Where config is read raw

About 270 reads. Every stage receives the merged plain dict, and defaults live
in four kinds of places:

1. **Inline `.get(key, default)`:** 105 sites. 61 match the schema, the rest
   differ (§3.3).
2. **Module `DEFAULTS` dicts merged under the config:** `runpod_infra.DEFAULTS`
   (21 keys), `tune.DEFAULTS` (6), `prep.PREP_DEFAULTS` (7).
3. **Fallback constants in code:** `tts_engine.DEFAULT_RANK_W`, the
   `synth_hash` legacy salts, and `manifest.DEFAULT_QWEN_MODEL`.
4. **Entry points that bypass validation:** `pipeline/course.py:84`,
   `qc/blind.py:260` and `qc/avoidance_ab.py:149` read `config.yaml` with
   `yaml.safe_load` and never validate it.

### 1.5 Implicit anchors (why only `--help` works from a wheel)

- **`dub.py`:** `ROOT = Path(__file__).parent`, then `os.chdir(ROOT)`.
  Installed, `ROOT` is site-packages.
- **Paths relative to cwd:** `Path("glossary")`, `f"prompts/{name}.md"`,
  `Path("ref")`, `Path("runs.jsonl")`, `Path("work/.runpod_active.json")`,
  `f"ratings_{lang}.json"`, `f"comparisons_{lang}.json"`,
  `"judge_validation.json"`, `f"stress_lexicon_{lang}.json"`,
  `"specs/batch.yaml"`, `Path("FIXES.md")`, `"work/.models/…"`.
- **Module-relative paths:** `course.py` and `experiments.py` use
  `ROOT = Path(__file__).parents[1]`.
- **Inconsistent CLI resolution:** `--config` is resolved against cwd first,
  but `--overlay` is resolved against `ROOT` only.

---

## 2. Target architecture

### 2.1 Layout

```
src/dubadabidu/                       the only package in the wheel
  __init__.py  __main__.py            python -m dubadabidu == the console script
  core/                               L0: no internal dependencies except core
    config/schema.py                  pydantic models; THE ONLY place a config default is declared
    config/loader.py                  load(project, --config, --overlay…, --engine) -> Settings; overlay search path
    languages.py                      the language registry (§4)
    project.py                        project-root resolution + named paths (§6)
    manifest.py  text_norm.py  logic.py  device.py  runlog.py
    audio.py                          trim/fade shared by s6 and take selection (today s6_mix._clean)
  engines/                            L1: model and service adapters, no policy
    tts/__init__.py                   synthesize(), engine_available(), release_models(), engine table
    tts/qwen.py  tts/edge.py
    asr.py                            faster-whisper / mlx-whisper
    llm.py                            OpenAI-compatible client (_chat, _loads, schemas, missing_models)
  qc/                                 L2: production scoring only
    metrics.py  backcheck.py  evaluate.py
    selection.py                      best-of-N take selection + ranking (moved OUT of tts_engine)
    stress_words.py                   lexicon IO and avoid list (production: s3, review, verdicts)
  stages/                             L3: s1_extract … s8_mux (one module each, unchanged names)
  review/                             L4: the verdict flywheel
    review_page.py  verdicts.py  refit.py  batch_report.py  blind.py  compare.py  listen.py  tune_page.py
  lab/                                L5: experiments the CLI still exposes
    bakeoff.py  tune.py  prep.py  queue.py (= pipeline.experiments)
  infra/remote/                       L5: GPU rental (§5)
    provider.py  runpod.py  orchestrator.py  transport.py  bootstrap.py  state.py  budget.py  diagnostics.py
  workflows/                          L6: multi-stage drivers
    autopilot.py  course.py  preamble.py
  cli/                                L7: argparse, dispatch, doctor
    __init__.py (main)  parser.py  doctor.py  commands/{run,qc,remote,lab,review}.py
  prompts/*.md                        package data (importlib.resources)

research/        NOT in the wheel. stress_mfa, stress_detect, stress_var, stress_variants,
                 stress_consensus, axis_check, avoidance_ab, judge (`python -m research.judge`)
pipeline/  qc/   clone-only compat shims for the documented `python -m` entry points (Q3).
                 They are not in the wheel.
```

### 2.2 Why each boundary is where it is

| Boundary | Justification |
|---|---|
| **core** at the bottom | Manifest IO, hashes, config and language data are needed by everything, including the pod. They must import nothing above them, so a pod never needs pedalboard, mlx or the review stack just to read a manifest. |
| **engines** vs **qc** | An engine turns text into a WAV. *Which* take ships is policy that depends on QC metrics. Keeping the policy inside `tts_engine` is the only reason an engine imports `qc`. Moving `synth_best_of`, `_take_rank`, `early_accept_ok` and `_f0_delivered` to `qc/selection.py` removes two of the three true inversions. The third (`→ s6_mix._clean`) goes away when the trim moves to `core/audio.py`. |
| **qc** below **stages** | s4 and s5 rank takes and s3 reads the avoid list, so stages *consume* QC. Production QC holds the metric definitions that 72 rated groups depend on (bit-identity), so it must stay small and stable, and must not import a stage. |
| **review** above stages | The review pages, verdicts and refit read finished manifests and write the ledger. Nothing in the pipeline needs them, so a pod never imports them. |
| **lab** and **infra** above review | `bakeoff` builds a review page; `tune` and `prep` score with QC. `infra` needs `stages.s8_mux` (local mux after `remote autopilot`). The two are independent of each other. |
| **workflows** and **cli** on top | `autopilot`, `course` and `preamble` compose stages, review and infra. The CLI is glue only; doctor checks live in their own module. |
| **research/** outside the package | Refuted tools (FINDINGS §2.1g/h/i/q/s) must stay reproducible but must never ship, and must never be imported by production. A separate top-level directory makes that structural: it is not in the wheel, and nothing under `src/` can import it without a test failing. |

### 2.3 Allowed dependency direction and how it is enforced

```
cli → workflows → {lab, infra} → review → stages → qc → engines → core
                   (siblings in one layer may import each other, except lab ↔ infra)
research → anything in src/dubadabidu          src/dubadabidu → research : FORBIDDEN
```

**Enforcement: `tests/test_layers.py`.** It is a pure-AST check with no new
dependency. It walks every module under `src/`, records every import (the lazy
ones too), maps the module to its layer, and fails on:

1. an upward edge;
2. any import of `research`;
3. `lab ↔ infra`;
4. a module outside the layer map (a new module must declare its layer).

It also checks **(5) the pod import closure**: the modules transitively
imported by `cli.commands.run` must not include `stages.s1_extract`, `s6_mix`,
`s8_mux`, `review.*` or `research.*`. This needs lazy per-command dispatch in
the CLI (commit C3).

import-linter would express the same contracts, but it needs a new dev
dependency and therefore a `uv.lock` change, which the brief forbids (Q9).

During migration the test runs with an explicit allowlist of today's three
violating edges. Each commit that removes an edge deletes its allowlist line.
The allowlist must be empty at the end.

---

## 3. Config: one typed object, defaults only in the schema

### 3.1 Target

`core/config/loader.load()` does, in order:

1. YAML load;
2. overlays, deep-merged left to right (unchanged `deep_merge`);
3. `--engine` applied to the raw dict before validation, exactly as today;
4. `Config.model_validate`;
5. returns a frozen `Settings` (`ConfigDict(frozen=True, extra="forbid")`).

Stages, QC, lab, infra and workflows receive `Settings`. Derived configs go
through one helper:

```python
def derive(s: Settings, patch: ConfigPatch) -> Settings:
    """Re-validate deep_merge(s.model_dump(exclude_unset=True), patch)."""
```

`ConfigPatch` is a recursive TypedDict alias, not `Any`. Today's derivations
are `preamble`'s tune overrides, bakeoff grid variants and the manifest's
`tts_overrides` merge in s4, s5, evaluate and bakeoff. That keeps the "later
wins" semantics byte-for-byte.

### 3.2 How neutrality is proven (commit A4, before any consumer changes)

**(a) Round-trip fidelity, for every overlay combination in the repo.** The
combinations are the base, base+gpu, base+gpu+each overlay, and
base+source-ru:

```
json.dumps(validate(raw).model_dump(exclude_unset=True), sort_keys=True)
  == json.dumps(raw, sort_keys=True)
```

This proves pydantic coerces nothing: no int↔float, no `"12.8"` string turning
into a number, no reordering that matters. Any coercion fails it, and that
matters because `synth_hash` JSON-encodes raw config values (`reference_max_s`,
`qwen_gen_kwargs`).

**(b) Effective-value equivalence for every read site.** The A5 scanner emits a
checked-in inventory, `tests/config/reads_inventory.json`, with one row per read
site: `path`, `kind` (`[]` / `.get(k)` / `.get(k, d)` / `DEFAULTS` merge) and
the literal default. For every overlay combination and every row, the test
asserts:

```
raw_lookup(raw, path, kind, code_default)  ==  typed_lookup(settings, path)
```

`.get(k)` with no default is compared by truthiness, because that is how every
such site uses it. A second test regenerates the inventory and asserts it is
current, so a new raw read cannot slip in unlisted.

**(c) A drift ledger.** Every row where code and schema disagree is listed in
`tests/config/drift.py`, with the decision from §3.3 and its reason. A row may
leave the list only in the commit that implements its decision. When a
decision changes an *effective* value for some combination, that commit is a
labelled `fix:` and needs your approval first.

**(d) Migration consumer by consumer.** During the transition `Settings`
carries `.raw` (the validated, merged dict), so an unmigrated helper still
works. Each migration commit converts one layer. It deletes the code defaults
for the keys it converts, shrinks the inventory, and keeps (a), (b), the
goldens (§8 A3) and the neutrality harness (§9) green. The last commit removes
`.raw`, and the scanner then asserts **zero** raw reads.

**(e) Cache keys.** `synth_hash` keeps its exact key layout. It reads typed
fields, and the two chatterbox-era salts (`cfg_weight` 0.0, `exaggeration`
0.55) become named constants inside the hash function. They are a persisted
key format, not config defaults. A golden matrix pins the result: every
overlay × language × fixture text → hash (A3). The `seg_hash` and staleness
keys are pinned the same way.

### 3.3 Every place a code default disagrees with the schema

**Effective today in a shipped combination:**

| Key | Schema | Code | Combos where it bites | Recommendation |
|---|---|---|---|---|
| `tts.qwen_fast` | `False` | `False` at `tts_engine._load_qwen:104` and falsy at `synth_hash:337` / `bakeoff:152`; **`True` at `engine_available:280`** (s5's "is the engine installed here" probe) | `config.yaml` alone (gpu.yaml sets it). The probe then demands `faster_qwen3_tts` while the loader would run stock qwen. Visible only on a machine with `qwen_tts` and without `faster_qwen3_tts`. | **Bug**: list it, and preserve it during migration. The site reads `model_fields_set` and carries a `# BUG(§10)` marker. Fix it to the schema default in a separate `fix:` commit if you approve (Q8a). |

**Latent:** the code says "absent = off", the schema says `True`, and every
shipped `config.yaml` sets the key, so nothing changes today.

| Key | Code site | Recommendation |
|---|---|---|
| `asr.vad_filter` | `asr.py:121` `.get("vad_filter")` | schema wins (`True`) |
| `separation.enabled` | `dub.py:78` (doctor) | schema wins |
| `mix.master.enabled` | `s6_mix.py:252`, `dub.py:86` | schema wins |
| `mux.per_language` | `s8_mux.py:75` | schema wins |

**Code-only defaults:** the schema says `None`, and the code supplies the real
value.

| Key | Code default | Recommendation |
|---|---|---|
| `tts.qwen_warmup_len` | 100 (`tts_engine:146`) | move 100 into the schema |
| `bakeoff.subset_size` / `takes` / `scoring_workers` | 6 / 3 / 1 (`bakeoff:369-419`) | move into the schema |
| `bakeoff.tune.subset_size` / `takes` | 3 / 2 | move into the schema (gpu.yaml sets 3 / 4) |
| `translation.lang_workers` | `len(langs)` | keep `None` = "one per language", resolved in one helper |
| `tts.cfg_weight` / `exaggeration` | 0.0 / 0.55 (`synth_hash`) | hash-format constants (§3.2e), not schema defaults |
| `runpod.gpu_count`, `ports`, `ssh_user`, `remote_dir`, `provision_timeout_s`, `hold_on_failure_min`, `capacity_wait_min`, `capacity_retry_s`, `volume_gb`, `min_cuda_version`, `allowed_cuda_versions`, `name` | `runpod_infra.DEFAULTS` and `pod_payload:141` | move into the schema (Q8b) |
| `tune.rounds` | `["R1","R2","R3"]` (`tune.DEFAULTS`) vs schema `None` | move into the schema |

**Two defaults for one key:**

| Key | Defaults | Recommendation |
|---|---|---|
| `tune.r1_takes` | **3** (`tune.DEFAULTS`), **1** (`tune.py:178`, dead because DEFAULTS fills it first), **1** (schema). config.yaml sets 3. | schema = 3, the value that applies today when the key is unset |
| `bakeoff.engines` | `[]` (`dub.py:105`, `runpod_infra:936/1361/1445`) vs `[incumbent, "qwen"]` (`bakeoff.py:368`), applied only when the `bakeoff` block is absent | Q8c: `Config.bakeoff` defaults to `None`. Each site keeps its absent-block semantics through one typed helper. Alternatively, pick one meaning, which changes `dubadabidu bakeoff` without the gpu overlay. |
| `qc.eval.weights` | `evaluate:96` `{sim .5, mos .35, tempo .15}`, `refit:379` `{sim .25, mos .4, f0 .2, tempo .15}`, `tts_engine.DEFAULT_RANK_W` `{mos .4, f0 .2, sim .2, tempo .2}`; the schema makes it **required** | dead under validation; delete all three when that consumer migrates |

**Dead because the schema requires the key:** the code default can never apply
once config is validated, so delete it when its consumer migrates.

- `qc.eval.score_flag` 0.55 in four places (shipped value 0.36);
  `qc.eval.sim_floor_default`.
- `qc.refit.*` (7 keys).
- `tts.best_of` 1, `retake_mos_below` 0.0, `early_accept_mos` 3.6,
  `early_accept_sim` 0.5.
- `fit.soft_tempo`, `drift_max_s`, `min_gap_s`.
- `mix.duck_*` (4 keys).
- `prep.*` (`PREP_DEFAULTS`: `min_s` 12.0 vs shipped 10.0, `n_refs` 3 vs
  shipped 4).
- The required `runpod.*` fields: `gpu_type_ids`, `image`, `container_disk_gb`
  (30 vs 40), `budget_usd`, `assumed_price_per_hr` (0.25 vs 0.70),
  `max_runtime_hours` (3.0 vs 6.0), `ssh_key`. These DEFAULTS apply **only**
  when the whole `runpod` block is absent: `remote status|kill`, `doctor`,
  `remote run --reuse` without the gpu overlay. See Q8b.

**Undeclared:** the code reads these with a default, but `extra="forbid"`
rejects them if set, so they are constants dressed up as config.

- `asr.no_speech_threshold` 0.6, `asr.compression_ratio_threshold` 2.4,
  `asr.condition_on_previous_text` False (`asr.py:100-106, 150-152`).
- `runpod.direct_ssh_wait_s` 150, `runpod.provision_attempts` 3
  (`runpod_infra:646, 690`).

Recommendation (Q8d): make them named module constants and keep their
comments.

**Falsy-equivalent:** `.get(k)` with a schema default of `False`, `{}` or `""`.
About 20 rows; no decision is needed, and the equivalence test covers them.

---

## 4. Language registry

### 4.1 Today: five tables in five places (after the branch merges)

| Table | Where | Holds |
|---|---|---|
| `LANG_NAMES` | `s3_translate.py:17` | en fr de es ru pl uk → prompt names |
| `QWEN_LANGS` | `tts_engine.py:81` | en fr de es ru → Qwen names (anything else becomes `"Auto"`) |
| `tts.edge_voices` | `config.yaml` (required) | en fr de es ru pl uk → edge voice |
| `mux.lang_tags` | `config.yaml` (required) | uk en fr de es ru pl → ISO 639-2 |
| `_GLOSSARY_LEAD` | `s2_transcribe.py:21` | uk, ru → "Lesson vocabulary:" lead-in |
| `STRESS_LANGS` | `qc/stress_words.py:58` | ru, uk |
| routing | `config.yaml` `engine_by_lang: {uk: edge}` + `source_language: Literal["uk","ru"]` | which engine and which sources |

### 4.2 Target: `core/languages.py`

```python
Role = Literal["source", "target"]
Engine = Literal["qwen", "edge"]

@dataclass(frozen=True)
class Language:
    code: str                    # manifest / CLI / glossary key (ISO 639-1)
    name: str                    # English name used in prompts ({SRC}/{LANG}, f-strings)
    whisper: str | None          # Whisper language code; None = cannot be a source
    qwen: str | None             # Qwen3-TTS language name; None = qwen cannot voice it
    edge_voice: str | None       # edge-tts voice
    sub_tag: str                 # ISO 639-2 for mux / mov_text
    roles: frozenset[Role]
    forced_engine: Engine | None # engine this target must use whatever tts.engine says
    glossary_lead: str | None    # s2 initial_prompt lead-in (source languages)
    stress_marking: bool         # review page offers word marking (STRESS_LANGS)

LANGUAGES: Mapping[str, Language] = {   # one row per language
  "uk": Language("uk", "Ukrainian", "uk", None,      "uk-UA-OstapNeural",  "ukr", {"source","target"}, "edge", "Словник уроку: ", True),
  "ru": Language("ru", "Russian",   "ru", "Russian", "ru-RU-DmitryNeural", "rus", {"source","target"}, None,   "Словарь урока: ", True),
  "en": Language("en", "English",   None, "English", "en-US-AndrewNeural", "eng", {"target"},          None,   None, False),
  ... fr, de, es, pl (pl: qwen=None, as today, where it falls through to "Auto")
}
```

The rules:

- **Engine resolution keeps today's order and adds the registry in the
  middle:** `tts.engine_by_lang[lang]` → `LANGUAGES[lang].forced_engine` →
  `tts.engine`. With `engine_by_lang: {uk: edge}` moved from `config.yaml`
  into the registry, `resolve_engine` returns the same value for every
  language and every `--engine` (Q10). `synth_hash`, which keys on the
  resolved engine, is unchanged; the A3 golden proves it.
- **Config overrides stay possible.** `tts.edge_voices` and `mux.lang_tags`
  become optional per-key overrides, applied on top of the registry. A test
  asserts registry ∪ `config.yaml` equals today's tables exactly.
- **Qwen language names keep their fallback:** `qwen_name(code)` returns
  `LANGUAGES[code].qwen or "Auto"`, so `pl` behaves as before.
- **Validation:** `source_language` stays a `Literal`, and a test ties its
  arguments to `{c for c, l in LANGUAGES.items() if "source" in l.roles}`.
  `languages` gains a validator (every entry must be a target). That is
  stricter, but every shipped config already passes it.

**Adding a language** becomes one registry row, plus `glossary/<src>/<lang>.csv`
if wanted. The prompts are already parameterised (`{SRC}` / `{LANG}` on the
branch).

### 4.3 `text_uk` → `text_src`

The branch already does the rename. `manifest._rename_source_text` renames on
read for manifests and `refs.json`, and `s3.load_terms` renames `uk` → `src`
in `terms_*.json`. The next `save()` writes the new key.

What is left (F3) is an explicit, idempotent `dubadabidu migrate [video…]`
command:

- it is a dry run by default and lists what would change;
- it rewrites manifests, `refs.json` and `terms_*.json` on disk;
- it **never** touches the ledger files;
- doctor warns when any `work/*/manifest.json` still says `text_uk`.

The read shim stays for at least one minor release (Q11).

### 4.4 Prompts

The branch's templates already use `{SRC}`, `{LANG}`, `{TOL}` and `{NVAR}`. The
f-strings in s3 move from `LANG_NAMES[...]` to `lang_name(...)`. A golden
records the **rendered** system prompts and user payloads, byte for byte, for
every (source, target) pair. The neutrality harness records every request the
stubbed LLM receives, so a change to a prompt's text, the payload keys or the
batching shows up as a diff.

---

## 5. Remote infrastructure

### 5.1 How `runpod_infra.py` splits (1,525 lines today)

| New module | Contents | Provider-specific? |
|---|---|---|
| `infra/remote/bootstrap.py` | `UV_VERSION`, `REMOTE_SETUP`, `apt_setup()`, `REMOTE_TASK`, `ENGINE_MODULE`, `engine_install_cmd()`, `engines_for_task()` | no (`REMOTE_SETUP`'s comment mentions the RunPod base image only as context) |
| `infra/remote/transport.py` | `ssh_exec`, `ssh_capture`, `wait_ssh`, `rsync`, `rsync_paths`, `_ssh_base`, `_MODEL_CACHE_EXCLUDES`, `free_engine`, `_verify_ready`, taking an `SshEndpoint(key, user, host, port)` | no |
| `infra/remote/state.py` | state-file protocol: `save`, `held`, `clear`, `terminate_tracked`, `sweep_orphans` (needs a `Provider`) | no |
| `infra/remote/budget.py` | `_deadline`, re-deriving the deadline from the real price (shorten only), and the `RunRecord` that replaces the global `_LAST_POD` | no |
| `infra/remote/provider.py` | `Provider` Protocol, `InstanceInfo` / `PodShape` TypedDicts, `NoDirectSsh` | — |
| `infra/remote/runpod.py` | `RunPodProvider`: REST v2 `_req`/`_key`, `pod_payload`, `create` (with the capacity loop over `gpu_type_ids`), `get`, `list_all`, `destroy`, `is_gone`, `ssh_target`, `container_started`, `shape` (price/vcpu/ram), `watchdog_command` (`runpodctl remove pod`) | **yes** |
| `infra/remote/orchestrator.py` | `remote_run` lifecycle: reuse/held decision → sweep → pre-flight refusals → deadline → provision with type rotation → arm watchdog → apt → sync up → setup/verify → engine install → task → sync back → local mux → ledger row → hold / keep-alive / **always terminate**. Also the **terminate retry-and-verify loop**. | no |
| `infra/remote/diagnostics.py` | `smoke_test`, `setup_check`, `status` | no (provider passed in) |

### 5.2 The provider Protocol

This makes `MIGRATION_VAST.md` one new class plus a config block.

```python
class InstanceInfo(TypedDict, total=False):
    id: str
    status: str
    raw: Mapping[str, object]        # provider payload, for diagnostics only

class PodShape(TypedDict):
    price_per_hr: float | None
    vcpu: float | None
    ram_gb: float | None

class Provider(Protocol):
    name: str                         # "runpod" | "vast"
    key_env: str                      # "RUNPOD_API_KEY" — read locally, NEVER sent to the instance
    def create(self, spec: RemoteSettings, deadline: float) -> InstanceInfo: ...
    def get(self, iid: str) -> InstanceInfo: ...
    def list_all(self) -> list[InstanceInfo]: ...
    def destroy(self, iid: str) -> None: ...          # raises; "gone" is decided by is_gone()
    def is_gone(self, info: InstanceInfo) -> bool: ...
    def ssh_target(self, info: InstanceInfo) -> tuple[str, int] | None: ...
    def container_started(self, info: InstanceInfo) -> bool: ...
    def shape(self, info: InstanceInfo) -> PodShape: ...
    def watchdog_command(self, iid: str, seconds: int) -> str | None: ...
        # pod-side self-destruct shell snippet; None = the provider has none, so the
        # orchestrator arms the local detached reaper instead (MIGRATION_VAST §4 L2)
```

Settings follow the same split. `RemoteSettings` holds the provider-neutral
keys: budget, deadline cap, ssh, `remote_dir`, image, disk, `engine_setup`,
hold, capacity wait. `RunpodSettings(RemoteSettings)` adds `gpu_type_ids`,
`cloud_type` and the CUDA filters. The YAML key stays `runpod:`, so validation
and every overlay stay as they are. The `runpod:` → `remote: {provider: …}`
rename that `MIGRATION_VAST.md` §6 proposes belongs to the Vast commit, not
this refactor. So do the state-file `"provider"` field and its rename: they
would change the state file's bytes, so they also wait for Vast.

### 5.3 The pod-side bootstrap stays byte-identical

These strings move module **unchanged**, comments included:

- `REMOTE_SETUP` (and its `.format(dir=…)` output);
- `apt_setup(True)` and `apt_setup(False)`;
- `engine_install_cmd(…)`;
- `REMOTE_TASK[task].format(…)` for each task;
- the `arm_pod_watchdog` command;
- `pod_payload(rp, gpu)` for `config.gpu.yaml`.

A3 pins all of them as golden strings, before the split.

Overlay paths in `REMOTE_TASK` (`--overlay config.gpu.yaml --overlay
config.deepseek.yaml`) and the name filter `Path(o).name != "config.gpu.yaml"`
stay valid without edits because of the overlay search path (§7, Q5). **No
diff to any pod-side command.**

### 5.4 How the split's neutrality is proven

Commit G1 adds a characterization test **before** the split. It drives
today's `remote_run` with fake REST, ssh and rsync functions (monkeypatched at
module level) through eight scenarios:

1. fresh run succeeds;
2. task fails with a hold;
3. `--keep-alive`;
4. `--reuse` with a live pod;
5. `--reuse` with no pod;
6. `bakeoff` sync;
7. the missing-snippet refusal;
8. `NoDirectSsh` type rotation.

It records the ordered call log: every REST call with its body, every ssh
command string, every rsync argv, every state-file write and the ledger row.
After the split, the same scenarios run against `orchestrator` +
`RunPodProvider` + a fake transport, and the logs must match exactly.

---

## 6. Running from an installed package

**Project-root resolution** lives in `core/project.py`. The first match wins:

1. `--project DIR` (new global flag);
2. `$DUBADABIDU_HOME`;
3. the nearest ancestor of cwd that contains `config.yaml` (git-style walk-up),
   so running from any subdirectory of a clone keeps working;
4. **editable-install fallback:** the source checkout that contains the
   package, if it has `config.yaml`. That is exactly today's
   `ROOT = dirname(dub.py)` behaviour for clone users who run the venv's
   `dubadabidu` from anywhere;
5. otherwise, exit with: "no dubadabidu project here: pass `--project` or set
   `DUBADABIDU_HOME` (a project is a directory containing `config.yaml`)".

**The anchor.** I recommend keeping exactly **one** `os.chdir(project.root)`
in `cli.main`, after resolution and replacing `chdir(ROOT)` (Q7). The reason
is that relative config values are themselves hash inputs: `"ref":
tts_cfg["reference_wav"]` goes into `synth_hash` as the literal string
`ref/sketch_ref_08.wav`. Anchoring every path explicitly would mean touching
every use site without changing any of those strings, which is a large surface
with no output benefit.

Everything else becomes explicit:

- packaged resources come from `importlib.resources`;
- ledger, specs and configs paths come from `Project`;
- `course` and `experiments` run `[sys.executable, "-m", "dubadabidu", …]`
  with `cwd=project.root`;
- the `sys.path.insert` hacks are removed.

| Lives in the project directory (clone or user dir) | Lives in the package |
|---|---|
| `config.yaml`, `configs/` (overlays, `configs/experiments/`), `specs/`, `glossary/`, `ref/`, `input/`, `output/`, `work/`, `ledger/`, `.env` | code, `prompts/*.md`, the config schema, the language registry, the bootstrap strings |

The CLI semantics are kept as they are:

- `--config` is resolved against cwd, then the project;
- `--overlay` is resolved as an absolute path, then project-relative, then the
  search path;
- video paths are resolved against cwd before the chdir, as today;
- `.env` is read from `project/.env`.

**Pod:** it runs `cd {remote}; dubadabidu run …`, so step 3 finds the synced
project. `REMOTE_SETUP`'s `uv sync` installs the package editable from `src/`.
The Docker image uses `--no-install-project`, so it is layout-independent.
`uv.lock` is unchanged: the project entry is `editable = "."`, and the package
layout is not in the lock.

---

## 7. Root-directory layout

The rule: the root keeps metadata and config entry points; data and docs go
into named directories. Basenames are **kept** wherever a ledger, a command
line, `REMOTE_TASK`, FINDINGS or `experiments.yaml` refers to a file by name.
With the overlay search path (`project root` → `configs/` →
`configs/experiments/`, with a test that basenames are unique across them),
every existing `--overlay config.X.yaml` keeps working, so the ledger files
need **no edits**.

| Current file(s) | Destination | References to update |
|---|---|---|
| `config.yaml` | stays (entry point) | — |
| `config.gpu.yaml`, `config.deepseek.yaml`, `config.gemini.yaml`, `config.source-ru.yaml` | `configs/` (Q5) | README table and commands (optional, since the search path resolves them), `tests/test_config_schema.py` glob, `test_runpod`/`test_translation_models`/`test_gpu_phase` fixture paths. `REMOTE_TASK`, `course.py` and `experiments.py` need **none**. |
| 17 × `config.exp.*.yaml` | `configs/experiments/` | the schema-test glob and the README table. FINDINGS and `experiments.yaml` need none. |
| `experiments.yaml` | `specs/experiments.yaml` (next to `specs/batch.yaml`) | default path in `lab/queue.py`, the README command list, `test_runpod.py` |
| `runs.jsonl`, `ratings_{en,ru}.json`, `comparisons_{en,ru}.json`, `judge_validation.json`, `stress_lexicon_ru.json` | `ledger/` | `runlog.py`, `verdicts`, `blind`, `refit`, `compare`, `stress_words`, `research/judge`, `research/axis_check`, `research/avoidance_ab`, `autopilot`; tests `test_refit`, `test_stress_words`, `test_text_norm`, `test_deps`, `test_avoidance_ab`; README "Cost" one-liner, AUTOPILOT flywheel paragraph, CHANGELOG, a Dockerfile comment. Pod: `rsync` is exclude-based, so `ledger/` still syncs as today (s3 reads the lexicon on `remote autopilot`). |
| `FINDINGS.md` | `ledger/FINDINGS.md` (Q6) | README (3 mentions), CHANGELOG, `runlog.py` docstring, `config*.yaml` comments ("FINDINGS §" references are by name and stay), `test_deps.py` |
| `AUTOPILOT.md`, `MIGRATION_VAST.md` | `docs/` | README, CHANGELOG, `.env.example`, 6 module docstrings, `specs/batch.yaml` comment |
| `ARCHITECTURE_PLAN.md` (this file) | `docs/ARCHITECTURE.md`, rewritten to the final state at the end | README |
| `dub.py` | `src/dubadabidu/cli/` (`git mv`, then split) | `pyproject [project.scripts]`, `tests.yml` comment, `test_audit.py`, `test_runpod.py`, `MIGRATION_VAST.md`, `course.py` / `experiments.py` subprocess argv |
| `prompts/` | `src/dubadabidu/prompts/` | `s3._prompt` (`importlib.resources`) |
| `glossary/`, `specs/`, `docker/`, `scripts/`, `tests/` | stay | — |
| `README.md`, `LICENSE`, `CHANGELOG.md`, `THIRD_PARTY.md`, `pyproject.toml`, `uv.lock`, `requirements.txt`, `.env.example`, `.gitignore` | stay (metadata; Q14) | `.gitignore` gains `!tests/fixtures/**` (it ignores `*.mp4` and `*.wav`) |
| untracked `ARCHITECTURE_PROMPT.md`, `CLEANUP_REPORT.md` | not touched (Q16) | — |

The resulting root:

```
README.md  LICENSE  CHANGELOG.md  THIRD_PARTY.md  pyproject.toml  uv.lock  requirements.txt
.env.example  .gitignore  config.yaml
configs/  docker/  docs/  glossary/  ledger/  research/  scripts/  specs/  src/  tests/
```

---

## 8. Migration plan

Every commit is independently green: `uv run pytest -q`, `uvx ruff check .`
and the neutrality harness (§9). History follows the files because moves use
`git mv` and a move is never mixed with edits beyond import lines.

The **branch** column says whether the commit would conflict with
`chore/torch-2.14.1` if that branch were *not* merged first. Because step 0
merges it, **none of them conflicts in practice**.

**Rollback** is `git revert <sha>` for every commit, since each is
self-contained. A later commit may build on an earlier one; reverting out of
order then needs the dependents reverted first, and the table names them.

### Phase 0: prerequisite (your action)

| # | Change | Proof |
|---|---|---|
| 0 | Fast-forward `main` to `chore/torch-2.14.1` (8 commits, 0 behind), or tell me to wait | Re-baseline: pytest count, ruff, and the harness goldens are generated **after** this |

### Phase A: safety net (tests only, no production change)

| # | Commit | Files | Proves | Branch |
|---|---|---|---|---|
| A1 | `test: neutrality harness: local stages on a fixture, network stubbed` | `tests/neutrality/{conftest,stubs,test_golden}.py`, `golden_fast.json`, `tests/fixtures/neutrality/{clip.mp4,make_fixture.sh}`, `.gitignore` | §9 fast mode. Green on unchanged code; two consecutive runs give identical output (determinism). | Y |
| A2 | `test: full-mode harness + differential runner` | `tests/neutrality/test_full.py`, `scripts/neutrality_diff.py` | real ffmpeg, pedalboard, ECAPA, Distill-MOS and Whisper from local caches, offline; base-vs-HEAD byte diff | N |
| A3 | `test: golden strings and hashes` | `tests/golden/test_{hashes,prompts,remote_cmds,cli}.py` + json | `synth_hash`/`seg_hash` matrix (every overlay combo × lang × text); rendered prompts per (src, tgt); `REMOTE_SETUP`, `apt_setup`, `engine_install_cmd`, `REMOTE_TASK`, watchdog command, `pod_payload`; the argparse `--help` text and parse results for an argv matrix | Y |
| A4 | `test: config round-trip, read inventory, drift ledger` | `tests/config/{test_equivalence.py, reads_inventory.json, drift.py}`, `scripts/scan_config_reads.py` | §3.2 (a) to (c) | Y |
| A5 | `test: layer contract with today's allowlist` | `tests/test_layers.py`, `scripts/import_graph.py` | §2.3; allowlist = the 3 inverted edges | Y |
| A6 | `test: project_root fixture; no Path(__file__) or cwd reliance` | `tests/conftest.py`, ~18 test files | the suite passes from any cwd (`pytest --rootdir . tests` run from `/tmp`) | Y |
| A7 | `test: comment census` | `scripts/comment_census.py`, `tests/test_comment_census.py` | Every comment line that holds `FINDINGS §`, `PINNED`, `load-bearing`, `measured` or a date survives somewhere in the tree, compared as a multiset against a checked-in census. A move cannot drop one. | Y |

### Phase B: remove the true inversions in place (small)

| # | Commit | Files | Proves | Branch |
|---|---|---|---|---|
| B1 | `refactor: move segment trim/fade (s6_mix._clean) to pipeline/audio.py` | `pipeline/{audio,s6_mix,tts_engine}.py` | harness, A3; allowlist −1 | Y |
| B2 | `refactor: extract take selection from tts_engine into qc/selection.py` | `tts_engine`, new `qc/selection`, `s4`, `s5`, `tune`, `qc/bakeoff`, `qc/blind`, 5 test files | harness (s4 selection is exercised with best_of 3), `test_early_accept`, `test_best_of_cache`, `test_zero_slot_rescue`; allowlist −2 → empty | Y |
| B3 | `refactor: move research tools to research/` (`git mv` ×8) | `qc/{stress_mfa,stress_detect,stress_var,stress_variants,stress_consensus,axis_check,avoidance_ab,judge}.py` → `research/`, their tests → `tests/research/`, README/CHANGELOG mentions | `test_layers` gains a research-isolation rule; `python -m research.judge --help` works | Y |

### Phase C: one package

| # | Commit | Files | Proves | Branch |
|---|---|---|---|---|
| C1 | `refactor!: move pipeline/ and qc/ into src/dubadabidu/<layer>/` | **Pure `git mv`** plus a scripted import rewrite (the mapping table goes in the commit message). `pyproject [tool.setuptools]` → src layout plus `[tool.pytest.ini_options] pythonpath = ["src", "."]`. Clone-only shims `pipeline/{course,experiments}.py` and `qc/{blind,compare,listen,tune_page}.py`. Test imports and monkeypatch targets; harness stub map. | harness plus a differential diff against B3 (byte-identical); `uv lock --check` unchanged; the `uv build` wheel lists only `dubadabidu/` (+ `dub.py`) | Y (all files) |
| C2a | `refactor(cli): project-root resolution (core/project.py) in dub.py` | `dub.py`, `src/dubadabidu/core/project.py`, tests | `chdir(ROOT)` → `chdir(resolve_project(…))`; resolution unit tests; harness; A3 CLI golden | Y |
| C2b | `refactor(cli): dub.py → dubadabidu.cli (git mv, then split)` | `git mv dub.py src/dubadabidu/cli/__init__.py` in its own commit, then the split into `parser.py`, `doctor.py`, `commands/*`, `workflows/preamble.py` in the next; `__main__.py`; `[project.scripts] dubadabidu = "dubadabidu.cli:main"`; `course.py`/`experiments.py` argv; `test_audit` paths | A3 `--help` / parse golden, harness, `test_audit` | Y |
| C3 | `refactor(cli): lazy per-command imports` | `cli/commands/*` | `test_layers` pod-closure rule (§2.3 #5); harness | N |

### Phase D: root layout

| # | Commit | Files | Proves | Branch |
|---|---|---|---|---|
| D1 | `feat(config): overlay search path (root → configs/ → configs/experiments/)` | `core/config/loader.py`, tests | A3 CLI golden (resolution of every existing overlay argument); uniqueness test | Y (dub.py) |
| D2 | `chore: git mv config.exp.*.yaml → configs/experiments/` | 17 files, schema-test glob, README | A4 round-trip over the same combos; harness | Y |
| D3 | `chore: git mv provider and profile overlays → configs/` (if Q5 = A) | 4 files, tests, README | A3 `REMOTE_TASK` golden **unchanged**; A4 | Y |
| D4 | `chore: ledger/ for runs, ratings, comparisons, judge, lexicon, FINDINGS` | the 7 files plus the reference list in §7 | `test_refit`, `test_stress_words`, `test_deps`; harness (verdicts ingestion writes to `ledger/`); `git log --follow` works | Y |
| D5 | `chore: experiments.yaml → specs/; AUTOPILOT, MIGRATION_VAST → docs/` | 3 files + references | `pipeline.experiments --dry-run` output golden (commands identical) | Y |
| D6 | `refactor: prompts → package data` | `prompts/*.md` → `src/dubadabidu/prompts/`, `stages/s3_translate.py` | A3 rendered-prompt golden | Y |

### Phase E: typed config (one layer per commit, bottom-up)

| # | Commit | Proves |
|---|---|---|
| E1 | `refactor(config): single loader; Settings + .raw flow everywhere; course/experiments/blind validate` | A4; harness. Note that `course.py`, `qc.blind` and `avoidance_ab` now validate, which is stricter but every shipped config passes. |
| E2 | `refactor(config): schema adopts the §3.3 decisions (code-only and duplicate defaults → schema)` | A4 (b): effective values unchanged for every combo |
| E3 | `… core: manifest helpers, synth_hash, text_norm take Settings` | A3 hash matrix |
| E4 | `… engines: asr, tts.qwen, tts.edge` | harness |
| E5 | `… stages s1, s2` | harness |
| E6 | `… stage s3 + engines.llm` | harness (recorded LLM requests) |
| E7 | `… qc.selection, s4, s5` | harness; `qwen_fast` probe keeps its marked bug |
| E8 | `… s6, s7, s8` | harness (audio and mp4 bytes) |
| E9 | `… qc metrics, backcheck, evaluate` | full-mode metric bit-identity |
| E10 | `… review/*` | `test_refit`, `test_qc_staleness`, harness verdict ingest |
| E11 | `… lab: tune/prep DEFAULTS deleted; bakeoff grids via derive()` | `test_bakeoff*`, `test_variant_presynth` |
| E12 | `… workflows + infra (Runpod model replaces the DEFAULTS dict)` | A3 remote goldens, `test_runpod` |
| E13 | `… cli: doctor, report` | doctor output snapshot (env-independent lines) |
| E14 | `refactor(config): drop Settings.raw; scanner asserts zero raw reads` | A4 inventory = empty; the schema docstring is rewritten |

Rollback for E3 to E13: revert in reverse order. E14 depends on all of them.

### Phase F: language registry

| # | Commit | Proves |
|---|---|---|
| F1 | `refactor: core/languages.py replaces LANG_NAMES, QWEN_LANGS, _GLOSSARY_LEAD, STRESS_LANGS` (pure; can land right after Phase D) | A3 prompts, hashes, engine routing; registry ∪ config == today's tables |
| F2 | `chore(config): edge_voices / lang_tags / engine_by_lang{uk} come from the registry` (Q10) | A4 effective values; harness `--langs en,uk` |
| F3 | `feat: dubadabidu migrate (text_uk → text_src on disk; dry run by default)` (Q11) | unit tests on legacy fixtures; never touches `ledger/` |

### Phase G: remote split (§5)

| # | Commit | Proves |
|---|---|---|
| G1 | `test(remote): orchestrator characterization: 8 scenarios, recorded call logs` (on unsplit code) | green on current code |
| G2 | `refactor(remote): bootstrap.py (strings moved verbatim)` | A3 byte-identical strings, A7 comments |
| G3 | `refactor(remote): transport.py (SshEndpoint)` | G1 logs |
| G4 | `refactor(remote): state.py + budget.py (RunRecord replaces _LAST_POD)` | G1 logs, ledger row identical |
| G5 | `refactor(remote): Provider protocol + RunPodProvider; terminate retry-and-verify in the orchestrator` | G1 logs; fake-provider conformance test |
| G6 | `refactor(remote): orchestrator.py + diagnostics.py; CLI wired through the provider` | G1, A3, `test_runpod` |
| G7 | `test(remote): split test_runpod.py into tests/infra/{test_runpod_provider, test_orchestrator, test_bootstrap, test_transport, test_cuda_pins}.py` | same test count, same names |

### Phase H: module splits and tests

| # | Commit | Proves |
|---|---|---|
| H1 | `refactor: engines/tts → __init__ / qwen / edge` | harness, A3 |
| H2 | `refactor: s3's LLM client → engines/llm.py` | harness (recorded requests) |
| H3 | `refactor: lab/bakeoff → bakeoff/{variants,runner,scorecard}` (optional, Q12) | `test_bakeoff*` |
| H4 | `refactor: review/refit → refit/{data,fit,gates}` (**deferred** by recommendation, Q12) | `test_refit` |
| H5 | `test: tests/ mirrors the layers (git mv)` + `tests.yml` paths | same count |
| H6 | `test(layers): allowlist empty; final contracts` | — |

### Phase I: docs and acceptance

| # | Commit | Proves |
|---|---|---|
| I1 | `docs: README (layout, --project, wheel use), AUTOPILOT paths, CHANGELOG [Unreleased], docs/ARCHITECTURE.md` | links checked by a grep test |
| I2 | *(no commit)* wheel acceptance in a scratch venv | see the bullets below |

What I2 does:

- `uv build`, then install into a scratch venv from the local uv cache with
  `--offline`;
- `dubadabidu --project <fixture-project> doctor`;
- `run --from s6_mix` on a copy of the harness's `work/`: local ffmpeg only, no
  network, and the output bytes must equal the harness's.

**Expected end state:** about 40 commits; `runpod_infra` (1,525) becomes eight
modules of ≤ 450 lines each; `dub.py` (697) becomes `cli/` modules of ≤ 250
lines; `tts_engine` (671) becomes `tts/qwen` ~300, `tts/edge` ~30 and
`qc/selection` ~300.

---

## 9. Neutrality harness (A1 and A2)

### 9.1 Fixture

A committed 6-second clip, `tests/fixtures/neutrality/clip.mp4` (~60 KB,
320×180, a voice-band tone plus noise). It is committed, not generated per
run, so the input bytes are the same on every machine. `make_fixture.sh`
documents how it was made.

### 9.2 Running it

The harness runs the real CLI in-process, `cli.main(argv)`, with
`config.yaml` plus a harness overlay. The overlay sets:

- `work_dir` and `output_dir` under the pytest tmp dir;
- `--engine edge`, `languages: [en, ru]` (plus `uk` after the merge);
- `translation.passes: 3` with `adequacy_check: true`, so all three prompts
  and the judge are exercised;
- `tts.best_of: 3` with `rank_takes: true`, so selection is exercised;
- in fast mode only, `mix.master.enabled: false` (pedalboard is absent in CI).

It runs in four steps, which exercise `--from`/`--to` and resume:

1. `run --to s2_transcribe`;
2. `run --from s3_translate --to s5_fit`;
3. `run --from s6_mix`;
4. `run` again with nothing to do. This is the **resume check**: zero new
   synthesis calls and zero new LLM calls, and identical bytes.

### 9.3 Stubs at the outermost boundary

Stubs sit at the edge of the system, never on internal names, so a refactor
cannot route around them.

| Boundary | Stub |
|---|---|
| Separation | the `audio_separator` / `demucs` entry used by s1 is replaced: vocals = input mono, background = seeded low noise |
| ASR | the `asr.transcribe` backend returns three fixed Ukrainian segments with word timings |
| LLM | an `openai.OpenAI` fake. Its reply is a pure function of the request (e.g. a deterministic transliteration plus fixed variants and adequacy scores). It **records every request**: model, system prompt, user payload, `response_format`. |
| edge-tts | an `edge_tts.Communicate` fake whose `.save()` writes an MP3 by running ffmpeg on a seeded sine whose length is proportional to the text, so the real `_synth_edge` ffmpeg path still runs |
| QC models | fast mode: `qc.metrics` scoring and the Whisper back-transcription are replaced by deterministic functions of the WAV's sha256. Full mode: the real models from `work/.models` and the HF cache, with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. |
| Network | `socket.create_connection` and `getaddrinfo` raise, which proves nothing leaks |

### 9.4 What the golden records

- The exact bytes of every `manifest.json`, after replacing the tmp root with
  `<TMP>`.
- The SRTs.
- The recorded LLM requests (canonical JSON).
- The sha256 of every WAV, M4A and MP4 under `work/` and `output/`.
- **The argv of every `subprocess.run`** (all ffmpeg command lines). A
  refactor that changes a filter argument fails even if the audio happens to
  match.
- `synth_hash` for every segment.
- In full mode: every QC metric, compared with `==` on the float's `repr`.
  Bit-identical, not approximate.

### 9.5 Portability

Audio hashes depend on the ffmpeg build, so `golden_fast.json` keys them by the
first line of `ffmpeg -version` plus the pedalboard version. On a different
toolchain the audio hashes are reported as "not comparable here". The
manifests, SRTs, LLM requests, argv and hashes are still compared.

CI has no ffmpeg (it pip-installs a minimal set), so CI runs only the A3
goldens; the harness skips there with a stated reason (Q13). Locally, on the
Mac, the full harness runs after every commit.

### 9.6 Differential mode

`scripts/neutrality_diff.py <base-ref>` is the strongest check:

1. `git worktree add` the base and HEAD into the scratchpad;
2. run each tree's own harness in dump mode with the same venv;
3. byte-diff the two dumps;
4. diff the A7 comment census.

It is machine-independent and needs no stored goldens, so it covers audio
across toolchains. I run it after every Phase B to H commit, against the
previous commit and against the post-merge baseline.

---

## 10. Risks, bugs found, and decisions for you

### 10.1 Risks

| Risk | Mitigation |
|---|---|
| A monkeypatch target drifts after a move: a test patches `tts_engine.synthesize`, but `selection` imported its own name | The harness stubs at the outer boundary (§9). Tests get their targets updated in the same commit as each move. `pytest -p no:randomly` order is kept. |
| Pydantic coerces a value that feeds a hash | A4 round-trip plus the A3 hash matrix, both before any consumer changes |
| A derived config (`derive`) carries a key the schema forbids (e.g. bakeoff's internal variant keys) | It surfaces in E11's tests. Internal keys move out of the config dict into explicit parameters. |
| Lazy CLI imports change import-time side effects (`qc.metrics.set_device` imports torch) | `set_device` stays at startup for every command, as today; the harness compares log-free outputs |
| `os.chdir` stays (Q7) | It is one documented line. `Project` holds every path we create; config strings that feed hashes stay relative. |
| The editable-install fallback hides a wrong project | doctor prints the resolved project and how it was found |
| A ledger move breaks a human's muscle memory | basenames are unchanged, the README has a layout table, and `git log --follow` works |
| The Vast seam is designed without a second implementation | G5 adds a `FakeProvider` that implements the Protocol, and runs the G1 scenarios against it |

### 10.2 Bugs found and left alone (each would be a separate, labelled `fix:` commit, only if you approve)

1. **`tts.qwen_fast` probe default `True` vs loader `False`**
   (`tts_engine.py:280` vs `:104`). s5 can refuse to synthesize a fit variant
   with "engine not installed" on a machine that has stock `qwen_tts`, with
   base config only.
2. **Code treats absent keys as False while the schema documents `True`:**
   `asr.vad_filter`, `separation.enabled` (doctor), `mix.master.enabled`,
   `mux.per_language`. Latent, because `config.yaml` sets all four.
3. **Two defaults for one key:** `tune.r1_takes` (3 in `DEFAULTS`, 1 inline,
   1 in the schema) and `bakeoff.engines` (`[]` vs `[incumbent, "qwen"]`).
4. **Three different fallback weightings for `qc.eval.weights`** (`evaluate`,
   `refit`, `DEFAULT_RANK_W`). Dead under validation, but misleading.
5. **`runpod_infra.DEFAULTS` disagrees with `config.gpu.yaml`:** disk 30 vs 40,
   assumed price 0.25 vs 0.70, runtime cap 3 h vs 6 h, a different GPU list.
   These apply on `remote run --reuse` without the gpu overlay, so the deadline
   there is computed from 0.25 $/h and 3 h.
6. **Five undeclared knobs** that the code reads but `extra="forbid"`
   rejects: `asr.no_speech_threshold`, `asr.compression_ratio_threshold`,
   `asr.condition_on_previous_text`, `runpod.direct_ssh_wait_s`,
   `runpod.provision_attempts`.
7. **Unvalidated entry points:** `pipeline/course.py`, `qc/blind.py` and
   `qc/avoidance_ab.py` load `config.yaml` without validation and without
   overlays.
8. **`--overlay` relative paths ignore cwd** (only `ROOT`), while `--config`
   tries cwd first. Kept as is.
9. **The wheel ships top-level packages named `pipeline` and `qc`.** That is a
   site-packages collision risk with any other distribution using those names.
   The C1 move fixes it as a side effect of the restructure, not as a separate
   behaviour change.
10. **On the branch:** the `text_uk` read shim rewrites a manifest's key on its
    next `save()`. That is the branch's accepted change, and the reason the
    harness baseline must be post-merge.

### 10.3 Decisions only you can make (my recommendation first)

- **Q1. Sequencing.** Recommendation: **fast-forward `chore/torch-2.14.1`
  into `main` before Phase A.** Every commit here touches a file it changes,
  and the goldens must describe post-merge behaviour. Alternative: wait until
  you have finished with it (its worktree has untracked `work-edge/`).
- **Q2. Package.** Recommendation: **`src/dubadabidu/` with layers.**
  Alternative: restructure in place and keep the top-level `pipeline` and `qc`
  names (less churn, keeps the collision risk).
- **Q3. Entry-point shims.** Recommendation: **clone-only shims** for
  `python -m pipeline.course|experiments` and
  `qc.blind|compare|listen|tune_page`, not shipped in the wheel. Also add
  `dubadabidu course` / `dubadabidu experiments` subcommands so the shims can
  be retired later.
- **Q4. `dub.py`.** Recommendation: **remove it**; `dubadabidu` and
  `python -m dubadabidu` replace it. Alternative: a 3-line root shim (keeps a
  root `.py` file).
- **Q5. Overlay placement.** Recommendation: **A, the search path, with every
  overlay under `configs/`** and only `config.yaml` at the root. No command
  line, `REMOTE_TASK`, `experiments.yaml` or FINDINGS reference changes.
  Alternative B: keep the 4–5 production overlays at the root as entry points
  (the brief allows it), move only experiments, and edit the paths in
  `experiments.yaml`.
- **Q6. FINDINGS.md location.** Recommendation: **`ledger/FINDINGS.md`**,
  with `runs.jsonl` (its other half). No edits inside it.
- **Q7. Path anchoring.** Recommendation: **keep one
  `os.chdir(project.root)`** in `cli.main`. The alternative, fully explicit
  paths everywhere, is much larger and has no output benefit. Defer it.
- **Q8. Drift decisions.**
  - **(a) `qwen_fast` probe:** preserve during the refactor, then a separate
    `fix:` to the schema default.
  - **(b) runpod `DEFAULTS` → schema defaults:** this relaxes validation of a
    *partial* `runpod` block, because the 7 required fields become optional
    with today's DEFAULTS values. Effective values are unchanged everywhere.
    The alternative keeps them required and keeps a second defaults table for
    the absent block, which violates "defaults in one place".
  - **(c) `bakeoff.engines`:** keep per-site absent-block semantics behind one
    helper.
  - **(d) undeclared knobs → module constants.**
  - The latent rows: the schema value wins.
- **Q9. Layer enforcement.** Recommendation: **the pytest AST check**, with no
  dependency change. Alternative: import-linter, which means adding a dev
  dependency and changing `uv.lock`.
- **Q10. Registry and config.** Recommendation: the registry owns
  `edge_voices`, `lang_tags` and uk→edge; the config keys stay as optional
  overrides.
- **Q11. `dubadabidu migrate`.** Recommendation: **yes**, dry run by default,
  and keep the read shim one more release.
- **Q12. Splitting `bakeoff` and `refit`.** Recommendation: split `bakeoff`
  (H3); **defer `refit`**. It is closed by FINDINGS §2.1p, stable, and gains
  nothing.
- **Q13. CI.** Recommendation: CI runs the A3 goldens and the layer test, and
  the harness stays local. Alternative: add ffmpeg to CI and run fast mode
  there (a CI workflow edit).
- **Q14. `requirements.txt` and `THIRD_PARTY.md`.** Recommendation: **stay at
  the root** as packaging and licensing metadata.
- **Q15. Test layout.** Recommendation: mirror the layers (H5), done after the
  code moves.
- **Q16. Untracked `ARCHITECTURE_PROMPT.md` and `CLEANUP_REPORT.md`.**
  Recommendation: leave them untouched. Alternative: commit them under `docs/`.
- **Q17. `dubadabidu init`** (writes a starter project for wheel users).
  Recommendation: **defer**. A copied `config.yaml` is enough for the
  definition of done, and a packaged template would be a second copy of the
  config that can drift.

---

## Stop

Phase 1 ends here. Nothing has been moved, committed or pushed. Phase 2 starts
with step 0 (Q1) once you approve this plan or tell me what to change.
