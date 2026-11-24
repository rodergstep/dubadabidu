# PLAN v0.5 — from the 2026-11-19 review to the 20-video batch

Source: the 2026-11-19 project review (web research + code/lint/test pass).
Every item says what changes, where, how we know it is done, and what it costs.
House rules carried over from FINDINGS: **measure before adopting**, compute the
binomial before believing a blind test, one axis per experiment, and nothing in
the CLOSED list is re-opened without NEW evidence (§1 below is the one item that
has it).

**Progress 2026-11-19 (implementation pass):** P0, P1, P2, P5, P6, P7 done;
P3 code done (image not yet built — needs a push to `main`); P4 wired, run is
human-gated; P8 dropped; P9 pending. `remote setup-check` PASSED on REST v2
with the uv bootstrap: ~2.3 min SSH-ready -> probe vs ~11.5 min with pip. Discovered on the way and fixed: RunPod
REST v1 was retired 2026-11-15 (every `remote` command was broken) — client
migrated to v2 (see README "New in v0.5").

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why)

## Order and dependencies

```
P0 housekeeping ─┐
P1 DeepSeek pin ─┼─► P2 lockfile + lint ─► P3 pod image ─┐
                 │                                       ├─► P9 the batch
P4 RU-stress A/B ┴──────────► (adopt?) P4.5 productionise ┘
P5 judge validation (offline, independent)
P6 docs/config cleanup  (after P4 — it rewrites the same tts comments)
P7 typed config/manifest (after P6 — validate the cleaned config, not the old one)
P8 pydub replacement    (optional, any time after P2)
```

P0, P1, P4 and P5 can run at the same time — they touch different files.
P4 does NOT wait for P3: one A/B on today's bootstrap costs well under $1.

---

## P0 — Housekeeping (free, ~15 min)

- [x] Switch to `main` and pull. `feat/ru-stress-word-marking` was squash-merged
      as PR #11; it shows 21 commits "ahead" only because of the squash, and
      `main` has 2 commits it lacks.
- [x] Delete merged local branches (`feat/ru-stress-word-marking`,
      `fix/gpu-bakeoff-working`, `fix/setup-check-guard`,
      `fix/stale-3h-comments-and-doctor`, `claude/reverent-babbage-a724cb`) and
      their remotes after a `git branch --merged main` check.
- Done when: `git branch` shows `main` plus whatever this plan creates.

## P1 — DeepSeek model pin (free, ~1 h) — URGENT

`deepseek-v4-flash` is retired upstream and temporarily routed to V4.1 Flash.
The official changelog points at `deepseek-flash`; one third-party source says
`deepseek-v4.1-flash`. Any NEW video is already being translated by a model
nobody evaluated, with no warning, and the temporary route has no end date.

- [x] Confirm the live id: `GET https://api.deepseek.com/models` with the key.
- [x] `config.yaml`, `config.deepseek.yaml`, README provider table: set the
      confirmed id; delete the stale "deprecated 2026-07-24" lines.
- [x] `pipeline/s3_translate.py`: record the model that ANSWERED (the
      `response.model` field, not the requested name) in the manifest stage
      entry `stages["s3_<lang>"]`, so a silent upstream swap is visible later.
- [x] `dub.py doctor`: when the key is set, list `/models` and fail if
      `translation.model` (and `reflect_model` / `adequacy_model`) is absent.
- [x] Test: doctor check with a mocked `/models` response (present / absent).
- [x] FINDINGS §5: one line recording the rename and the date.
- Done when: `dubadabidu doctor` passes with the new id, and a fresh `s3` on
  sketch60 writes the answering model into the manifest.
- Note: cached translations are unaffected (s3 skips segments with text). The
  20 batch videos will be the first to use V4.1 Flash — accept that knowingly,
  or spot-check one video's adequacy scores against sketch60's before P9.

## P2 — Lockfile and lint in CI (free, ~half a day)

The `transformers` 5.x incident (a pod lost to an unpinned transitive resolve)
is the failure a lockfile prevents structurally.

- [x] `uv lock` against `pyproject.toml`; commit `uv.lock`. Pin the currently
      floating deps there (`jiwer`, `num2words`, `pyyaml`, `mlx-whisper`,
      `pedalboard`, `pytest`) to the versions in today's `.venv`.
- [x] torch stays OUT of the lock's default resolution (it comes from the cu128
      index on the pod). Model it as a uv index source for the `gpu` marker
      group, or keep installing it first and `uv sync --inexact` after.
      Decide by trying both on one `remote setup-check` (~$0.015).
- [x] Replace the hand-mirrored `requirements.txt` with
      `uv export --no-hashes > requirements.txt`; change `tests/test_deps.py`
      to assert the export is current instead of asserting two hand-written
      files agree.
- [x] `REMOTE_SETUP` interim speed-up (until P3 lands): `pip install uv` then
      `uv pip install` / `uv sync --frozen` instead of pip. Keep the cu128
      index pin and the CUDA assert exactly as they are.
- [x] Lint, real items only (from `ruff --select F,B`):
      - unused imports (13), `dub.py:139` unused `stem`, the 3 F811
        redefinitions in `tests/test_gpu_phase.py` / `tests/test_runpod.py`;
      - 7 `zip()` without `strict=` in `qc/` — add `strict=True` where the
        lengths MUST match (refit, axis_check, bakeoff, stress_var);
      - B023 (closure over loop variable) is a false positive here — the
        closures run inside the same iteration (checked `pipeline/s5_fit.py:82`).
        Add a per-file ignore with that reason rather than restructuring.
- [x] `pyproject.toml` `[tool.ruff]` with `select = ["F", "B"]`; CI step
      `uvx ruff check .` before the tests.
- Done when: CI green with ruff + 411 tests; a fresh `uv sync` reproduces the
  `.venv`; one `remote setup-check` passes and prints the bootstrap time.

## P3 — Prebuilt pod image (one-off ~2 h + ~$0.10 of pods; saves ~11 min/pod)

`runs.jsonl`: bootstrap 694 s of a 724 s run, 732 s of 1464 s. The image ships
torch 2.9.1, then `REMOTE_SETUP` builds a fresh venv and reinstalls torch 2.11
and everything else on every pod.

- [x] `docker/Dockerfile`: base `runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404`
      (keeps sshd/ports behaviour), apt rsync+ffmpeg (`--no-install-recommends`,
      the sshd lesson), `uv sync --frozen` from P2's lock, torch 2.11.0 from
      cu128, qwen_tts at the pinned SHA `022e286`, `faster-qwen3-tts==0.3.2`.
- [x] Weights: bake `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, ECAPA and Distill-MOS into
      `/opt/hf` with `HF_HOME` set, so no first-call download. Measure image
      size; if it exceeds ~15 GB, bake code only and keep weights downloading.
- [x] `.github/workflows/image.yml`: build on changes to `uv.lock`,
      `docker/**`, `runpod.engine_setup`; push to GHCR tagged with the lock hash.
      Private registry → RunPod registry auth in `.env` (human touchpoint).
- [x] `pipeline/runpod_infra.py`: if `/opt/venv` exists and its lock hash
      matches the local `uv.lock`, skip `REMOTE_SETUP` entirely; otherwise run
      it as today (fallback, never a hard dependency on the image).
- [ ] `config.gpu.yaml` `runpod.image`: the GHCR tag. `remote setup-check`
      must still print resolved torch + CUDA (never trust an image tag — §4).
- [x] Tests: image-skip path chosen when hashes match, full setup when they
      don't; image tag in config matches the lock hash.
- Done when: two consecutive `remote setup-check` runs show bootstrap < 120 s
  in `runs.jsonl`, and `cost_per_video_hour()` is re-derived.
- Follow-up experiment (only after it lands): spot re-test. FINDINGS §3 refuted
  spot because every death came during the ~11-min bootstrap; that window is
  what P3 removes. 3 spot pods on sketch60, `interruptible: true` in a throwaway
  overlay; adopt only if all 3 reach synthesis AND survive to the end.

## P4 — Russian stress: test the stress-aware Qwen3-TTS fine-tune (~$1 pods + ~1 h listening)

NEW EVIDENCE per the CLOSED rule: `siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf`,
a fine-tune of our exact base (`Qwen3-TTS-12Hz-1.7B-Base`), Apache-2.0 weights,
same loader, reads U+0301 after the vowel — which `text_norm` already emits.
Author-reported mark adherence 96% vs base 17% (his own detector, not ears).

### P4.0 Gates before spending anything
- [x] **License call (human) — ACCEPTED 2026-11-24 by the owner.** The weights are Apache-2.0, but part of the
      training TEXT (biggest-ru-book) declares no license and the model card
      itself says to check before commercial use. Decide whether that is
      acceptable for paid courses BEFORE building on it.
- [x] **Loader check.** Read `faster-qwen3-tts` 0.3.2 `from_pretrained` source:
      does it accept an arbitrary HF repo id? If not, the A/B runs the stock
      `qwen_tts` path (`qwen_fast: false`, RTF ~1.4; fine for 46 segments) and
      the speed path is a P4.5 problem.
- [x] **Unknown that shapes the arms:** the fine-tune was trained in ICL layout
      (Russian reference + Russian transcript). Ours is a Ukrainian reference
      and production uses x_vector-only. Whether x_vector-only keeps the stress
      gain is unknown; that is why it gets its own arm.

### P4.1 Experiment wiring
- [x] `qc/bakeoff.py variant_key`: add a suffix for a non-default
      `qwen_model_dir` (e.g. `+m:<last path part>`). Today two runs differing
      only in model dir would OVERWRITE each other's row and audio — the exact
      failure the function exists to prevent. Test it.
- [x] Overlays (one axis apart each, `bakeoff.tune.enabled: false`, ref fixed):
      - `config.exp.ru-ft-plain.yaml` — fine-tune, x_vector, NO marks
        (does the fine-tune hurt ordinary Russian?);
      - `config.exp.ru-ft-stress.yaml` — fine-tune, x_vector, `ru_stress: true`
        (the candidate);
      - `config.exp.ru-ft-icl-stress.yaml` — fine-tune, ICL with the existing
        UA reference transcript, `ru_stress: true` — run ONLY if the x_vector
        candidate fails on adherence.
      - Control: `config.exp.ru-control.yaml` (production), already exists.
- [x] `experiments.yaml`: one entry per arm with `asks/decides`, one pod via
      `--reuse --keep-alive`.

### P4.2 Metric gate (cheap, before any listening)
- [ ] Reject an arm outright if backcheck WER > 0.10 on the 46 ru segments
      (base model with marks went 0.068 → 0.811; that failure is loud).
- [ ] sim / mos / f0st against the noise floor (§0). Recorded, not decisive —
      none of them hears stress.

### P4.3 Ear (the decision)
- [ ] Blind same-sentence pairs, candidate vs control, stress-focused axis
      printed on the page. Prioritise sentences containing the 9 words in
      `stress_lexicon_ru.json` plus RUAccent homographs.
- [ ] ≥ 23 pairs (at n=23 a win needs 17–6 for p < 0.05). Use the existing
      positional-bias gate.
- [ ] Also mark mis-stressed words on the review page for both arms → a direct
      before/after stress-error rate against the ~29%-of-takes baseline.

### P4.4 Decision
- [ ] Adopt only if: stress-error rate falls with p < 0.05, no rise in
      "unusable" marks, WER gate passed. Otherwise: FINDINGS entry, REFUTED.
- [ ] Either way: new FINDINGS §2.1r and an updated CLOSED table row.

### P4.5 Productionise (only if adopted)
- [x] `tts.qwen_model_dir_by_lang: {ru: siriusfreak/...}` resolved per
      language, mirroring `engine_by_lang` / `manifest.resolve_engine`.
- [x] `synth_hash` already keys on `qwen_model_dir`; make it key on the
      RESOLVED per-language value. Test: same text, ru vs en, different keys.
- [x] `tts_engine._load_qwen`: key the singleton on the model id and reload
      (via `release_models`) when it changes — today a multi-language run
      would keep the first model loaded for every language. Test it.
- [x] Safety rail: refuse `ru_stress: true` unless ru resolves to a
      stress-trained model (2.1b: marks destroy the base model's output).
- [ ] P3 image bakes the second checkpoint; `remote setup-check` probes it.
- [ ] Flip `tts.ru_stress` for ru in `config.gpu.yaml`; re-synthesize ru.
- Bonus to check, not to build: the fine-tune's author uses a stress detector
  ("meter2", 98.3% reported). If it is public, validate it against the
  human-marked lexicon BEFORE use (method rule; four detectors failed that
  gate). Its training data includes non-commercial Silero audio, so QC-only.

## P5 — Validate a learned judge against the 72 same-sentence groups (offline, ~$0–0.50)

FINDINGS §2.1p reopens take ranking only for "a metric that hears what these do
not". The data to test that already exists: `comparisons_{en,ru}.json` (72
groups, winner + unusable labels) and the rated audio under
`work/*/bakeoff/compare*/`.

- [x] `qc/judge.py` with two scorers behind one interface:
      - Meta Audiobox Aesthetics (small, per-clip, CPU-capable) — PQ/CE/CU axes;
      - SpeechJudge-GRM (pairwise naturalness reward model, ICLR 2026; needs a
        GPU — run on MPS if it fits, else one short pod).
- [x] Score every group with the SAME statistic as §2.1p: argmax agreement with
      the listener's pick, chance 0.444, permutation p.
- [x] Bias controls, both documented for these judges: score pairwise judges in
      BOTH orders and average (SpeechJudge has a reported 29-point preference
      for position B); loudness-match clips first (judges prefer louder audio).
- [x] Gate: agreement above chance at p < 0.05 in BOTH languages, held out.
- [x] If it passes: add `qc_judge` to `qc/metrics.py`, stamp it like the others
      (`_QC_STAMPS`), and only THEN rerun `dubadabidu refit` with it as a 5th
      feature. If it fails: FINDINGS entry, REFUTED, nothing wired in.
- Licences: check both before wiring anything into production (QC-only use
  is the bar, the audio is not derived from them).

## P6 — Docs and config cleanup (free, ~half a day; after P4)

- [x] `config.yaml` / `config.gpu.yaml`: every comment ≤ 2 lines plus a
      `FINDINGS §x` pointer. History moves to FINDINGS (most is already there).
      Fix the contradictions found in review:
      - `qc.eval.score_flag` says qc_score "actually tracks the ear (+0.364)";
        §2.1p says it agrees at chance — keep §2.1p's reading;
      - `tts.reference_wav` differs between the two files with stale "beats
        studio ref_04" text; state which wins and why in one line;
      - chatterbox-era keys (`cfg_weight`, `exaggeration`, `tune.cfg_weights`,
        `tune.exaggerations`) — remove from config. `synth_hash` keeps its
        hard-coded defaults so cache keys do not move; add a test that hashes
        of a fixture are unchanged after the removal.
- [x] `THIRD_PARTY.md`: the "torch stays at 2.6.0" section is superseded by
      the 2.11/cu128 stack — rewrite to current state.
- [x] `README.md`: stage table (mlx/faster-whisper, rubberband, soft-anchor),
      remove the hard-coded `/Users/...` path, install via `uv sync`, drop the
      "Old install section".
- [x] `AUTOPILOT.md`: the spec example still shows `engines_allowed:
      [chatterbox, qwen]` and `mean_score_min: 0.60` — point it at
      `specs/batch.yaml` instead of duplicating it.
- [x] `IMPROVEMENT_PLAN.md`: mark superseded by this file at the top.
- Done when: `grep -ri chatterbox config*.yaml` returns nothing outside a
  FINDINGS pointer, and tests still pass (cache-key test included).

## P7 — Typed config and manifest (~1 day; after P6)

CI's own header lists "a config key that never existed" among the shipped
bugs. There are no TypedDicts or dataclasses anywhere today.

- [x] `pipeline/config_schema.py`: pydantic v2 models per block (`asr`,
      `separation`, `translation`, `tts`, `fit`, `mix`, `mux`, `qc`, `prep`,
      `tune`, `runpod`, `bakeoff`) with `extra="forbid"`. Validate in `dub.py`
      after overlays are merged; pass the validated object's `model_dump()` on
      so no stage changes in this step.
- [x] Test: every `config*.yaml` (alone and with `config.gpu.yaml`) validates;
      a misspelled key fails with the key's path.
- [x] `pipeline/manifest.py`: TypedDicts for `Manifest`, `Utterance`,
      `Translation`, `Take`; annotate `load`/`save` and the stage `run`s.
      No `Any` — model unknown shapes explicitly.
- [ ] Optional after it lands: `pyright --basic` in CI on `pipeline/`.

## P8 — Replace pydub in s6 — DROPPED 2026-11-19

DROPPED on a closer read, not deferred. The RAM concern was already fixed:
`_assemble` lays segments into ONE numpy int32 accumulator (no pydub O(n^2)
overlay loop). The Python 3.13 concern has a standard drop-in (`audioop-lts`)
for the day torch/qwen allow 3.13 at all — they don't yet. A sample-exact port
of pydub's trim/fade/ratecv is regression risk for no current gain.

- [-] Rewrite `_assemble` / `_clean` / `_normalize_segment` on numpy +
      soundfile (same sample rate handling, same fades, same gain math).
- [-] Null test on sketch60: new vs old premix, max |diff| below −60 dBFS, and
      identical `norm_gain_db` per segment.
- [-] Drop pydub from deps; CI install line loses it.
- Does not by itself lift `requires-python <3.13` (torch/qwen pins decide that).

## P9 — The 20-video batch (IMPROVEMENT_PLAN Phase E, the actual goal)

Needs P1 (model pinned). Much cheaper after P3. ru quality depends on P4.

- [ ] One full ~1 h lesson end-to-end first (Phase D), all 5 languages, via
      `python -m pipeline.course` with `--reuse --keep-alive` amortising one
      pod; record wall clock + $ in `runs.jsonl`.
- [ ] Re-test `best_of: 2` on that run (config.gpu.yaml asks for it): how
      often is the shipped take marked unusable?
- [ ] Then the batch. Human budget ~15 min/video on worst-first review pages.
- [ ] Publish: YouTube multi-language audio is open to all creators since
      Sept 2025 — upload per-language audio in Studio; keep `per_language`
      MP4s as the fallback.

---

## Not doing (checked in the review, no action needed)

- Duration-aware translation: the per-segment seconds slot + measured cps +
  shorter variants already matches the 2025–26 literature; FINDINGS §4 shows
  61 variants pre-made and 0 used — fit is not a bottleneck.
- ASR, reference sweeps, generation params, 0.6B, ICL for en: CLOSED, no new
  evidence.
- B023 lint warnings: false positives (see P2).

## Human touchpoints in this plan

1. P1: run doctor with the real key (or let me).
2. P3: GHCR/registry credential for RunPod, if the image is private.
3. P4.0: the licence decision. P4.3: ~1 h of blind listening.
4. P9: review pages, as before.
