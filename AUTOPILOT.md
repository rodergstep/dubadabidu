# AUTOPILOT — self-improving dubbing with a human only at the eval gate

Goal: the human provides credentials and verdicts; LLM agents do everything
else. This repo is unusually ready for that: the objective function already
exists and is human-calibrated (qc_score, weights fit to your ratings), all
state lives in one editable manifest per video, every stage is an idempotent
CLI command, and content-hash caching makes fix -> re-run surgical. An agent
loop needs exactly those properties. What's missing is policy, not plumbing.

## Architecture (three layers + one flywheel)

### 1. Spec layer — specification-driven, specs are DATA not prose
`specs/batch.yaml` — acceptance criteria the agent must satisfy per video/lang:

```yaml
accept:
  mean_score_min: 0.60      # per language
  flagged_pct_max: 5        # segments below qc.eval.score_flag
  wer_bad_max: 0            # after fixes
  overflow_max: 0
  overlap_max: 0
budget:
  usd_per_video_max: 3.0
  gpu_hours_per_video_max: 1.5
policy:
  translation_model: deepseek-v4-flash
  engines_allowed: [chatterbox, cosyvoice]
  never: [publish, delete work/, change eval weights without human ack]
```

The agent's contract: make every row of `dubadabidu batch` satisfy `accept`
within `budget`, or stop and escalate with a reason + options. The human
improves the system by editing the spec, not the pipeline.

### 2. Orchestrator agent (Claude Code headless / Agent SDK)
Perceive -> diagnose -> act -> re-evaluate, using ONLY existing commands:

- perceive: `dubadabidu batch`, manifests, tune reports
- diagnose flagged segments by failure class:
  - wer high        -> re-roll take (delete cached wav) or fix translation text
  - sim low         -> re-roll; if systemic, re-run preamble ref pick
  - overflow/overlap-> rescue already automatic; else edit text in manifest
  - term wrong      -> fix glossary/terms_*.json, re-run s3 for that segment
- act: edit manifest / run stage / `run --from s4`
- re-evaluate; repeat until spec met or budget hit
- append every (symptom -> fix -> outcome) to `FIXES.md` — the playbook the
  next run reads first. This is the cheap half of self-improvement: the
  agent's diagnostic memory compounds across the 20-video batch.

### 3. Infra agent — RunPod lifecycle
`RUNPOD_API_KEY` + `TRANSLATE_API_KEY` in `.env` (human provides, once).
Scripted via RunPod REST/runpodctl: create spot pod from a pinned template ->
rsync up -> execute orchestrator remotely -> rsync down -> TERMINATE.
Guardrails: hard budget cap, max-runtime auto-kill, terminate-on-idle.
The pod is cattle; all state returns to this repo.

### The flywheel — human verdicts are training data
The review page grows accept/reject buttons per flagged segment; verdicts are
written back to the manifest (`human_verdict`) and accumulated in
`ratings_*.json`. Disagreement between human verdicts and qc_score IS the
signal: periodically re-fit qc.eval.weights against all accumulated ratings
(same spearman procedure used on 2026-07-08, now automated). Each cycle the
objective gets closer to your ear -> the tune loop optimizes a better target
-> fewer flags reach you next video. Human time per video decreases
monotonically if this loop is real.

## Human touchpoints (all of them)
1. Credentials in `.env` (once): DeepSeek, RunPod.
2. Verdicts: ~15 min/video on worst-first review pages (accept/reject).
3. Taste gates the spec can't encode: native-French listen (video 1),
   publish decision. The agent NEVER publishes.

## Milestones
- M1 `.env` contract + orchestrator loop running LOCALLY against the spec
     (edge engine, test30/test60 as fixtures) — proves the loop, zero cost.
- M2 RunPod lifecycle scripts; same loop remotely on the real engines.
     DONE: `pipeline/runpod_infra.py` + `dubadabidu remote <task>` (provision ->
     rsync (never the source video) -> run -> sync back -> ALWAYS terminate).
     Layered leak-prevention: state-file cleanup, terminate retry+verify,
     independent pod-side self-destruct watchdog, hard budget->deadline cap.
     `run`/`autopilot`/`bakeoff` tasks; mux happens locally. Lifecycle validated
     live (smoke test); full engine run pending the first real batch session.
- M3 Review-page verdict capture -> manifest/ratings writeback (closes the
     flywheel; do this BEFORE the 20-video batch so every verdict counts).
     DONE v0.4: review pages have accept/reject; `dubadabidu verdicts` writes
     back to the manifest + ratings_<lang>.json; the loop honors verdicts
     (accept = settled, reject = force re-roll) and WER re-rolls carry a
     per-take back-transcription veto.
- M4 Automated weight re-fit + FIXES.md playbook accumulation.
     DONE: `dubadabidu refit` (qc/refit.py) re-derives qc.eval.weights from the
     accumulated ratings_<lang>.json rows, by Spearman against your ratings and
     through the production composite itself. It PROPOSES only — this spec lists
     "edit eval weights" under `never`, and a scoring function that rewrites
     itself between videos makes every cross-video number incomparable.
     Adoption needs THREE gates: a cross-validated rho floor, a permutation test
     over shuffled ratings, and beating the incumbent out of fold. The last one
     alone is confounded — on ratings with no signal a fixed weighting can sit
     at a spuriously negative correlation while the fit drifts toward 0, reading
     as a large "gain" (measured: -0.136 -> -0.015). Refuses to propose below
     qc.refit.min_rows (30): four free parameters on fewer points fit noise.
     FIXES.md accumulation landed earlier with the autopilot (_log_fix).

## Inherited invariants (from IMPROVEMENT_PLAN, still binding)
- The eval harness decides, not model reputation — now enforced by spec.
- Every candidate change beats the incumbent on the harness before adoption.
- License hygiene list stands. Pinned versions stand.
