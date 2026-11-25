# AUTOPILOT — self-improving dubbing with a human only at the eval gate

Goal: the human provides credentials and verdicts; the tooling does everything
else. That works here because the objective already exists (qc_score), all
state lives in one editable manifest per video, every stage is an idempotent
CLI command, and content-hash caching makes fix -> re-run surgical. What an
agent loop needs on top is policy, not plumbing.

Caveat on the objective: FINDINGS §2.1p closed weight re-fitting — on 72
length-free groups no weighting of the current metrics reproduces the ear. The
loop below therefore treats qc_score as a triage signal and the human verdict
as the decision.

## Architecture (three layers + one flywheel)

### 1. Spec layer — `specs/batch.yaml`, data not prose
The live spec is **`specs/batch.yaml`**; read it there (a copy here drifted
once). Three blocks: `accept` (per video/lang thresholds on qc_score, flagged
%, WER, overflow/overlap), `budget` (mechanical re-roll rounds; the pod's dollar
cap is `runpod.budget_usd`), and `policy` (`never`: publish, delete work/, edit
eval weights, edit the spec). The human improves the system by editing the
spec, not the pipeline.

### 2. Orchestrator — `dubadabidu autopilot` (`pipeline/autopilot.py`)
Perceive -> diagnose -> act -> re-evaluate, using only existing commands:

- perceive: `dubadabidu batch`, manifests, tune reports
- diagnose flagged segments by failure class:
  - wer high         -> re-roll the take (each fresh take is back-transcribed
                        and a hallucination is vetoed) or fix the translation
  - sim low          -> re-roll
  - overflow/overlap -> rescue is automatic; else edit the text in the manifest
  - term wrong       -> fix glossary/terms, re-run s3 for that segment
- act: edit manifest / run stage / `run --from s4`
- re-evaluate until the spec is met or the budget is spent, else ESCALATE with
  reasons. Every (symptom -> fix -> outcome) is appended to `FIXES.md`, the
  local playbook (git-ignored runtime output).
- human verdicts win: an accepted segment is never re-rolled, a rejected one
  always is.

### 3. Infra agent — `dubadabidu remote` (`pipeline/runpod_infra.py`)
`RUNPOD_API_KEY` + `TRANSLATE_API_KEY` in `.env` (the human provides them once).
RunPod REST v2, on-demand SECURE pods (spot is refuted, FINDINGS §3):
provision -> rsync (never the source video) -> run -> sync back -> ALWAYS
terminate. Guardrails: hard budget -> deadline cap, terminate retry + verify,
an independent pod-side self-destruct watchdog, `remote kill` for orphans.
The pod is cattle; all state returns to this repo. Vast.ai is the planned next
provider (docs/MIGRATION_VAST.md).

### The flywheel — human verdicts are training data
Review pages carry accept/reject buttons per segment; `dubadabidu verdicts`
writes them back to the manifest (`human_rating`/`human_verdict`) and
accumulates rows in `ledger/ratings_<lang>.json`. `dubadabidu refit` re-derives
qc.eval.weights from those rows and **proposes only** (the spec forbids the
agent editing weights). Adoption needs three gates: a cross-validated rho floor,
a permutation test, and beating the incumbent out of fold; it refuses below
`qc.refit.min_rows`.

## Human touchpoints (all of them)
1. Credentials in `.env` (once): translation provider, RunPod.
2. Verdicts: ~15 min/video on worst-first review pages.
3. Taste gates the spec cannot encode (a native listen per language) and the
   publish decision. The agent NEVER publishes.

## Milestones (all done; code comments cite them by number)
- **M1** orchestrator loop against the spec, locally (edge engine).
- **M2** RunPod lifecycle (`dubadabidu remote`), validated on a real lesson.
- **M3** review-page verdict capture -> manifest + `ledger/ratings_<lang>.json`.
- **M4** weight re-fit (`dubadabidu refit`) + the FIXES.md playbook.

## Invariants
- The eval harness and the ear decide, not model reputation.
- Every candidate change beats the incumbent before adoption.
- Licence hygiene (THIRD_PARTY.md) and pinned versions stand.
