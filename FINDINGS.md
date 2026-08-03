# FINDINGS — what has actually been measured

Insights and verdicts. The other half is `runs.jsonl`, written automatically by
`pipeline/runlog.py` on every pod run: **if a number can be measured again it
belongs there, if it is a conclusion someone has to remember it belongs here.**

Why this file exists: findings were spread across code comments, commit
messages, `experiments.yaml`, `THIRD_PARTY.md` and three memory files. Each is
right in its place and none is searchable, so the same questions were re-derived
— "what does an hour cost" was answered three times in one day with three
different numbers, and a refuted hypothesis was re-proposed twice.

**Entry format.** CLAIM, then EVIDENCE (numbers, with the date), then VERDICT,
then where it is ENCODED so the code and this file cannot drift apart.
Verdicts: `CONFIRMED` · `REFUTED` · `SUPERSEDED` · `OPEN`.

**Read the noise floor first.** Every number below is meaningless without it.

---

## 0. Noise floor — the bar every result must clear

**CLAIM** Run-to-run variation on an unchanged config bounds what any comparison
can resolve.
**EVIDENCE** 2026-08-01, repeated identical runs: `sim ±0.010`, `mos ±0.007`,
`wer ±0.005`, **`f0st ±0.438`**, `mos_sd ±0.18`, `s/take ±1.7`. The ru band
matches en (`sim 0.007`, `mos 0.008`), so reading ru against the en floor is
justified.
**VERDICT** CONFIRMED.
**CONSEQUENCES** — two prior results were retroactively invalidated by it: a
reference sweep whose top-three spread was 0.018 could not resolve its winner,
and an `f0st` weight added to the tune objective was weighting noise. **`f0st`
cannot rank anything at this sample size.**
**ENCODED** `experiments.yaml` (`noise-floor`, `ru-noise-floor`).

---

## 1. ASR

**CLAIM** Whisper `large-v3` enters a repetition loop that replaces real speech.
**EVIDENCE** 2026-08-02, first production lesson: the tail returned "Він
практично не має запаху." **7 times** over audio measuring **30–77% voiced** —
not silence. Two sentences of technique and the outro were lost, then translated
into 5 languages, synthesized on a GPU pod, mixed and muxed. Every stage
reported success. Found by the user watching the result.
**CAUSE** Two settings, both required: `condition_on_previous_text=True` fed each
wrong segment back as context; `temperature=0.0` with no fallback ladder
disabled Whisper's own rescue (it detects a bad decode via
`compression_ratio_threshold` and retries hotter).
**VERDICT** CONFIRMED and fixed. Old: 62 utterances, 4 with internal repetition.
New: 57 segments, 0. The lost sentences return.
**GOTCHA** It does **not** reproduce on a short clip — transcribing the bad
region alone gives correct text under *both* settings, because the poisoned
context never accumulates. Validate on the whole file.
**ENCODED** `pipeline/asr.py`, `s2_transcribe._warn_on_repetition`,
`tests/test_audit.py`.

**CLAIM** A different ASR model would do better.
**EVIDENCE** 2026-08-03, same audio, same fixed settings: `large-v3` 7/8 domain
terms and 1.5 s of voiced audio uncovered; `large-v3-turbo` 6/8, garbles words;
`large-v2` 6/8 and misses 7.0 s. Qwen3-ASR is SOTA-class but its 30 languages
**exclude Ukrainian**.
**VERDICT** REFUTED for Ukrainian source. No model was uniformly best — v3
dropped one clause both others caught.
**OPEN** Qwen3-ASR **does** support Russian, so it is a live candidate the day a
lesson's source language is Russian. Parakeet TDT (a transducer, structurally
immune to this loop) was dropped by the user; do not re-propose.

---

## 2. TTS — qwen

Scope, set 2026-08-03: **qwen only.** chatterbox/voxcpm comparisons are closed.

**CLAIM** Lower sampling temperature improves consistency.
**EVIDENCE** 2026-08-01. `mos±` 0.443 (temp 0.5) and 0.458 (0.7) vs 0.492
baseline — all inside the 0.18 band. But `mos` rises **monotonically**
2.340/2.347 → 2.408 → 2.429 (+0.085 against a 0.007 band, 12×). Both cost
**~60% more per take** (3.89/3.80 s vs 2.32/2.46).
**VERDICT** Consistency REFUTED; small real quality gain not adopted — that
compute buys more as extra takes.

**CLAIM** `best_of` past 3 is waste.
**EVIDENCE** 2026-08-01, k=1..6: +0.0343 +0.0191 +0.0123 +0.0088 +0.0077 — still
rising at 6, no plateau.
**VERDICT** REFUTED. Caveat kept: max-of-k partly selects for measurement noise,
so the measured gain overstates the real one. Shipping `best_of: 2`.

**CLAIM** A longer reference clip clones better.
**EVIDENCE** 2026-08-03, tune R1 over a length-stratified pool, this lesson:

| ref | length | sim_raw | mos | f0st | score |
|---|---|---|---|---|---|
| ref_04 | **10.0s** | 0.568 | 3.98 | **3.29** | **0.7787** |
| ref_03 | 14.5s | 0.568 | 4.09 | 2.74 | 0.7399 |
| ref_02 | 16.5s | 0.575 | 3.98 | 2.36 | 0.6988 |
| ref_01 | 19.8s | 0.543 | 3.87 | 1.49 | 0.6307 |

**VERDICT** REFUTED, and it SUPERSEDES the earlier "18s beat 12s" result. Score
falls monotonically with length; the driver is `f0st` (1.80 spread against a
±0.438 band, ~4×). Shorter reference → livelier delivery. Matches upstream's
10–15 s guidance.
**IMPORTANT SECOND READING** `sim_raw` is **flat** across all four (0.543–0.575)
and no better than the previous cross-video reference (0.614). **Reference
choice moves liveliness, not identity** — so it is not the fix for weak speaker
similarity.
**ENCODED** `pipeline/prep.py` (`_spread`), `config.yaml` `prep.min_s: 10.0`.

**CLAIM** Generation parameters need tuning.
**EVIDENCE** Upstream's official values are `temperature 0.9 / top_k 50 /
top_p 1.0 / repetition_penalty 1.05`; `tts.qwen_gen_kwargs` is empty, so we
already inherit exactly those.
**VERDICT** Not a lever. Nothing to change.

**CLAIM** ICL (reference audio **+ its transcript**) beats `x_vector_only`.
**EVIDENCE** 2026-08-03, same reference (ref_04), one flag apart, 6 segments:

| | en ctl | en ICL | ru ctl | ru ICL | band |
|---|---|---|---|---|---|
| sim→real | 0.524 | **0.700** | 0.721 | **0.757** | ±0.010 |
| mos | 2.783 | **2.927** | 2.493 | **2.665** | ±0.007 |
| f0st | 2.884 | *1.982* | 3.161 | *2.005* | ±0.438 |
| wer | 0.030 | *0.062* | 0.187 | 0.184 | ±0.005 |

Then **36 blind ratings per language** (shuffled, unlabelled):

| | control | ICL | Δ | 95% CI |
|---|---|---|---|---|
| en | 2.06 | **1.22** | **−0.83** | ±0.60 → DECIDED |
| ru | 2.11 | 2.33 | +0.22 | ±0.78 → no verdict |

**VERDICT** **REFUTED for en by the ear**, despite winning sim by 17× the noise
band and mos by 20×. 15 of 18 ICL clips were rated 1 (unusable). ru is
undecided. `wer` doubling on en was the only metric that told the truth — that
is what accent bleed looks like, exactly as the overlay predicted.
**WHY THE METRICS LIED** see §2.1 — the objective weights `sim` at 0.25 and the
ear weights it at **0.0**. ICL traded the thing nobody hears for the thing they
do.
**ENCODED** `config.exp.icl.yaml`, `config.exp.icl-control.yaml` (one axis
apart — the bake-off's own tune sweep otherwise re-picks the reference and
varies two axes at once).

### 2.1 The take-selection objective does not track the ear

**CLAIM** `qc.eval.weights` (sim .25 / mos .40 / f0 .20 / tempo .15) ranks takes
the way a listener would.
**EVIDENCE** 2026-08-03, first ratings this project has ever had (66 en / 68 ru):

| | en | ru |
|---|---|---|
| current objective, cross-validated Spearman | **+0.022** | +0.134 |
| best-fit, cross-validated | +0.308 | +0.149 |
| best-fit weights | sim **0.0** · mos 0.05 · **f0 0.85** · tempo 0.10 | sim **0.0** · mos 0.75 · f0 0.10 · tempo 0.15 |

**VERDICT** CONFIRMED that it does **not** track the ear — en correlates +0.022,
i.e. selection has been ~random with respect to what the listener prefers.
**`sim` gets weight 0.0 in BOTH languages**, and it is the metric this whole
session chased (reference selection, ICL). The two languages disagree on what
replaces it (en → f0, ru → mos), which at n≈34 each is likely noise.
**NOT ADOPTED YET** en passes 2 of 3 gates; permutation **p 0.059** just misses
0.05. ru fails all three. More ratings are the unblock.
**CONSEQUENCE** Every "improvement" judged by the current composite is suspect,
including the tune R1 reference winner, which was scored by it.

**OPEN, and now the top priority** Overall quality is rated **~2.1/5** by the
ear — *for both variants*, on a scale where 3 = acceptable. The ICL question was
a distraction from this: the shipped dubs are below acceptable to their owner,
and the selection objective that picks every shipped take is uncorrelated with
his judgement.

---

## 3. Cost

**CLAIM** RAM and vCPU are chosen and billed.
**EVIDENCE** `create_pod` sends no memory or vCPU field. Two consecutive runs:
`$0.25/h, 21 vCPU, 83 GB` and `$0.25/h, 16 vCPU, 62 GB` — different RAM,
identical price.
**VERDICT** REFUTED. RunPod bundles vCPU/RAM per GPU tier and bills per GPU.
*(Asked twice. Answer with the two log lines, not reassurance.)*

**CLAIM** `allowed_cuda_versions` costs ~12% more per hour.
**EVIDENCE** $0.25 → $0.28 on consecutive runs, then $0.25 again **with the same
filter**.
**VERDICT** REFUTED — that was host variance, n=1 each. The filter's real cost is
unmeasured and may be zero.

**CLAIM** Cost per hour of video, 5 languages.
**EVIDENCE** measured, not estimated — see `runs.jsonl`. 2026-08-02: 8.13 min
video, 5 langs, 38.8 min pod = **$0.161**. 2026-08-03 (46 utterances + 61
pre-made variants): 41.1 min = **$0.192**, 6.9 min/language vs 6.2 before.
**VERDICT** ~**$1.10–1.22 per video-hour** for 5 languages, ≈$0.22/language.
Bootstrap is ~7 min FIXED — ~20% of an 8-minute lesson, under 4% of an hour.
**DERIVE IT, DON'T QUOTE IT** — `runlog.cost_per_video_hour()`.

**CLAIM** Scoring is a minor part of the bill.
**EVIDENCE** PHASE breakdown 2026-08-01: synth 60% / **scoring 40%**, while the
GPU idles at ~7%. Scoring runs on a rented vCPU.
**VERDICT** REFUTED — this is the largest untaken lever. `qc.metrics_device`
exists and defaults to `cpu` deliberately: every stored score was produced on
CPU, so flipping it needs a measured diff first (`qc/metrics.py set_device()`).

**OPEN** Spot (`interruptible: true`) is 2–3×, and `synth_best_of` checkpoints
per segment so preemption resumes from cache. Not flipped: this project lost
four runs to spot reclamation on COMMUNITY. Needs one measured run.

---

## 4. Infrastructure lessons

**`--reuse` could never skip setup.** The probe was hardcoded to
`bakeoff.engines` and got `[]` for every other task; `_verify_ready([])` is False
by design. `course.py` phase B — one pod, twenty videos — re-bootstrapped per
video, which is exactly the cost it exists to avoid. Fixed via
`engines_for_task`.

**A reused pod counted down on the first run's clock.** The watchdog was armed
once at provision while each `--reuse` recomputes a deadline locally. Now
re-armable via a pid file.

**s5 must never need a GPU.** `course.py`'s phase C is local and free by design,
but `s5_fit` synthesizes fit variants on demand — fine while the engine was
`edge` (a network service), fatal with qwen. s4 pre-makes them (gated at
`VARIANT_MARGIN`) and s5's ladder is lazy. **Measured 2026-08-03: 61 variants
pre-made, 0 used** — the gate at `soft × 0.85` fires for any segment over ~90% of
its slot. Do not tighten on one run; the run where insurance is needed is the one
that strands phase C.

**Never trust a container tag.** `runpod/containers#114` shipped a "torch280"
image containing torch 2.4.1. `remote setup-check` prints the resolved torch and
CUDA for ~$0.015 — run it after any image or pin change.

**The wheel must match the OLDEST admitted driver.** Plain PyPI serves torch 2.11
as `+cu130`, needing a 13.0 driver, while the filter also admits 12.8/12.9. One
pod passed only because it drew a 13.0 host. Pin the index to `cu128`.

**Comments that assert unverified behaviour are the recurring bug.** Four this
week: `torchaudio.save` (never called), "the REST API cannot request a driver
version" (it can — `allowedCudaVersions`), `tests/test_deps.py` (did not exist),
`-disposition:s:0 0` (inert — MP4 forces default on a lone subtitle track). Each
was one command from being checked.

---

## 5. Open questions, ranked

1. **ICL vs x_vector** — the only identified lever on speaker similarity. *(in flight)*
2. **Spot instances** — 2–3× cost, needs one measured run.
3. **`metrics_device: cuda`** — up to ~40% of s4, needs a score diff.
4. **Eval calibration** — `refit` says KEEP CURRENT WEIGHTS (rho 0.237 vs a 0.30
   floor, p 0.078). Until this clears, the harness cannot arbitrate quality
   changes and every lever above is judged by ear. **Needs more human ratings.**
5. **Cross-model ASR consensus** — the only technique that catches silent
   omissions (v3 dropped a clause v2 and turbo both caught). Proposed, not built.
