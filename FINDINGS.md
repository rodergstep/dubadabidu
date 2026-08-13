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

## CLOSED — do not re-attempt without NEW evidence

Read this before spending a pod. Every line was measured, not argued, and the
section exists because this project has re-derived the same answers more than
once. "New evidence" means an upstream capability that did not exist when the
entry was written — not a fresh idea about how to try the same thing again.

### Russian lexical stress — CLOSED 2026-08-09, shipping as-is
The product decision: **ru ships with the defect.** ~29% of takes carry at least
one stress error; the other four languages have no equivalent problem. Everything
below was tried and failed, each for a different reason:

| attempt | outcome | § |
|---|---|---|
| RUAccent marks in the target text | 97% unusable by ear, WER 0.068 → 0.811 | 2.1b |
| RUAccent marks + ICL-grounded reference | still destroyed, WER 0.621 | 2.1bb |
| `instruct` pronunciation control | does not exist on the cloning path | 2.1c |
| Engine swap (OmniVoice / XTTS / Silero / F5) | none both clones AND controls stress | 2.1c |
| chatterbox for ru only | ruled out by the user | — |
| Detector: acoustic prominence | 29% self-disagreement | 2.1f |
| Detector: wav2vec2 phonemes | model does not transcribe vowel reduction | 2.1g |
| Detector: MFA variant alignment | AUC 0.641, and the RUAccent oracle itself errs | 2.1h |
| Selection: 5-take consensus | ANTI-correlated — prefers the systematic error | 2.1i, 2.1j |
| Lexical avoidance from a human-marked table | **PARTIAL WIN** — 2-3 of 9 marked words | 2.1k |

**The reason it is closed, and it is structural (§2.1j):** the errors are partly
SYSTEMATIC. For some words qwen is reliably wrong, so the majority placement is
wrong and no selection rule can reach them. `best_of`, vetoes and consensus can
only touch the smaller stochastic band.
**PARTIALLY REOPENED 2026-08-11 (§2.1k), and only partially.** Systematic also
means ENUMERABLE, so per-word control can be built OUTSIDE the engine: the
listener marks the word on the review page, it accumulates in
`stress_lexicon_ru.json`, and s3 is asked to prefer a synonym. Measured on the
first real pass it reaches **2-3 of 9** marked words — real, free, and not a
route to a clean ru track. **ru still ships with the defect.**
**What would reopen it fully:** per-word pronunciation control in Qwen3-TTS
(open upstream request, no roadmap), or a cloning TTS with a stress-aware
frontend. Neither is a thing we can build — but note §2.1k: "we cannot build
per-word control" was too strong, because the INPUT is ours to change.
**Untested and next in line:** reduction respelling in ordinary Cyrillic, which
would reach 4 of 9 (§2.1k.6). Not the refuted U+0301 route (§2.1b/2.1bb).

### Also closed
| question | verdict | § |
|---|---|---|
| Spot instances | 6 pods, none ever reached synthesis | 3 |
| Reference choice as a quality lever | 21–21 across 42 blind judgements, p = 1.000 | 2.2 |
| ICL vs x_vector (en) | refuted by ear, 15/18 unusable | 2 |
| Per-video references | worse than a curated one | 2.2 |
| Longer reference clips | score falls monotonically with length | 2 |
| Sampling-temperature tuning | gain real but cheaper spent on takes | 2 |
| Generation params | already at upstream defaults | 2 |
| 0.6B model | costs identity for no reliable gain | config.gpu.yaml |
| Alternative ASR for Ukrainian | no model uniformly better | 1 |
| `sim` as a quality predictor | noise in both languages, weight now 0 | 2.1e |
| Mixed rating axes as refit's problem | refuted by permutation test | 2.1d |
| "More ratings will fix refit" | no — the audio is uniformly good now | 2.0 |

### Method rules earned the hard way
- **Compute the binomial before believing a small blind test.** A 2-of-3 has a
  two-sided p of 1.00 and drove a config change (§2.2).
- **Matching base rates is not per-item agreement.** 33% vs 29% looked like
  corroboration and was coincidence (§2.1i).
- **A control that cannot fail is not a control.** A noise-based null said 0%
  where a timing-based null said 29% (§2.1f); a storage test compared page
  shapes when the bug was page content.
- **Validate a detector against labels BEFORE wiring it in.** Four have failed
  that gate; none reached production or cost a pod beyond its own test.
- **A knob that looks load-bearing may not be.** `qc.eval.weights` did not
  select takes; production was routed to chatterbox for a week (§2.1e).

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

### 2.0 Post-processing costs nothing, and the engine work is DONE

**CLAIM** The listener's "almost perfect quality" verdict describes what ships.
Not obvious: every comparison page in this project used RAW bake-off takes,
straight out of s4, while the product goes through s5 placement, the s6 master
chain (HPF -> compressor -> limiter) and a loudnorm pass. None of that had ever
been A/B'd — the quality verdicts were all measured upstream of it.
**EVIDENCE** 2026-08-09. Raw take vs the mastered vocals actually in the shipped
video, same sentence, 12 pairs, loudness-matched (mastering raises level and
louder wins blind tests regardless of quality), blind:

| | best | unusable |
|---|---|---|
| shipped (mastered) | 5 | 1 |
| raw take | 5 | 0 |

**5-5, two-sided p = 1.000.**
**VERDICT** CONFIRMED — post-processing is transparent. The product is as good as
the audio being rated. Compared against the MASTERED VOCALS, not `dub_ru.m4a`,
on purpose: the full mix carries the background bed, which is instantly
recognisable and would turn a blind test into "do you like background music".
**THE BIGGER READ, and it changes what is worth working on.** The listener now
rates the audio near-perfect apart from ru stress. On 2026-08-03 the shipped en
ratings were 1:1 2:2 3:18 4:20 5:5 — mean ~3.5, with real 1s and 2s. Monotony,
fidelity and pacing dominated every earlier session and are now closed.
**CONSEQUENCE FOR CALIBRATION:** this is also why `refit` cannot clear its gates,
and more ratings will NOT fix it. When the audio is uniformly good there is
almost no rating variance left to predict, and what remains is driven by a
defect no metric measures (§2.1f). A flat objective against uniformly good audio
is a success, not a failure to calibrate.
**CONSEQUENCE FOR PRIORITIES:** one defect remains — ru stress, one language,
~29% of takes, with a manual fix that works today. Everything else is polishing
audio the listener already calls near-perfect.

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

### 2.1b RU lexical stress: RUAccent marks DESTROY qwen's output

**CLAIM** Marking Russian lexical stress with RUAccent's combining acutes fixes
the wrong-stress the listener has reported three times (2026-08-03, twice
2026-08-09). It worked for chatterbox, which was trained on the marks.
**EVIDENCE** 2026-08-09, first time this A/B has ever actually run — the
2026-08-03 attempt died on the `transformers` conflict before synthesis. Both
arms, all 46 segments, same reference, one axis:

| metric | stress | control | Δ | noise band |
|---|---|---|---|---|
| sim→real | 0.690 | 0.687 | +0.003 | ±0.010 — inside |
| mos | 2.051 | 2.394 | −0.343 | ±0.008 |
| f0st | 2.829 | 2.874 | −0.045 | ±0.438 — inside |
| **wer** | **0.811** | **0.068** | **+0.743** | ±0.005 |
| pace | 1.196 | 0.864 | +0.332 | — |
| s/take | 4.05 | 2.34 | +1.71 | ±1.7 |

**VERDICT** **REFUTED, and actively harmful.** Back-transcription WER goes from
7% to 81%: the audio no longer contains the words. `pace` and `s/take` corroborate
— it generates far more audio for identical text, the signature of garbled or
spelled-out output. qwen was not trained on U+0301 and reads it as content.
**MEASUREMENT VALIDITY, checked before believing it:** the marks could have
inflated WER cosmetically by surviving normalization. They do not —
`backcheck._norm`'s `[^\w\s]` strips U+0301 (combining marks are not `\w`), and
accented-vs-plain text scores WER 0.0 after normalization. The 0.811 is real.
**ENCODED** `tts.ru_stress` stays `false`. `config.exp.ru-stress.yaml` keeps the
recipe so nobody re-runs it from scratch to re-learn this.
**CONFIRMED BY EAR 2026-08-09**, 30 sentences x 4 clips, blind:

| | control | stress |
|---|---|---|
| best | **27** | 2 |
| unusable | 6/60 = **10%** | 58/60 = **97%** |

two-sided binomial **p = 0.00000**. Metrics and ear agree for once, and they
agree the marks are catastrophic.

**THE COMPLAINT IS STILL OPEN.** The listener still hears wrong stress in the
control arm: 4 of 30 groups had nothing shippable. No metric here sees it —
`sim`, `mos` and `f0st` are all blind to which syllable is stressed, which is
also why `refit` cannot calibrate on ru.

### 2.1bb Stress marks fail WITH in-context grounding too — the route is closed

**CLAIM** The refuted stress test was unfair: it ran `qwen_x_vector_only: true`,
so the model got acutes in the target having never seen a mark paired with the
audio it denotes. Upstream [#185](https://github.com/QwenLM/Qwen3-TTS/discussions/185)
says marking the REFERENCE transcript teaches the model to read them.
**EVIDENCE** 2026-08-09, three arms, 46 ru segments, reference held constant
(`sketch_ref_08.trim12s`, transcript taken from the trimmed clip itself so it
cannot mismatch, marked with lang-uk/ukrainian-word-stress, MIT):

| arm | sim | mos | f0st | **wer** | pace |
|---|---|---|---|---|---|
| A production — x_vector, unmarked | 0.687 | 2.394 | **2.874** | **0.068** | 0.864 |
| C ICL, unmarked | **0.722** | **2.515** | 1.931 | 0.080 | 0.803 |
| B ICL, marked ref + marked target | 0.718 | 2.326 | 2.059 | **0.621** | 1.046 |

**VERDICT (marks) REFUTED, definitively.** B vs C holds ICL constant: WER
0.080 → 0.621. Grounding helps against the ungrounded run (0.811 → 0.621) — the
mechanism is real — but the audio is still destroyed. Marks fail with grounding
and without it, so **text-side stress marking is closed for qwen**, not merely
unpromising.
**VERDICT (ICL) OPEN, and it SPLITS BY SENTENCE LENGTH.** C vs A is the best
identity measured here (sim +0.035, 3.5x the noise band; mos +0.121, 15x) while
f0st falls 2.874 → 1.931.

Blind ear, axis printed on the page ("which is LESS flat and more expressive"),
run in two halves — and the halves disagree:

| round | sentences | ICL | production | ICL unusable |
|---|---|---|---|---|
| 1 | the 12 LONGEST | **9** | 3 | — |
| 2 | the next 13, shorter | 2 | **9** | **6** vs 2 |
| combined | 25 | 11 | 12 | — |

**Fisher exact p = 0.0123** between rounds. The combined 11-12 (p = 1.000) is
averaging over a real split, not measuring sameness — quoting it alone would be
the mistake. ICL appears to help on long sentences and hurt on short ones, which
is what accent bleed should look like: less room to recover in a short clip.
**DO NOT ADOPT** on this. The next question is whether the effect is length or
something correlated with it.

**A CLAIM I MADE AND RETRACTED WITHIN THE HOUR.** After round 1 alone I reported
that f0st was "inverted against the ear" and the user reasonably asked me to fix
or remove it. Round 2 reversed the sign, and on the full corpus f0st is the
STRONGEST feature we have (below). Twelve sentences was not enough to claim an
inversion in a metric whose measured band is ±0.438. The retraction cost nothing
only because the weights had not been touched yet.

### 2.1e What the 226 ratings actually say about each metric

**CLAIM** The take-selection features all carry some signal.
**EVIDENCE** 2026-08-09, pooled per language, permutation-tested against 5000
shuffles of the ratings:

| feature | en rho | en p | ru rho | ru p |
|---|---|---|---|---|
| **f0st** | **+0.290** | **0.0016** | **+0.190** | **0.0386** |
| mos | −0.016 | 0.87 | **+0.295** | **0.0014** |
| sim | −0.075 | 0.42 | −0.136 | 0.15 |

**VERDICT** **`sim` is REFUTED as a predictor** — indistinguishable from noise in
both languages, and NEGATIVE in sign in both. It is the metric this project
chased hardest: three reference sweeps, the ICL experiment, and a blind test
where the ear picked the lowest-sim candidate. Five failures.
**`f0st` is CONFIRMED** as the only feature significant in both languages.
**NOT A CONTRADICTION with `qc/bakeoff.py`, which excludes f0st from its score.**
The bake-off ranks whole RUNS, where f0st's run-to-run band (±0.438) swamps any
difference. `qc.eval.weights` ranks TAKES inside one segment. Same metric,
different comparison, different noise — both exclusions are correct in place.
**ENCODED** `config.yaml` `qc.eval.weights: {sim: 0.0, mos: 0.55, f0: 0.30,
tempo: 0.15}`, guarded by `tests/test_deps.py`. This is a targeted removal of a
measured non-signal, NOT an adopted refit — refit still says KEEP CURRENT
WEIGHTS and its fits remain unstable at this n (§2.1d).

### 2.1f RU stress errors are STOCHASTIC per take — the fix is selection

**CLAIM** qwen mis-stresses a given word the same way every time, so re-rolling
cannot escape it and `best_of` is the wrong lever. (The premise the veto idea
rests on; my acoustic detector failed to settle it — §2.1b — at 29% self-flip.)
**EVIDENCE** 2026-08-09. Settled with the listener's ear instead of an
instrument: 14 ru sentences, TWO takes of the SAME config, single question —
does the stress differ?

| | count |
|---|---|
| exactly ONE take had a stress error | **8** |
| **BOTH** takes had a stress error | **0** |
| no error heard | 6 |

Determinism predicts every group with an error has BOTH takes bad: predicted 8,
**observed 0**.
**VERDICT** **REFUTED — errors are stochastic per take.** 8 of 28 takes (29%)
carry at least one stress error, and a correct take existed for every affected
segment. **Every lesson already contains correct-stress audio; we discard it at
random because the objective cannot see stress.**
**QUANTIFIED PAYOFF** with a working stress-aware selector, assuming
independence at p=0.29: `best_of=2` → **8.2%** of segments still wrong,
`best_of=3` → **2.3%**, `best_of=4` → 0.7%. Against ~29% today.
**THIS IS THE HIGHEST-VALUE OPEN ITEM IN THE PROJECT.** It converts the largest
known quality defect from "no solution exists" (§2.1c) into "build the
detector", with the payoff measured rather than hoped for.

**BUT NO EXISTING METRIC DETECTS IT** — `qc/stress_wer.py`, the 28 rated takes:

| feature | error-take mean | clean mean | AUC | p |
|---|---|---|---|---|
| wer | 0.090 | 0.089 | 0.647 | 0.232 |
| mos | **2.232** | **1.961** | 0.725 | 0.067 |

`wer` does not separate them — the means are identical, so a mis-stressed
Russian word still back-transcribes to the right word. Whisper is robust to
stress, which is exactly what makes it useless here.
**AND `mos` POINTS THE WRONG WAY.** Stress-error takes score *higher* (2.232 vs
1.961). `mos` now carries **0.55** of the take-selection weight (§2.1e), so
selection may be actively biased TOWARD mis-stressed takes — which would make
the shipped rate worse than the 29% a coin flip would give. Marginal (p=0.067,
n=8 vs 20) and NOT acted on, but it is the first hypothesis to test with more
ratings, and it would explain the listener's complaints better than chance does.
**NEXT** phoneme-level forced alignment (wav2vec2/MMS CTC) for real vowel
boundaries, with RUAccent as the oracle for which vowel *should* carry stress.
Validate against these 28 labelled takes — a free, already-collected test set —
before spending a pod.

### 2.1g Phoneme-based stress detector: BUILT, VALIDATED, FAILED

**CLAIM** Russian reduces unstressed vowels (/o a/ → [ɐ ə]), so a multilingual
phoneme recogniser reports stress position directly: the unreduced vowel is the
stressed one. That reads a CATEGORICAL symbol instead of ranking a continuous
prominence, which is why the earlier acoustic detector self-flipped 29% (§2.1f).
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` has both vowel sets in its vocabulary.
**EVIDENCE** `qc/stress_detect.py`, 2026-08-09, no pod.
1. **The premise is empirically false.** Over 320 vowels from 8 real takes the
   model emits **99% full vowels, 1% reduced**. It transcribes Russian broadly,
   not narrowly. Having the symbols in the vocabulary is not the same as using
   them, and only measuring showed the difference.
2. **One sub-signal survives: akanye.** Unstressed orthographic о is realised
   [a], so a full [o] marks a stressed о. At word level, with ±120 ms slice
   padding: stressed-о words produce [o] **89%** of the time, unstressed-о words
   **26%** — separation **+0.63**, a real effect.
3. **Slice padding matters as much as the model.** [o] found on a stressed о:
   50% at no padding, 68% at ±60 ms, **86% at ±120 ms**. Whisper word bounds
   clip short Russian words and lose the very vowel being judged — the same
   localisation weakness that killed the prominence detector.
4. **It still fails the gate.** Against the 28 human-labelled takes:
   `mismatches` **AUC 0.562**, `rate` AUC 0.539 — below the **0.647** that
   back-transcription WER already gives for free. **4 of 8 error takes scored
   ZERO detected mismatches.**
**VERDICT** **REFUTED — not wired in.** A real word-level signal did not survive
aggregation to take level. Two reasons, both structural: coverage (only words
containing о can be judged, 3-8 per take, and the listener's error is often on a
word without one) and precision (a 26% word-level false-positive rate over ~6
words produces ~1.5 spurious mismatches per take, swamping the single real one).
**WHAT WOULD BE NEEDED** a recogniser that transcribes Russian NARROWLY, with
reduction — which this one demonstrably does not — or a Russian-specific phoneme
model, plus real forced alignment rather than padded Whisper bounds.
**PROCESS NOTE, the point of the exercise:** the gate ran BEFORE the detector was
wired into take selection and before any pod was spent. The previous detector
was built and used before it was ever checked against a label, and it silently
measured itself. Cost here: a few hours and no money, for a clean negative.

### 2.1h MFA variant-alignment detector: BUILT, VALIDATED, FAILED (and why)

**CLAIM** Pronunciation-variant forced alignment detects wrong stress. The MFA
Russian dictionary (v3.1.0, CC BY 4.0) transcribes narrowly — stress IS vowel
quality, and it already ships `замок` twice (`z̪ ɐ m o k` / `z̪ a m ə k`) on
exactly that principle. Offer the aligner every stress placement, let its
acoustic model pick, read the winner's vowels back.
**EVIDENCE** `qc/stress_mfa.py` + `qc/stress_variants.py`, 2026-08-09, no pod.
The machinery works: **14.2 words resolved per take** (against 3-8 for the
wav2vec2 route), and spot checks are right — `пото́м`→1, `о́хра`→0, `жёлтая`→0.
Against the 16 labelled takes: `rate` **AUC 0.641**, `mismatches` AUC 0.578 —
under the **0.647** that WER already gives free. Within-pair (the actual use
case, picking the better of two takes): **3 correct, 2 wrong, 3 ties**.
**VERDICT** **REFUTED — not wired in.** Chance-level discrimination.
**WHY IT FAILS, which is the useful part.** 12 words were flagged in BOTH takes
of a pair. Errors are stochastic per take (§2.1f), so a both-take mismatch is
not a TTS error — it is the DETECTOR AND ORACLE disagreeing systematically:
- **RUAccent is wrong sometimes.** It marks `перед` on slot 1; the preposition
  is `пе́ред`, slot 0. The oracle this whole family of approaches depends on is
  not itself reliable.
- **Some words have no single right answer.** `цвета` is `цве́та` (gen. sg.) or
  `цвета́` (nom. pl.), both valid; a context-free oracle cannot arbitrate, and
  disagreement there is noise, not signal.
**CONSEQUENCE FOR THE WHOLE APPROACH** every oracle-based stress detector
inherits the oracle's error rate, and here that rate is comparable to the defect
being measured. Improving the acoustic side cannot fix that.
**PROCESS NOTE** the gate ran on 16 takes. Iterating on the detector until it
passed would be fitting to the validation set, so it was run ONCE per design
change and the design was not tuned against the labels. Three detectors have now
failed this gate before shipping; none reached take selection or cost a pod.
**RETAINED** the code stays, unwired, with its gate — so the negative is
reproducible and nobody rebuilds it. MFA install is a throwaway micromamba env,
NOT a project dependency.

### 2.1i Consensus take-selection: REFUTED, and a base-rate trap

**CLAIM** Errors are stochastic per take (§2.1f), so across K takes the MODAL
stress placement per word is right, and the take agreeing with it most often is
the clean one. No oracle (RUAccent never enters, so §2.1h's cap does not apply),
no human, and systematic MFA misreadings cancel because they hit every take
identically.
**EVIDENCE** 2026-08-09, one pod (~$0.20): 46 ru segments x 5 takes = 230, all
MFA-aligned. Consensus decided 45 of 46 segments and the top take had **zero
deviations in every one** (min 0, max 0, mean 0.00). Spread between takes was
real, e.g. u0007 `[0, 0, 1, 2, 4]` over 15 words.
Blind ear, consensus-pick vs the MOST deviant take, 12 sentences, stress only:

| | best | unusable |
|---|---|---|
| consensus-pick | 0 | **4** |
| most-deviant | **1** | 3 |
| both unusable | | 3 of 12 |

**VERDICT** **REFUTED.** The consensus pick was marked unusable MORE often than
the take the algorithm ranked worst. MFA deviations do not correspond to the
errors the listener hears.
**THE TRAP, and I walked into it.** 33% of takes carried >=1 MFA deviation
against the 29% of takes the listener marked with >=1 stress error, and I
reported that as corroboration BEFORE the ear had spoken. **Two methods agreeing
on a RATE say nothing about whether they flag the same INSTANCES.** They did not.
A matching base rate is the cheapest possible coincidence — with a defect this
common, any detector firing at roughly the right frequency will match it. Check
per-item overlap, or claim nothing.
**IT ALSO WEAKENS §2.1f.** 3 of 12 pairs here had BOTH takes unusable, against
0 of 14 in the test that established "a correct take always exists". Not fatal —
these are 2 of 5 takes chosen by a selector now known to be wrong, not 2
arbitrary takes — but that premise carried this entire line of work and is
softer than it looked.
**FOUR detectors have now failed the same gate:** acoustic prominence (§2.1f),
wav2vec2 phonemes (§2.1g), MFA variant alignment (§2.1h), consensus (here). Each
failed for a different reason. Automated Russian stress detection is CLOSED
until something changes upstream — a stress-aware TTS, or a Russian phoneme model
that transcribes reduction.
**ENCODED** `qc/stress_consensus.py` + `config.exp.ru-consensus.yaml`, unwired,
kept so the negative is reproducible.

### 2.1j RU stress errors are partly SYSTEMATIC — selection cannot fix them

**CLAIM** (§2.1f, which this supersedes) Errors are stochastic per take, so a
correct take always exists and the problem is one of selection.
**EVIDENCE** 2026-08-09, from the listener's own two rating rounds plus the
5-take alignments.

*The rate did not change; the CLUSTERING did.* He reported the latest batch as
"much better stressing" — and at sentence level he is right, but the per-take
rate is identical:

| | earlier (2 takes) | latest (5-take pool) |
|---|---|---|
| takes with a stress error | **29%** | **29%** |
| sentences fully clean | 43% | **67%** |
| sentences with BOTH presented takes bad | **0** | **3** |

Same defect rate, concentrated into fewer sentences. More fully-clean sentences
is exactly what "sounds much better" means, and it is a clustering effect, not
an improvement.

*And the clustered ones are systematic.* All three BOTH-BAD segments had a
ZERO-deviation take available — it was the consensus pick, and he marked it bad:

```
u0015  BOTH BAD  deviations [0, 1, 1, 2, 3]
u0025  BOTH BAD  deviations [0, 0, 1, 1, 2]
u0030  BOTH BAD  deviations [0, 0, 0, 2, 2]
```

**VERDICT** **§2.1f is SUPERSEDED.** Errors are not uniformly stochastic. For a
subset of words qwen is consistently wrong, so the MAJORITY placement is wrong,
the take matching it scores zero deviations, and consensus selects it with
confidence — while any take that got the word RIGHT deviates from the mode and is
flagged as the bad one.
**CONSENSUS IS THEREFORE ANTI-CORRELATED on exactly the segments that matter**,
which is why the blind test came back consensus 4 unusable vs most-deviant 3
rather than a tie. It is not blind to these errors; it prefers them.
**THE REAL SHAPE OF THE DEFECT.** Three populations: words qwen reliably gets
right (most), words it reliably gets WRONG (the clustered failures), and a
smaller stochastic band. `best_of` and every selection rule can only touch the
third.
**CONSEQUENCE** No selection-based fix exists, and that closes the last route
that did not require changing the engine or the input. What remains:
per-word pronunciation control (qwen has none — §2.1c), a stress-aware engine,
or accepting that ru ships weaker than the other four languages.
**THE EARLIER 0-of-14 WAS AN ARTEFACT OF SAMPLING TWO TAKES.** With two draws a
systematically-wrong word usually produces two wrong takes and gets marked as one
bad take against one unrated — it never looks like "both bad". Five takes made it
visible. A premise measured at the smallest possible K carried four experiments.

### 2.1k Per-word avoidance from a human-marked table — WORKS, and is small

**CLAIM** §2.1j treats "partly SYSTEMATIC" as terminal. It is the opposite:
systematic means deterministic, and deterministic means ENUMERABLE. Every
attempt above tried to solve stress IN GENERAL — a detector that works on any
word, conditioning that works on any word. A painting course has a bounded
lexicon, so the words qwen reliably mis-stresses are a finite set, and a word
that is reliably wrong needs fixing ONCE rather than detecting every take. The
listener is the only labeller that ever worked; the review page rated SEGMENTS
and discarded WHICH WORD every time it was used.

**EVIDENCE** First real word-marking pass, 2026-08-11, no pod.
1. **The label is cheap and the rate matches.** 46 ru segments, 13 marks over
   10 segments (22%) — consistent with the 29%-of-takes rate in §2.1j. 9
   distinct words: `цвета` 3x, `выдавливаю` 2x, `натюрморте` 2x, then
   `белилам`, `лиловой`, `лиловую`, `стронциановая`, `тёмным`, `хватит`.
   Marginal cost is one click on audio already playing.
2. **Two of the three BOTH-BAD segments from §2.1j are in it** — u0015
   (`натюрморте`) and u0030 (`цвета`, `белилам`). The systematic population is
   now named rather than inferred.
3. **Avoidance works, and covers less than estimated.** Live A/B on the real
   endpoint, four segments, same prompt with and without the block. I predicted
   synonyms for 6-7 of 9; measured **2-3**. `цвета` -> `тона`/`оттенки` (3/3
   occurrences) and `выдавливаю` -> `выжимаю` are reliable. `натюрморте`,
   `лиловую`, `белилам`, `стронциановая` are KEPT, correctly, under the "no
   natural alternative" clause — a still-life course cannot route around
   `натюрморт` and `белила` is a specific pigment.
4. **Three failure modes on the first pass, all now forbidden in the prompt.**
   It INFLECTED instead of substituting (`натюрморте` -> `натюрморта`,
   `белилам` -> `белилами`) — another case of the same word carries the same
   stress, so that is the defect wearing a different ending. It DELETED the
   noun ("разместить самые светлые цвета" -> "разместить самые светлые"), a
   content loss the adequacy judge could pass because nothing was
   mistranslated. It rewrote words never on the list (`участков` -> `зон`).
5. **The `ё` lever has ZERO applicable cases.** Only `тёмным` contains `ё`, it
   already has it, and it was marked anyway — so `ё` is not a reliable cue AND
   no marked word is missing one. (`Жёлтые` and `тёмных` in the same sentence
   went unmarked, so it is honoured sometimes.)
6. **Reduction respelling would reach 4 of 9** — only words with an unstressed
   о/е to reduce: `цвета`, `натюрморте`, `белилам`, `стронциановая`.
   `выдавливаю`, `лиловую`, `лиловой`, `тёмным`, `хватит` have nothing to
   respell. UNTESTED; needs the same gate the four detectors got.

**VERDICT** **CONFIRMED, and smaller than claimed.** Avoidance is real, free
and needs no capability from anyone — but it is a partial fix reaching roughly
a quarter of marked words, not the route to a clean ru track. §2.1j's "no fix
exists without changing the engine or the input" is now wrong in its second
half: the INPUT is ours to change, and changing it helps a little.

**STILL OPEN, and it gates everything downstream:** nobody has checked by ear
whether the listener prefers a correctly-stressed synonym to a mis-stressed
exact word. Avoidance trades a known defect for an unrequested word choice, and
that trade is assumed, not measured. Judge it on the first video that ships
with the block active.

**WHAT IT DOES NOT REACH** the four words with no natural alternative. Those
need respelling (4 of 9, untested) or per-word engine control (§2.1c, absent).

**PROCESS NOTE** the coverage estimate was wrong by 2-3x and only an A/B caught
it — the same shape as the four detectors, where the gate ran before anything
was wired in. The estimate cost nothing because it was measured before the
feature was trusted, not after it shipped.

**ENCODED** `qc/stress_words.py` (tokenizer, lexicon key, avoid-list loader),
word marking on the review page for `STRESS_LANGS`, ingest into
`stress_lexicon_<lang>.json` via `dubadabidu verdicts`, and
`translation.avoid_mis_stressed` / `avoid_min_marks` in `config.yaml`.

### 2.1m SEGMENT LENGTH confounds every metric except f0 — and it fooled me

**CLAIM** (mine, 2026-08-11, acted on before it was checked) `qc_mos_min`
predicts the listener where whole-take `qc_mos` does not, so the composite
should read the windowed minimum. Evidence offered at the time: +0.226 vs
+0.005 on 140 rated rows, and the composite going from **-0.205 to +0.364**
against 46 accept/reject verdicts. The switch was made and everything rescored.

**EVIDENCE THAT IT WAS A CONFOUND** 2026-08-13, 46 rated ru segments, no pod.
`mos_min_window` takes a MINIMUM over sliding windows, so a longer segment has
more windows and a lower expected minimum by arithmetic alone:

| | vs segment duration |
|---|---|
| `qc_mos_min` | **-0.719** |
| `qc_mos` | **+0.854** |
| `qc_sim_cal` | **+0.548** |
| `qc_f0st` | **+0.018** |
| the human RATING | **-0.304** (and **-0.500** vs accept/reject) |

The listener rates long segments worse, so ANY length-correlated feature scores
a respectable rank correlation without measuring quality. Within an 8-16 s band
`qc_mos_min` vs rating **reverses to -0.276** (n=11).

**CONFIRMED INDEPENDENTLY BY EAR, and this is what triggered the check.** The
five worst-scoring segments across en/fr/es/de — every one with whole-take MOS
4.5-4.7 and `mos_min` 1.5-2.3, i.e. the new metric claiming a glitch — were
played blind: *"mostly sound clean, good quality."* The metric was flagging
LENGTH, not defects.

**VERDICT** **REFUTED.** Neither MOS variant measures quality here. Whole-take
tracks length upward and so anti-correlates with the ear; the window tracks it
downward and so appears to agree. The 2026-08-11 switch traded one length
artefact for another with a more convenient sign. `f0st` is the only clean
feature (+0.018 with duration, +0.247 with rating).

**WHY IT SURVIVED THE GATES** the permutation test shuffles RATINGS, so it
cannot break a relationship that runs through a third variable present on both
sides. Five adoption gates passed a proposal built on this.

**THE PROJECT ALREADY KNEW.** `qc/compare.py`'s header says it: *"An absolute
score on differing content confounds two variables — how good the take is, and
how hard that sentence was. A 1.4 s fragment and a 14.9 s sentence are not on
the same scale."* It was built to remove exactly this and has never been used
for a rating round. Its stated motivation — refit's cross-validated rho falling
to +0.119 while in-sample reached +0.293 — is the same effect.

**ENCODED** `qc.verdicts` now records `dur` on every row (the column whose
absence made this uncheckable), and `refit` gained a
`features measure quality, not length` gate that fails when any feature runs
|rho| >= 0.45 against duration. It currently fails, correctly.

**DO NOT** move the weights again on absolute per-segment ratings. The next
rating round has to come from `qc/compare.py`, which holds the sentence
constant. **NOT DONE:** the composite still reads `qc_mos_min` — reverting to
`qc_mos` would restore a feature that is anti-correlated with the ear, so
neither state is defensible and a third objective change on confounded data
would repeat the mistake.

### 2.1c No zero-shot cloning TTS handles Russian stress (survey, 2026-08-09)

**CLAIM** Some other open model solves this and we can swap engines for ru via
`tts.engine_by_lang`.
**EVIDENCE** Web survey:
- **Qwen3-TTS** — trained without stress marks; upstream discussions
  [#185](https://github.com/QwenLM/Qwen3-TTS/discussions/185) and
  [#53](https://github.com/QwenLM/Qwen3-TTS/discussions/53) report exactly our
  result ("manual workarounds produce worse results than unstressed text"). No
  maintainer response, no roadmap, open since April 2026.
- **OmniVoice** (k2-fsa, Apache-2.0, 600+ langs, zero-shot cloning) — U+0301 is
  **ignored**; plus signs, SSML phoneme tags and full ARPAbet all fail
  ([issue #65](https://github.com/k2-fsa/OmniVoice/issues/65), closed as
  duplicate). Reporter's partial workaround — ARPAbet for homographs only —
  imports an English accent on those words.
- **XTTS-v2** — clones, supports ru, but **CPML licensed** (non-commercial above
  revenue/user thresholds) and Coqui is defunct. No documented stress support.
- **Silero v5** — native `+` stress marks *and* a `silero_stress` predictor, i.e.
  correct Russian. **No voice cloning**: fixed speakers only.
- **F5-TTS** — en/zh; the Russian variant is a community fine-tune.

**VERDICT** CONFIRMED — there is no drop-in. The field splits cleanly into
*clones your voice but cannot be told where the stress goes* and *gets the stress
right but is not your voice*. Voice identity is the product (§2.3), so the
engine swap is not available.
**WHAT IS LEFT, and the one worth trying:** use RUAccent as an **oracle** rather
than as input conditioning. It predicts the correct stressed vowel reliably; the
failure was only ever in feeding marks to a model that cannot read them. Align
each synthesized word, measure which vowel actually carries prominence (F0 +
energy + duration), and **veto takes that disagree** — the same shape as the
existing WER veto, inside `best_of`, no engine change.
**PREREQUISITE, MEASURED 2026-08-09 — AND THE INSTRUMENT FAILED.** Whether the
error varies take-to-take was tested on the 46 ru control segments already on
disk (2 takes each), by detecting the stressed syllable acoustically and asking
whether the two takes agreed. Deliberately an AGREEMENT test, not a correctness
test: it needs only a consistent detector, not a right one.

| | rate |
|---|---|
| cross-take disagreement | 35.7% (140/392 words) |
| **null: same take, time-stretched 5%** | **29.4% (116/395)** |
| difference | 6.3 pp, z = 1.90, **p = 0.057** |

**VERDICT** **INCONCLUSIVE, and the detector is why.** A reliable detector would
put the null near 0%; at 29.4%, **82% of the apparent cross-take movement is
detector noise**. p = 0.057 hints the stress really does move but does not clear
0.05, and this is NOT evidence that placement is deterministic — only that this
instrument cannot tell.
**THE FIRST NULL WAS TOO WEAK AND SAID 0%.** It added faint noise, which leaves
every word boundary in place and therefore tests only that the detector is
deterministic — which it trivially is. The null has to perturb TIMING, the way a
real second take does. A control that cannot fail is not a control.
**CONSEQUENCE FOR THE VETO PLAN:** blocked on the detector, not on the premise.
A veto misfiring on 29% of words would reject good takes constantly. Energy
peak-picking over Whisper word timestamps is not a usable stress detector — the
weak link is localisation: word boundaries are coarse and syllable nuclei are
inferred from a smoothed sonority envelope rather than measured.
**NEXT INSTRUMENT** phoneme-level forced alignment (wav2vec2/MMS CTC, or MFA)
gives real vowel boundaries instead of guessed ones. Validate any replacement
against ratings already on disk — the 6 unusable control clips and the segments
marked bad in the reference test — before spending a pod on it.

### 2.1d Mixed rating axes did NOT break refit (my hypothesis, refuted)

**CLAIM** The listener said (2026-08-09) he had been rating STRESS while every
page's instruction said "the one you would ship". If rounds were judged on
different axes, refit has been fitting one weight vector to several different
questions, which would explain its near-zero cross-validated rho.
**EVIDENCE** `qc/axis_check.py`, on the 112 en / 114 ru rows. `variant` is a
proxy for rating round.
- **Rounds disagree no more than chance.** Permuting round labels 1000x within a
  language, the observed between-round spread of per-round Spearman sits inside
  the null for every feature: en sim p=0.155, mos p=0.208, f0st p=0.716; ru sim
  p=0.147, mos p=0.986, f0st p=0.198. The eye-catching sign flips (en sim +0.447
  in one round, -0.507 in another) are what Spearman does at n=8-46.
- **Removing round effects does not recover signal.** Rank-normalising ratings
  WITHIN round strips any per-round axis or scale offset. Cross-validated rho
  moves **en +0.024, ru -0.129**. If pooling heterogeneous rounds were the
  problem this is the fix, and in ru it makes things clearly worse.
**VERDICT** **REFUTED.** I proposed this an hour after the listener's remark and
it does not survive its own test. Recorded because the remark was TRUE — he was
rating a different axis — and the plausible inference from it was still wrong.
**WHAT THE TEST DID FIND, and it is worse:**
1. **Floor effects shrink the usable sample.** en `qwen+fast+icl` is 83% rated 1,
   `0.6B` 62%, `rank-best` 67%. A round with no rating variance contributes
   nothing whatever its axis, so the effective n is well under 112.
2. **The fit is not identified at this n.** en pooled best-fit is `f0 0.95`; the
   within-round fit of the SAME ratings is `sim 0.55, tempo 0.35`. Two defensible
   procedures, near-opposite weights — that instability IS why refit refuses.
3. **A MISSING feature, not a mixed one.** Stress drives 10-24% of ru ratings
   (§2.1b) and NO metric in the set encodes it. Unmodellable signal caps rho no
   matter how many ratings are added, so "needs more ratings" is not the whole
   answer and may not be the main one.

### 2.2 The reference is a CURATED asset, not a per-video by-product

**CLAIM** Cutting the clone reference from each lesson's own audio beats reusing
a good one from another video.
**EVIDENCE** 2026-08-03. `prep` + `tune` R1 cut four references from this lesson
and picked `ref_04`. Measured on the reference clips themselves:

| reference | mos | f0st |
|---|---|---|
| sketch60_ref_03 / sketch_ref_07 | **4.59** | 2.13 |
| this lesson's four | 4.23–4.33 | — |
| ref_04 (R1's winner) | 4.26 | **1.76 — flattest of all 11** |

And the listener's verdict lines up exactly: audio built on **sketch60_ref_03**
rated ~3.5/5, while the bake-off clips built on **ref_04** were marked **12 of
18 unusable**.
**VERDICT** **REFUTED.** Source recording quality varies per lesson, and a
lesson with worse audio yields a worse reference. The reference is a curated
asset — pick the best recording of the speaker that exists, from ANY video.
**ROOT CAUSE OF THE BAD PICK** `preamble` scopes `tune.refs_glob` to
`ref/{stem}_ref_*.wav`, so R1 compared this lesson's clips against each other
and **never saw the sketch60 references at all**. A sweep that cannot see the
incumbent cannot rank against it — the same shape as the bake-off's "ADVISORY —
incumbent not in this run" guard, which exists for exactly this reason.
**ENCODED** the reference is now a config default, not a per-video override.

**SETTLED BY EAR 2026-08-04.** R1 over all 11 references, then a blind
3-sentence comparison of the top three:

| reference | R1 sim | best | unusable |
|---|---|---|---|
| **sketch_ref_08.trim12s** | 0.586 | **2 of 3** | 0 |
| sketch_ref_07 | **0.678** | 1 | 1 |
| sketch60_ref_03 *(previous default)* | 0.630 | **0** | **2** |

The ear picked the **lowest-sim** candidate, and the previous default — chosen by
three metric sweeps — was never once preferred. I had argued for sketch_ref_07
on its 0.092 identity lead (9x the noise band) and was wrong. `tts.reference_wav`
is now `ref/sketch_ref_08.trim12s.wav`. n=3 sentences, so the ORDER is
provisional; "the incumbent was never preferred" is not.

**AND THEN UN-SETTLED, 2026-08-09 — the reference is not a lever at all.**
The n=3 caveat above turned out to be the whole story. The new reference was
re-synthesized across all five languages and paired against the old one, same
sentence, blind, ≥5 s, in both languages:

| | new (`ref_08.trim12s`) | old (`sketch60_ref_03`) | two-sided binomial |
|---|---|---|---|
| en | 13 | 10 | p = 0.68 |
| ru | 8 | 11 | p = 0.65 |
| **combined** | **21** | **21** | **p = 1.000** |

A perfect tie on 42 within-sentence judgements. At n=23 the split would have had
to reach 17–6 to clear p<0.05; the 2-of-3 result that drove the config change has
a two-sided p of **1.00** and could never have shown anything.
**VERDICT** **REFUTED — the reference choice is below the resolution of the ear.**
Four sweeps and three blind tests went into choosing between clips that are
indistinguishable in use. No revert: there is no evidence either way now, and
ref_08 is already synthesized everywhere. **Stop sweeping references.** The
generalisable lesson is the one this file keeps re-learning — *compute the
binomial before treating a small blind test as a decision.* A 2-of-3 is a
coin flip with extra steps.

### 2.3 The accent is identity, not a defect

**CLAIM** The Slavic accent in the English dub should be removed.
**EVIDENCE** 2026-08-03. The listener called it "a rude Slavic accent" and then,
shown alternatives, ranked them: qwen clone (accented) **good**, OpenVoice
tone-color conversion **bad**, native-English source **disaster**. His
correction: *"The accent shouldn't be gone at all, we can leave it just a
little bit, it's his voice identity."*
**VERDICT** The complaint was about DEGREE, not presence. Removing the accent
removes him. Measured on the OpenVoice path: identity transfer was real but
partial (sim to his reference −0.006 → 0.239, against qwen's 0.610) — and the
ear rejected it even though mos (4.65 vs 4.56) and f0st (2.53 vs 1.90) both
favoured it.
**NOT ADOPTED** OpenVoice v2 (MIT, 131 MB, CPU-fast, language-agnostic
converter) is a dead end for this project. Kept here so it is not re-proposed:
its checkpoint S3 bucket is dead (use HuggingFace), it watermarks by default and
its `enable_watermark` kwarg is broken upstream, and it needs its own py3.10
venv because `faster-whisper==0.9.0` drags `av==10`.
**LESSON** Two metrics favoured a version the listener called bad. This is the
third time in one day that metrics and the ear disagreed and the ear was right.

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

**CLAIM** Moving the torch-backed metrics to CUDA changes the scores.
**EVIDENCE** 2026-08-09, the validation `set_device()` asks for: 24 real takes
scored twice in one process, device switched between passes, nothing else
changed. ECAPA embeddings cosine cpu-vs-cuda **1.000000** (min and mean).
Distill-MOS max |Δ| **0.000412**, mean 0.000122, against a 0.007 noise floor —
17× inside it. Wall clock **33.9 s → 2.8 s, 12.3×**.
**VERDICT** **REFUTED — it is free.** Scores stay comparable with every manifest,
ratings row and scorecard on disk, which was the entire reason for the cpu
default.
**SCOPE, so the saving is not oversold:** `set_device` moves the two TORCH models
only. Of measured phase totals (sim 3% + mos 11% + f0 6% + wer 16%), this touches
the **14%** that is sim+mos. `f0` is librosa pyin and `wer` is faster-whisper —
both still CPU and both still worth their own look.
**ENCODED** `config.gpu.yaml` `qc.metrics_device: cuda` — in the pod profile, not
`config.yaml`, because a Mac run falls back and local scores must stay on cpu.

**CLAIM** Spot (`interruptible: true`) is a 2–3× cost lever, made safe by the
take cache: preemption resumes from cached segments rather than from zero.
**EVIDENCE** Six spot pods over two days, none of which ever reached synthesis.
2026-07-30 on COMMUNITY: died at 62 s / 7m27s / 8m32s / 8m53s. 2026-08-09 on
SECURE, after the pool change that was supposed to fix it: `EXITED` before SSH
at ~2 min, then `closed by remote host` at ~11 min at the end of the pip
install. Disk is ruled out — 40 GB with ~14 GB measured headroom, and the same
install completes on on-demand pods routinely.
**VERDICT** REFUTED, and the *reasoning* was refuted, not just the setting. The
cache argument only covers work already done, and every death has been during
the **~11 min bootstrap**, which caches nothing and is re-paid in full. Spot
does not make a short run cheaper; it makes it never finish. A run of this shape
is ~40 min, so the bootstrap is a quarter of it and the exposure is
structural.
**ENCODED** `config.gpu.yaml` `runpod.interruptible: false`, with the six
lifetimes in the comment. Reopening this needs a mechanism that survives a
mid-bootstrap kill — a network volume holding `.venv` is the honest
prerequisite, and it is region-locked, which shrinks the spot pool it would
depend on.

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

1. **RU wrong stress — SOLVABLE, and the path is now known (§2.1f).** Errors are
   stochastic per take: 8 of 14 sentences had exactly one bad take and NONE had
   two, so correct audio already exists in every lesson and is being discarded at
   random. A stress-aware selector takes the error rate from ~29% to 8.2% at
   `best_of=2` and 2.3% at 3. The one missing piece is a DETECTOR — no existing
   metric works (wer AUC 0.647; mos AUC 0.725 in the WRONG direction), and THREE
   detectors have now failed the gate: acoustic prominence (§2.1f, 29% self-flip),
   wav2vec2 phonemes (§2.1g, AUC 0.562) and MFA variant alignment (§2.1h, AUC
   0.641). The blocker is no longer acoustic — §2.1h showed the ORACLE itself
   (RUAccent) errs on real words, and some words have no context-free answer, so
   any oracle-based detector inherits an error rate comparable to the defect.
   **The remaining option that works today needs no detector: the listener flags
   bad ru takes by ear and we re-roll only those — correct takes always exist
   (§2.1f).**
   **Largest known quality defect, and now the best-understood one.**
2. **Eval calibration** — `refit` says KEEP CURRENT WEIGHTS (rho 0.237 vs a 0.30
   floor, p 0.078). Until this clears, the harness cannot arbitrate quality
   changes and every lever above is judged by ear.
   **NOT a mixed-axis problem — tested and refuted 2026-08-09 (§2.1d).** The
   likeliest cause is a MISSING feature: nothing in the set encodes stress, and
   stress drives 10-24% of the ru ratings. More ratings alone may not fix it.
3. **Cross-model ASR consensus** — the only technique that catches silent
   omissions (v3 dropped a clause v2 and turbo both caught). Proposed, not built.
4. **`f0` and `wer` scoring are still CPU** — 22% of measured phase totals, now
   the largest remaining cost lever after `metrics_device: cuda` took sim+mos.

*(Closed 2026-08-09: spot instances — REFUTED, §3. `metrics_device: cuda` —
adopted, free, §3. Reference selection — REFUTED as a lever, §2.2. ICL vs
x_vector left on 2026-08-04, REFUTED by ear, §2. It had sat here reading as
"in flight" for five days after it was settled, which is the same stale-status
failure §4 warns about.)*

*(ICL vs x_vector left this list on 2026-08-04 — REFUTED by ear, §2. It sat here
reading as "in flight" for five days after it was settled, which is the same
stale-status failure the file warns about in §4.)*
