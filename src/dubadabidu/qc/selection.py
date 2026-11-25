# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-of-N take selection: synthesize several takes, score, keep one.

Moved here from pipeline/tts_engine.py, unchanged. An engine turns text into a
WAV; WHICH take ships is policy driven by the QC metrics, and keeping it in the
engine module was the only reason an engine imported qc. Takes are synthesized
through `tts.synthesize` (looked up on the module at call time), and the
metrics are imported inside the functions, as before, so a test that patches
either still reaches this code.
"""
from __future__ import annotations

import logging
from pathlib import Path

from dubadabidu.engines import tts
from dubadabidu.core.manifest import resolve_engine
from dubadabidu.core.config.schema import Config, EvalWeights, Tts

log = logging.getLogger("dubadabidu.tts")   # same logger as before the move


PACE_TOL = 0.35        # |dur/target - 1| that zeroes the pace term
FIT_SLACK = 1.10       # takes within target*this count as "fitting the slot"


def _f0_delivered(cand: Path) -> float:
    """f0 liveliness of the take AS DELIVERED — i.e. after s6's edge trim.

    Take selection used to measure f0 on the raw take while the ear (and
    evaluate's qc_f0st) get the PLACED segment, which s6 trims at TRIM_DB
    before mixing. Trimming changes which frames librosa.pyin sees as voiced,
    and the gap is large and one-directional: measured on sketch60, raw-take
    f0st ran 0.1-1.4 semitones ABOVE the delivered value (ru/u0001: 3.45 raw
    vs 2.02 delivered, with no time-stretch involved at all).

    That made tts.min_f0st a floor in the wrong domain — takes clearing 2.2 at
    selection could land near 2.0 in the mix — and it fed the composite ranker
    an f0 term that partly measured leading silence. Measuring post-trim puts
    selection and evaluate in the same units, so min_f0st means one thing.

    Only f0 moves to the delivered domain. mos_min/sim/dur keep their raw-take
    thresholds (retake_mos_below, early_accept_*, FIT_SLACK) — those are
    calibrated against raw takes and s5 places on raw-take durations.
    """
    from dubadabidu.qc.metrics import f0_semitone_std
    from dubadabidu.core.audio import trim_and_fade  # the exact trim s6 applies
    from pydub import AudioSegment
    try:
        trimmed = trim_and_fade(AudioSegment.from_wav(cand).set_channels(1))
    except Exception as e:      # never let a measurement failure kill a synth
        log.warning("post-trim f0 measurement failed for %s (%s); "
                    "falling back to the raw take", cand.name, e)
        return f0_semitone_std(cand)
    tmp = cand.with_name(cand.stem + ".f0tmp.wav")
    try:
        trimmed.export(tmp, format="wav")
        return f0_semitone_std(tmp)
    finally:
        tmp.unlink(missing_ok=True)


# Fallback only. These are the values _take_rank hardcoded until 2026-08-09,
# normalised: mos .4 / f0 .2 / sim .2 / pace .2. They were a hand-copy of
# qc.eval.weights that then drifted out of contact with it — see _take_rank.
DEFAULT_RANK_W = EvalWeights(mos=0.40, f0=0.20, sim=0.20, tempo=0.20)


def _take_rank(m: dict, target_dur: float | None = None,
               weights: EvalWeights | None = None) -> float:
    """Relative ranking of takes of the SAME text/ref, from `qc.eval.weights`.

    THE WEIGHTS ARE NOW CONFIG, AND THAT IS A FIX. They were hardcoded here as
    (0.5 mos, 0.25 f0, 0.25 sim, 0.25 pace) — a hand-copy of the config values
    that could not follow them. `qc.eval.weights` was read in exactly one place,
    tune.py, so editing it changed the reference-sweep objective and NOTHING
    about which take ships. Three consequences, all live until this was found:
      - `refit` proposes and gates changes to a knob that did not select takes,
        so every cross-validated rho in FINDINGS 2.1 described a formula the
        pipeline never ran;
      - dropping `sim` to 0.0 on 2026-08-09 did not remove it from selection —
        it stayed at an effective 0.2 here;
      - a diagnostic that scored takes with the config weights was measuring
        something that does not exist.
    Same family as "production was still routed to chatterbox" and the adoption
    gate with no incumbent: a knob that looks load-bearing and is not.

    `tempo` in the config maps to the PACE term. Raw (uncalibrated) cosine is
    fine for sim — all takes share one reference, only the ordering matters.
    Terms that do not apply (no sim on the edge engine, no target_dur) drop out
    and the rest renormalise, so a weight set that sums to 1 still behaves.

    Pace term (when target_dur is known): pacing varies widely between rolls of
    the same text (measured −18%..+45% on sketch60), so without it the delivered
    speed is a dice roll — a fast-talking take can win on MOS alone and the
    dub's rhythm drifts run to run. Takes closest to the source slot's pace
    score highest; PACE_TOL away scores zero.

    NOTE the take metrics are cached in the manifest but `rank` is NOT part of
    synth_hash, so changing these weights does not invalidate cached audio —
    re-selection needs `--force` (or a re-rank from the stored per-take
    metrics). That is deliberate and documented on dub.py's --force.
    """
    w = weights or DEFAULT_RANK_W
    mos_n = max(0.0, min(1.0, (m["mos_min"] - 1.0) / 4.0))
    f0_n = max(0.0, min(1.0, m.get("f0st", 0.0) / 4.0))
    parts = [(float(w.mos), mos_n), (float(w.f0), f0_n)]
    if m.get("sim") is not None:
        parts.append((float(w.sim), m["sim"]))
    if target_dur and m.get("dur"):
        pace = max(0.0, 1.0 - abs(m["dur"] / target_dur - 1.0) / PACE_TOL)
        parts.append((float(w.tempo), pace))
    total = sum(x for x, _ in parts)
    if total <= 0:
        # every applicable weight is zero — fall back to MOS rather than
        # dividing by zero or silently returning a constant (which would make
        # `max` pick the FIRST take every time, i.e. best_of would stop working)
        return mos_n
    return sum(x * v for x, v in parts) / total


def early_accept_ok(m: dict, mos_floor: float, sim_floor: float,
                    min_f0st: float = 0.0, target_dur: float | None = None,
                    wer_max: float | None = None) -> bool:
    """Is this take good enough to stop rolling (tts.best_of_early_accept)?

    Every quality floor the winner would later have to clear must appear here,
    because accepting early leaves a ONE-take pool and every downstream filter
    in synth_best_of degrades to a no-op on a one-element pool
    (`pool = filtered or pool`). f0 was missing from this gate, so early-accept
    silently bypassed tts.min_f0st entirely — worst on the GPU config, where
    best_of is 5 and early-accept fires most often.
    """
    if m["mos_min"] < mos_floor:
        return False
    if target_dur and m["dur"] > target_dur * FIT_SLACK:
        return False
    if m.get("sim") is not None and m["sim"] < sim_floor:
        return False
    if wer_max is not None and m.get("wer", 1.0) > wer_max:
        return False
    if min_f0st and m.get("f0st", 0.0) < min_f0st:
        return False
    return True


def synth_best_of(text: str, lang: str, out: Path, t: Tts,
                  meta: list | None = None, verify_cfg: Config | None = None,
                  verify_text: str | None = None,
                  target_dur: float | None = None,
                  rank_weights: EvalWeights | None = None) -> float:
    """Synthesize best_of takes; keep the winner at `out`.

    Two modes (tts.rank_takes, default true):
      ranked — synthesize ALL best_of takes, score each on a composite of
        windowed MOS + f0 liveliness + ECAPA similarity (_take_rank), and pick
        the composite-best among takes clearing the MOS floor. The old
        MOS-only early-stop froze flat or off-voice takes into the cache
        forever the moment they cleared the naturalness gate — monotony and
        clone drift were invisible to take selection by design.
      legacy — early-stop on the first take with worst-3s-window MOS >=
        retake_mos_below (cheapest; set rank_takes: false).

    Optional WER veto (verify_cfg + verify_text, used by autopilot re-rolls of
    WER-flagged segments): takes whose back-transcription WER exceeds
    qc.wer_flag_threshold are disqualified unless every take fails — a re-roll
    triggered by hallucination must not be won by another hallucination.

    `target_dur` (the source slot in seconds, passed by s4/s5) adds a pacing
    term to the ranking and an eligibility gate: takes fitting the slot
    (dur <= target*FIT_SLACK) are preferred outright — a take that overflows
    forces a shortened variant downstream, losing content for no reason when
    a fitting roll exists. Relaxed only if every take overflows.

    `meta`, if given, receives one dict per take (mos_min/f0st/sim/dur/wer/
    rank/picked) — the per-take record the weight re-fit and FIXES.md
    diagnostics read. Used by s4 AND s5 (variant retries): a variant that
    replaces the primary in the mix must clear the same gate as the primary.

    Crash-safe by construction: takes roll to their own `_take{k}.wav` files and
    `out` (the hash-cache key) is created only by the winner's atomic rename. A
    run killed mid-best-of therefore caches NOTHING and re-rolls cleanly on the
    next pass — which is what makes preemptible spot pods safe to use here.
    """
    import soundfile as sf
    from dubadabidu.qc.metrics import ecapa_embed, cosine, mos_min_window
    best_of = int(t.best_of)
    threshold = float(t.retake_mos_below)
    ranked = bool(t.rank_takes) and best_of > 1
    engine = resolve_engine(t, lang)
    ref = t.reference_wav
    ref_emb = (ecapa_embed(ref) if ranked and engine != "edge"
               and ref and Path(ref).exists() else None)
    wer_max = (float(verify_cfg.qc.wer_flag_threshold)
               if verify_cfg and verify_text else None)
    # adaptive best_of (ranked mode): if an early take already clears a HIGH bar
    # AND fits the slot AND passes any WER veto, stop rolling — the extra takes
    # rarely beat an already-excellent one and each costs GPU time. Quality-safe
    # because the bar is well above the retake floor; disable with
    # tts.best_of_early_accept: false to always synthesize all best_of takes.
    early = ranked and bool(t.best_of_early_accept)
    early_mos = float(t.early_accept_mos)   # worst-3s window scores low
    early_sim = float(t.early_accept_sim)
    min_f0st = float(t.min_f0st)

    def _early_ok(m: dict) -> bool:
        return early_accept_ok(m, early_mos, early_sim, min_f0st,
                               target_dur, wer_max)

    def _measure(cand: Path) -> dict:
        info = sf.info(str(cand))
        m = {"mos_min": round(mos_min_window(cand), 2),
             "dur": round(info.frames / info.samplerate, 2)}
        if ranked:
            # delivered domain, so this is comparable to min_f0st and to
            # evaluate's qc_f0st (see _f0_delivered)
            m["f0st"] = round(_f0_delivered(cand), 2)
            if ref_emb is not None:
                m["sim"] = round(cosine(ref_emb, ecapa_embed(cand)), 3)
        if wer_max is not None:
            from dubadabidu.qc.backcheck import segment_wer
            m["wer"] = round(segment_wer(verify_cfg, verify_text, cand, lang), 3)
        return m

    # Sweep leftovers from a previous ATTEMPT at this exact segment (a run that
    # died mid-best-of, or one that rolled more takes than the current best_of).
    # They are never read — every take below is overwritten before it is scored —
    # but without this they accumulate on disk across crashed pod sessions.
    for junk in out.parent.glob(f"{out.stem}_take*.wav"):
        junk.unlink(missing_ok=True)
    for junk in out.parent.glob(f"{out.stem}_reroll*.wav"):
        junk.unlink(missing_ok=True)

    takes = []  # (path, metrics)
    for take in range(best_of):
        # EVERY take rolls to its own file — never straight to `out`. `out` is
        # the hash-cache key: s4 treats its existence as "this segment has an
        # accepted take" (fresh = not out.exists()). Writing take 0 there meant
        # a run killed before the ranking finished — spot preemption, the budget
        # deadline, the pod self-destruct watchdog, an OOM — left an UNRANKED,
        # ungated take cached as if it had won, with no tr.takes record and no
        # way to notice. Only --force cleared it. `out` is now created by exactly
        # one operation: the atomic rename of the winner, below.
        cand = out.with_name(f"{out.stem}_take{take}.wav")
        tts.synthesize(text, lang, cand, t)
        m = _measure(cand)
        takes.append((cand, m))
        if not ranked and m["mos_min"] >= threshold \
                and (wer_max is None or m["wer"] <= wer_max):
            break
        if early and _early_ok(m):
            log.info("early-accept take %d/%d for %r (mos %.2f)",
                     take + 1, best_of, text[:40], m["mos_min"])
            break
        if take + 1 < best_of and not ranked:
            log.info("retake %d for %r (mos %.2f < %.2f)",
                     take + 1, text[:40], m["mos_min"], threshold)

    # monotony re-roll (ranked mode): if EVERY take so far is flatter than
    # tts.min_f0st, the ranker has no lively read to choose from — roll more.
    # Take-to-take f0 variance is real (~0.4 st on sketch60) so extra samples
    # actually find a livelier take; the exaggeration parameter does NOT reliably
    # move f0 (measured 2026-07-14), so we sample instead of twiddling. The
    # composite ranker still picks the winner (a livelier take must also hold up
    # on MOS/sim), so this only ADDS candidates — it can't force a bad take in.
    reroll_max = int(t.f0_reroll_max)
    if ranked and min_f0st > 0 and reroll_max > 0:
        rr = 0
        while rr < reroll_max and max(
                (tk[1].get("f0st", 0.0) for tk in takes), default=0.0) < min_f0st:
            rr += 1
            cand = out.with_name(f"{out.stem}_reroll{rr}.wav")
            tts.synthesize(text, lang, cand, t)
            takes.append((cand, _measure(cand)))
            best_f0 = max(tk[1].get("f0st", 0.0) for tk in takes)
            log.info("monotony re-roll %d/%d for %r (best f0st %.2f, floor %.2f)",
                     rr, reroll_max, text[:40], best_f0, min_f0st)

    # eligibility gates, relaxed only when they would eliminate every take
    pool = takes
    if wer_max is not None:
        ok = [tk for tk in pool if tk[1]["wer"] <= wer_max]
        pool = ok or pool
    if target_dur:
        fits = [tk for tk in pool if tk[1]["dur"] <= target_dur * FIT_SLACK]
        pool = fits or pool
    gated = [tk for tk in pool if tk[1]["mos_min"] >= threshold]
    pool = gated or pool
    if ranked and min_f0st > 0:
        # prefer takes that clear the monotony floor — otherwise a flat take can
        # win the composite on MOS alone and the re-roll's lively take is wasted.
        # Relaxed only if none qualify (then rank picks the least-flat via its
        # f0 term). Applied after the MOS gate so a lively pick still holds up.
        lively = [tk for tk in pool if tk[1].get("f0st", 0.0) >= min_f0st]
        pool = lively or pool
    key = (lambda m: _take_rank(m, target_dur, rank_weights)) if ranked \
        else (lambda m: m["mos_min"])
    winner = max(pool, key=lambda tk: key(tk[1]))
    for path, m in takes:
        m["rank"] = round(key(m), 4)
        m["picked"] = path is winner[0]
        if meta is not None:
            meta.append(m)
    # Publish the winner LAST, with a single rename. Path.replace is an atomic
    # POSIX rename within the directory, so `out` — the cache key — goes from
    # "absent" straight to "the complete, ranked winner". There is no window in
    # which it holds a take that has not been through the gates above.
    winner[0].replace(out)
    for path, _ in takes:
        if path is not winner[0]:
            path.unlink(missing_ok=True)
    if len(takes) > 1:
        log.info("picked take %d/%d for %r (rank %.3f, mos %.2f)",
                 takes.index(winner) + 1, len(takes), text[:40],
                 winner[1]["rank"], winner[1]["mos_min"])
    return winner[1]["mos_min"]
