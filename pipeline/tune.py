"""tune: closed-loop TTS config search — run → eval → improve, deterministically.

Rounds (all local, all cached by content hash, resumable):
  R1 ref selection   — every ref/*.wav candidate, current params; timbre is
                       dominated by the reference clip, so pick this first.
  R2 param sweep     — cfg_weight × exaggeration grid on the winning ref.
  R3 best-of-N       — N takes per segment with the winning config; keeps the
                       highest-scoring take (autoregressive TTS variance is free
                       quality if you measure it).

Objective = qc.metrics.composite_score on a stratified segment subset
(sim calibrated per-ref: floor/ceiling measured, not assumed). No tempo term —
tune scores raw synthesis; fitting happens later in s5.

Output: work/<video>/tune/tune_report.md + a paste-ready config.yaml block.
"""
from __future__ import annotations
import logging
from pathlib import Path
from . import manifest as M
from .tts_engine import synthesize

log = logging.getLogger("dubadabidu.tune")

DEFAULTS = {
    "subset_size": 8,
    "refs_glob": "ref/*.wav",
    "cfg_weights": [0.0, 0.2, 0.3],
    "exaggerations": [0.4, 0.55, 0.7],
    "best_of": 3,
    "rounds": ["R1", "R2", "R3"],   # subset for tune-lite, e.g. ["R1"]
    # ---- tune's OWN objective. NOT qc.eval.weights, and that is the fix. ----
    # qc.eval.weights ranks TAKES of one segment. It dropped sim to 0.0 on
    # 2026-08-09 because sim does not predict this listener's per-take judgement
    # (en rho -0.075 p 0.42, ru -0.136 p 0.15). Correct there — and silently
    # catastrophic here, because tune R1 selects the CLONE SOURCE, where
    # similarity is the entire question. Measured with the shipped weights:
    #     ref A  sim_cal .80  mos 4.0  f0 2.0  -> 0.662
    #     ref B  sim_cal .10  mos 4.0  f0 2.6  -> 0.715   <- wins
    # A reference that sounds nothing like the speaker beat one that does, on
    # six tenths of a semitone, and `preamble` wrote that winner into the
    # manifest with no human in the loop.
    #
    # UNVALIDATED, and labelled so on purpose. sim_raw + mos is less degenerate
    # than an objective with no identity term at all; it is NOT calibrated
    # against the ear. The one blind test run picked the LOWEST-sim reference
    # (FINDINGS 2.2, 2026-08-04) and the 42-judgement comparison came back
    # 21-21. Treat R1 as a shortlist generator, not an arbiter — the ear
    # decides, which is why the override gate below defaults to off.
    "r1_weights": {"sim": 0.6, "mos": 0.4, "f0": 0.0},
    # Take-to-take spread here is mos +/-0.24-0.49 and f0st +/-0.438 (measured
    # 2026-08-01 by running one config twice). Scoring a reference on ONE roll
    # per segment therefore ranks the dice; qc/listen.py says so in as many
    # words. Average this many rolls per point instead. Cost is
    # refs x subset_size x this, and it is the one place in the project where
    # paying for more samples buys a decision rather than a number.
    "r1_takes": 3,
    # ---- per-video reference override: OFF, matching FINDINGS 2.2 ----------
    # FINDINGS 2.2 records the verdict as "ENCODED — the reference is now a
    # config default, not a per-video override", because source quality varies
    # per lesson and a bad lesson yields a bad reference (its ref_04 was rated
    # 12-of-18 unusable). Only the ROOT CAUSE was fixed (refs_glob was widened
    # so R1 can at least see the curated pool); `preamble` still wrote
    # man["tts_overrides"] on every video, and s4/s5/evaluate/bakeoff all merge
    # it. The refuted mechanism was still live and regenerating itself.
    # true + a margin re-enables it, gated: see dub.preamble.
    "per_video_ref_override": False,
    "override_margin": 0.05,
}


def _subset(utterances: list[dict], n: int) -> list[dict]:
    """Stratified by source length: sort, take evenly spaced — short and long
    segments fail differently, so both must be in the objective."""
    us = sorted(utterances, key=lambda u: len(u["text_uk"]))
    if len(us) <= n:
        return us
    if n <= 1:
        # n==1 has no spread to stratify over, and the even-spacing formula
        # divides by n-1. Take the median-length segment: the single most
        # representative sample. (bakeoff.tune.subset_size: 1 is the natural
        # cheapest setting, so this path is reachable from config.)
        return us[len(us) // 2:len(us) // 2 + 1] if n == 1 else []
    step = (len(us) - 1) / (n - 1)
    return [us[round(i * step)] for i in range(n)]


def _band(cfg: dict, wd: Path, man: dict, lang: str, ref: str) -> tuple[float, float]:
    from qc import evaluate as E
    c = dict(cfg, tts=dict(cfg["tts"], reference_wav=ref))
    cal = E.calibration(c, wd, man, lang)
    return cal["floor"], cal["ceiling"]


def _score_trial(cfg: dict, wd: Path, subset: list[dict], lang: str,
                 tts: dict, band: tuple[float, float], w: dict,
                 take: int = 0, n_takes: int = 1) -> dict:
    """Mean metrics for one config point over `subset` x `n_takes` rolls.

    `w` is passed in rather than read from cfg: this objective belongs to tune
    (DEFAULTS["r1_weights"]) and reading qc.eval.weights here is what let a
    take-ranking decision silently redefine reference selection.

    `n_takes > 1` averages consecutive take indices from `take`. Rolls are
    cached by (hash, take) so a re-run costs nothing, and R3 reads the same
    files back when it measures per-take spread.
    """
    from qc import metrics as X
    ref_emb = X.ecapa_embed(tts["reference_wav"])
    seg_dir = wd / "tune" / "seg"
    raws, sims, moss, f0s = [], [], [], []
    for u in subset:
        text = u["tr"][lang]["text"]
        h = M.synth_hash(text, lang, tts)
        for k in range(take, take + max(1, n_takes)):
            out = seg_dir / f"{u['id']}_{h}_t{k}.wav"
            if not out.exists():
                synthesize(text, lang, out, tts)
            raw = X.cosine(ref_emb, X.ecapa_embed(out))
            raws.append(raw)
            sims.append(X.calibrate_sim(raw, *band))
            moss.append(X.mos(out))
            f0s.append(X.f0_semitone_std(out))
    n = len(raws)
    sim, mos, f0 = sum(sims) / n, sum(moss) / n, sum(f0s) / n
    sim_raw = sum(raws) / n
    # no tempo term pre-fit (tune scores raw synthesis; fitting happens in s5),
    # so renormalize over the terms that apply
    tot = w.get("sim", 0.0) + w.get("mos", 0.0) + w.get("f0", 0.0)
    if tot <= 0:
        raise SystemExit(
            "tune.r1_weights sum to zero — there is no objective to rank "
            "references by. Give sim and/or mos a positive weight.")
    raw = (w.get("sim", 0.0) * sim + w.get("mos", 0.0) * (mos - 1) / 4
           + w.get("f0", 0.0) * min(1.0, f0 / 4.0))
    return {"sim_raw": round(sim_raw, 3), "sim_cal": round(sim, 3),
            "mos": round(mos, 2), "f0st": round(f0, 2), "takes": n,
            "score": round(raw / tot, 4)}


def run(cfg: dict, video: str, langs: list[str]) -> dict:
    """Returns the winning tts config subset:
    {reference_wav, cfg_weight, exaggeration} — preamble consumes it."""
    tn = {**DEFAULTS, **cfg.get("tune", {})}
    rounds = set(tn["rounds"])
    lang = langs[0]
    if len(langs) > 1:
        log.info("tuning on %s only (config transfers across languages)", lang)
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    subset = _subset(man["utterances"], tn["subset_size"])
    missing = [u["id"] for u in subset if "text" not in u["tr"].get(lang, {})]
    if missing:
        raise SystemExit(f"segments missing {lang} translation: {missing} — run s3.")
    log.info("subset: %s", [u["id"] for u in subset])

    refs = sorted(str(p) for p in Path().glob(tn["refs_glob"]))
    if not refs:
        raise SystemExit(f"no reference candidates match {tn['refs_glob']}")
    # refs.json (written by `dubadabidu prep`) carries each ref's transcript —
    # required by the cosyvoice engine, harmless for the others
    import json as _json
    rj = wd / "refs.json"
    ref_meta = _json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    bands = {r: _band(cfg, wd, man, lang, r) for r in refs}
    for r in refs:
        log.info("band %s: floor=%.3f ceiling=%.3f", r, *bands[r])

    # was hardcoded to chatterbox; tune must sweep whatever we actually ship
    base = dict(cfg["tts"], engine=cfg["tts"]["engine"])
    results = []  # (round, ref, cfg_weight, exaggeration, take, metrics)
    w = tn["r1_weights"]
    r1_takes = max(1, int(tn.get("r1_takes", 1)))
    log.info("R1 objective (tune.r1_weights, UNVALIDATED against the ear): %s; "
             "%d take(s) x %d segment(s) per reference",
             w, r1_takes, len(subset))

    def _tts_for(ref: str, **over) -> dict:
        rt = ref_meta.get(Path(ref).name, {}).get("text_uk", "")
        return dict(base, reference_wav=ref, reference_text=rt, **over)

    # R1: reference selection at current params
    r1_scores: dict[str, float] = {}
    for ref in refs:
        m = _score_trial(cfg, wd, subset, lang, _tts_for(ref), bands[ref], w,
                         n_takes=r1_takes)
        results.append(("R1", ref, base["cfg_weight"], base["exaggeration"], 0, m))
        r1_scores[ref] = m["score"]
        log.info("R1 %s -> %s", ref, m)
    best_ref = max((r for r in results), key=lambda r: r[5]["score"])[1]

    # R2: param grid on the winning ref
    if "R2" in rounds:
        for cw in tn["cfg_weights"]:
            for ex in tn["exaggerations"]:
                if cw == base["cfg_weight"] and ex == base["exaggeration"]:
                    continue  # already measured in R1
                tts = _tts_for(best_ref, cfg_weight=cw, exaggeration=ex)
                m = _score_trial(cfg, wd, subset, lang, tts, bands[best_ref], w,
                                 n_takes=r1_takes)
                results.append(("R2", best_ref, cw, ex, 0, m))
                log.info("R2 cfg=%s ex=%s -> %s", cw, ex, m)
    on_ref = [r for r in results if r[1] == best_ref]
    _, _, best_cw, best_ex, _, best_m = max(on_ref, key=lambda r: r[5]["score"])

    # R3: take-to-take variance with the winning config
    if "R3" in rounds:
        win = _tts_for(best_ref, cfg_weight=best_cw, exaggeration=best_ex)
        # SINGLE takes on purpose — R3 measures the per-take SPREAD, so
        # averaging here would erase the thing it exists to report.
        for take in range(1, tn["best_of"]):
            m = _score_trial(cfg, wd, subset, lang, win, bands[best_ref], w,
                             take=take, n_takes=1)
            results.append(("R3", best_ref, best_cw, best_ex, take, m))
            log.info("R3 take %d -> %s", take, m)
    takes = [r[5]["score"] for r in results
             if r[:4] == ("R3", best_ref, best_cw, best_ex)] + [best_m["score"]]

    lines = ["# tune report", "",
             f"video: {video}  lang: {lang}  subset: "
             f"{[u['id'] for u in subset]}", "",
             "| round | ref | cfg_weight | exaggeration | take "
             "| sim_raw | sim_cal | mos | f0st | score |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for rd, ref, cw, ex, take, m in sorted(results, key=lambda r: -r[5]["score"]):
        lines.append(f"| {rd} | {Path(ref).name} | {cw} | {ex} | {take} "
                     f"| {m.get('sim_raw', '-')} | {m['sim_cal']} | {m['mos']} "
                     f"| {m.get('f0st', '-')} | {m['score']} |")
    lines += ["", f"objective: tune.r1_weights = {w} over {r1_takes} take(s) "
              f"per segment. **UNVALIDATED against the ear** — R1 produces a "
              f"shortlist, not a verdict. The one blind test picked the "
              f"LOWEST-sim reference and a 42-judgement comparison came back "
              f"21-21 (FINDINGS 2.2).",
              "", "NOTE: sim_cal is calibrated per-ref (its own floor/ceiling) — "
              "valid within a ref, biased across refs. When comparing DIFFERENT "
              "refs, judge by sim_raw + mos."]
    lines += ["", f"take-to-take score spread: {min(takes):.3f}–{max(takes):.3f} "
              f"(best-of-{tn['best_of']} re-roll is worth it if this is wide)", "",
              "## winning config — paste into config.yaml", "```yaml", "tts:",
              f"  reference_wav: {best_ref}",
              f"  cfg_weight: {best_cw}",
              f"  exaggeration: {best_ex}", "```"]
    report = wd / "tune" / "tune_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[4:]))
    print(f"\n[tune] report: {report}")
    # r1_scores lets the caller gate on a MARGIN over the incumbent instead of
    # adopting whatever came first — the same shape as the bake-off's
    # "incumbent not in this run" guard. See dub.preamble.
    return {"reference_wav": best_ref, "cfg_weight": best_cw,
            "exaggeration": best_ex, "r1_scores": r1_scores}
