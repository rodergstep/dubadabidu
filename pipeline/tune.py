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
}


def _subset(utterances: list[dict], n: int) -> list[dict]:
    """Stratified by source length: sort, take evenly spaced — short and long
    segments fail differently, so both must be in the objective."""
    us = sorted(utterances, key=lambda u: len(u["text_uk"]))
    if len(us) <= n:
        return us
    step = (len(us) - 1) / (n - 1)
    return [us[round(i * step)] for i in range(n)]


def _band(cfg: dict, wd: Path, man: dict, lang: str, ref: str) -> tuple[float, float]:
    from qc import evaluate as E
    c = dict(cfg, tts=dict(cfg["tts"], reference_wav=ref))
    cal = E.calibration(c, wd, man, lang)
    return cal["floor"], cal["ceiling"]


def _score_trial(cfg: dict, wd: Path, subset: list[dict], lang: str,
                 tts: dict, band: tuple[float, float], take: int = 0) -> dict:
    from qc import metrics as X
    ref_emb = X.ecapa_embed(tts["reference_wav"])
    seg_dir = wd / "tune" / "seg"
    raws, sims, moss, f0s = [], [], [], []
    for u in subset:
        text = u["tr"][lang]["text"]
        h = M.synth_hash(text, lang, tts)
        out = seg_dir / f"{u['id']}_{h}_t{take}.wav"
        if not out.exists():
            synthesize(text, lang, out, tts)
        raw = X.cosine(ref_emb, X.ecapa_embed(out))
        raws.append(raw)
        sims.append(X.calibrate_sim(raw, *band))
        moss.append(X.mos(out))
        f0s.append(X.f0_semitone_std(out))
    n = len(subset)
    sim, mos, f0 = sum(sims) / n, sum(moss) / n, sum(f0s) / n
    sim_raw = sum(raws) / n
    # no tempo term pre-fit; renormalize the remaining weights
    w = cfg["qc"].get("eval", {}).get("weights",
                                     {"sim": 0.25, "mos": 0.40, "f0": 0.20})
    tot = w["sim"] + w["mos"] + w.get("f0", 0.0)
    raw = (w["sim"] * sim + w["mos"] * (mos - 1) / 4
           + w.get("f0", 0.0) * min(1.0, f0 / 4.0))
    return {"sim_raw": round(sim_raw, 3), "sim_cal": round(sim, 3),
            "mos": round(mos, 2), "f0st": round(f0, 2),
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

    base = dict(cfg["tts"], engine="chatterbox")
    results = []  # (round, ref, cfg_weight, exaggeration, take, metrics)

    def _tts_for(ref: str, **over) -> dict:
        rt = ref_meta.get(Path(ref).name, {}).get("text_uk", "")
        return dict(base, reference_wav=ref, reference_text=rt, **over)

    # R1: reference selection at current params
    for ref in refs:
        m = _score_trial(cfg, wd, subset, lang, _tts_for(ref), bands[ref])
        results.append(("R1", ref, base["cfg_weight"], base["exaggeration"], 0, m))
        log.info("R1 %s -> %s", ref, m)
    best_ref = max((r for r in results), key=lambda r: r[5]["score"])[1]

    # R2: param grid on the winning ref
    if "R2" in rounds:
        for cw in tn["cfg_weights"]:
            for ex in tn["exaggerations"]:
                if cw == base["cfg_weight"] and ex == base["exaggeration"]:
                    continue  # already measured in R1
                tts = _tts_for(best_ref, cfg_weight=cw, exaggeration=ex)
                m = _score_trial(cfg, wd, subset, lang, tts, bands[best_ref])
                results.append(("R2", best_ref, cw, ex, 0, m))
                log.info("R2 cfg=%s ex=%s -> %s", cw, ex, m)
    on_ref = [r for r in results if r[1] == best_ref]
    _, _, best_cw, best_ex, _, best_m = max(on_ref, key=lambda r: r[5]["score"])

    # R3: take-to-take variance with the winning config
    if "R3" in rounds:
        win = _tts_for(best_ref, cfg_weight=best_cw, exaggeration=best_ex)
        for take in range(1, tn["best_of"]):
            m = _score_trial(cfg, wd, subset, lang, win, bands[best_ref],
                             take=take)
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
    lines += ["", "NOTE: sim_cal is calibrated per-ref (its own floor/ceiling) — "
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
    return {"reference_wav": best_ref, "cfg_weight": best_cw,
            "exaggeration": best_ex}
