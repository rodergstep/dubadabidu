"""Batch report (IMPROVEMENT_PLAN Phase D): one matrix over every processed
video — per video x language means (score/sim/mos), flagged-segment counts,
and links to review pages. The worst-first table the human review loop starts
from: read the worst row, open its review page, fix, re-run, regenerate.

Reads only manifests — run it anytime, costs nothing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402


def _mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 3) if vals else None


def _lang_row(cfg: dict, video: str, lang: str, trs: list[dict]) -> dict:
    score_flag = cfg["qc"].get("eval", {}).get("score_flag", 0.55)
    wer_thr = cfg["qc"]["wer_flag_threshold"]
    engines = {t.get("synth_engine") for t in trs} - {None}
    return {
        "video": video, "lang": lang, "segs": len(trs),
        # EDGE rows are plumbing checks — their scores say nothing about the
        # cloned voice and must not be compared against real rows
        "engine": "⚠EDGE" if "edge" in engines else ",".join(sorted(engines)) or "-",
        "score": _mean([t["qc_score"] for t in trs if "qc_score" in t]),
        "sim": _mean([t["qc_sim_cal"] for t in trs if "qc_sim_cal" in t]),
        "mos": _mean([t["qc_mos"] for t in trs if "qc_mos" in t]),
        "flagged": sum(1 for t in trs
                       if t.get("qc_score", 1.0) < score_flag),
        "wer_bad": sum(1 for t in trs if t.get("qc_wer", 0) > wer_thr),
        "overflow": sum(1 for t in trs if t.get("fit") == "overflow"),
        # drift_exceeded superseded overrun_s (soft-anchor s5: mix overlap is
        # impossible; excessive timeline drift is the new failure mode)
        "overlap": sum(1 for t in trs
                       if t.get("overrun_s") or t.get("drift_exceeded")),
        "synth": sum(1 for t in trs if t.get("synth")),
    }


def run(cfg: dict, videos: list[str] | None = None) -> Path:
    work = Path(cfg["work_dir"])
    if videos:
        manifests = [M.manifest_path(cfg, v) for v in videos]
    else:
        manifests = sorted(work.glob("*/manifest.json"))

    rows = []
    for mp in manifests:
        if not mp.exists():
            print(f"[batch] skipped (no manifest): {mp}")
            continue
        man = json.loads(mp.read_text(encoding="utf-8"))
        name = mp.parent.name
        langs = sorted({l for u in man["utterances"] for l in u["tr"]})
        for lang in langs:
            trs = [u["tr"][lang] for u in man["utterances"]
                   if lang in u["tr"]]
            rows.append((_lang_row(cfg, name, lang, trs),
                         mp.parent / f"review_{lang}.html"))

    # worst-first: rows with a score sort ascending; unevaluated rows sink last
    rows.sort(key=lambda r: (r[0]["score"] is None,
                             r[0]["score"] if r[0]["score"] is not None else 0))

    lines = ["# batch report", "",
             "worst-first. flagged = qc_score below qc.eval.score_flag; "
             "review pages exist only after `dubadabidu review <video>`.", "",
             "| video | lang | engine | segs | synth | score | sim | mos | flagged "
             "| wer>thr | overflow | overlap | review |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else v  # noqa: E731
    for r, page in rows:
        link = f"[page]({page})" if page.exists() else "-"
        lines.append(
            f"| {r['video']} | {r['lang']} | {r['engine']} | {r['segs']} | {r['synth']} "
            f"| {fmt(r['score'])} | {fmt(r['sim'])} | {fmt(r['mos'])} "
            f"| {r['flagged']} | {r['wer_bad']} | {r['overflow']} "
            f"| {r['overlap']} | {link} |")
    if not rows:
        lines.append("(no processed videos found)")

    out = work / "batch_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[batch] report: {out}")
    return out
