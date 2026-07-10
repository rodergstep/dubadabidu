"""Verdict writeback (AUTOPILOT.md M3 — closes the flywheel).

`dubadabidu verdicts <video> <exported.json>` ingests the JSON exported from a
review page ({key, ratings, verdicts}) and:

  1. writes human_rating / human_verdict into the manifest per segment — the
     autopilot treats an accepted segment as settled and never re-rolls it;
  2. appends/updates rows in ratings_<lang>.json at the repo root — the
     accumulated (human verdict, qc metrics) pairs the periodic weight re-fit
     (M4) trains on. qc_mos_min is recorded as a candidate feature: the synth
     gate uses windowed MOS while the composite uses whole-take MOS, and the
     re-fit is where that disagreement gets reconciled with data.

The export key embeds a segmentation hash; a mismatch means the video was
re-segmented since the ratings were taken and they no longer describe these
utterance boundaries — the ingest refuses rather than poisoning the manifest.
"""
from __future__ import annotations
import hashlib
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402

log = logging.getLogger("dubadabidu.qc.verdicts")

QC_FEATURES = ("qc_score", "qc_sim2", "qc_sim_cal", "qc_mos", "qc_mos_min",
               "qc_f0st", "qc_wer", "tempo", "fit")


def _seg_hash(utterances: list[dict], lang: str) -> str:
    """Must mirror review_page.py: hash over worst-first-sorted ids+starts."""
    us = sorted(utterances, key=lambda u: u["tr"][lang].get("qc_score", 1.0))
    return hashlib.sha1(
        ",".join(u["id"] + str(u["start"]) for u in us).encode()).hexdigest()[:6]


def run(cfg: dict, video: str, export_file: str) -> None:
    data = json.loads(Path(export_file).read_text(encoding="utf-8"))
    key = data.get("key", "")
    ratings = {k: v for k, v in (data.get("ratings") or {}).items()
               if isinstance(v, (int, float))}
    verdicts = data.get("verdicts") or {}
    stem = Path(video).stem
    try:
        head, seg_hash = key.rsplit("_", 1)
        vstem, lang = head.rsplit("_", 1)
    except ValueError:
        raise SystemExit(f"malformed export key {key!r} — expected "
                         f"<video>_<lang>_<seghash> (re-export from the "
                         f"review page).")
    if vstem != stem:
        raise SystemExit(f"export is for video {vstem!r}, not {stem!r}.")

    man = M.load(cfg, video)
    if M.edge_langs(man, [lang]):
        raise SystemExit(
            f"{stem}/{lang} was synthesized with the EDGE fallback (generic "
            f"voice, no cloning) — these ratings would poison the qc-weight "
            f"re-fit with judgments of a voice that is not yours. Re-run s4 "
            f"with the real engine, re-review, then ingest.")
    if _seg_hash(man["utterances"], lang) != seg_hash:
        raise SystemExit(
            f"segmentation hash mismatch ({seg_hash}) — the video was "
            f"re-segmented since these ratings were taken; they describe "
            f"different utterance boundaries. Re-review and re-export.")

    rows_path = Path(f"ratings_{lang}.json")
    rows = (json.loads(rows_path.read_text(encoding="utf-8"))
            if rows_path.exists() else [])
    by_key = {(r["video"], r["id"]): r for r in rows}

    n_man = 0
    for u in man["utterances"]:
        uid = u["id"]
        rating, verdict = ratings.get(uid), verdicts.get(uid)
        if rating is None and verdict is None:
            continue
        tr = u["tr"][lang]
        if rating is not None:
            tr["human_rating"] = rating
        if verdict is not None:
            tr["human_verdict"] = verdict
        n_man += 1
        row = {"video": stem, "lang": lang, "id": uid,
               "rating": rating, "verdict": verdict,
               "text": tr.get("fitted_text", tr.get("text", ""))}
        row.update({k: tr[k] for k in QC_FEATURES if k in tr})
        by_key[(stem, uid)] = row  # replace: latest verdict wins
    M.save(cfg, video, man)

    rows = sorted(by_key.values(), key=lambda r: (r["video"], r["id"]))
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    n_rej = sum(1 for r in rows if r.get("verdict") == "reject")
    print(f"[verdicts] {stem}/{lang}: {n_man} segments written to manifest; "
          f"{rows_path} now holds {len(rows)} rows ({n_rej} rejects) "
          f"for the weight re-fit.")
