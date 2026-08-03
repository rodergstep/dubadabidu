"""runs.jsonl — an append-only ledger of what every pod run actually cost.

WHY THIS EXISTS. "What does an hour of dubbing cost?" was answered three times
in one day, each time by grepping timestamps out of a log file that only existed
in a scratch directory, and each answer differed because the runs differed
($0.98/h, then $1.09, then $1.22 once pre-synthesized variants were counted).
The numbers were never wrong — they were unrecorded, so every question re-derived
them from whatever evidence happened to survive.

Hand-maintained stats rot; this file is written by the code that spends the
money, so it cannot drift from reality. FINDINGS.md is the other half: insights
and verdicts that are not numbers. Rule of thumb — if a number could be measured
again, it belongs here; if it is a conclusion someone has to remember, it
belongs in FINDINGS.md.

Never let bookkeeping break a run: every write is best-effort and swallows its
own errors. A missing row costs a data point, a raised exception could leak a
pod.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

log = logging.getLogger("dubadabidu.runlog")

LEDGER = Path("runs.jsonl")


def append(record: dict) -> None:
    """Append one run. Best-effort: never raises into the caller."""
    try:
        rec = {k: v for k, v in record.items() if v is not None}
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as e:                                   # noqa: BLE001
        log.debug("runlog append failed (ignored): %s", e)


def load(path: Path | str = LEDGER) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a torn line must not poison the whole ledger
    return out


def cost_per_video_hour(rows: list[dict] | None = None) -> dict:
    """Derive $/hour-of-video from the ledger instead of from memory.

    Uses only rows that recorded a price, a duration and the video length, and
    that actually synthesized (a setup-check has no audio to divide by). The
    bootstrap is reported separately because it is FIXED per run — it dominates
    an 8-minute lesson and is noise on a 60-minute one, so a single blended
    number misleads at both ends.
    """
    rows = load() if rows is None else rows
    usable = [r for r in rows
              if r.get("video_minutes") and r.get("wall_s")
              and r.get("price_per_hr") and r.get("langs")]
    if not usable:
        return {"runs": 0}
    tot_cost = sum(r["wall_s"] / 3600 * r["price_per_hr"] for r in usable)
    tot_vid_h = sum(r["video_minutes"] / 60 for r in usable)
    per_lang = [
        (r["wall_s"] - r.get("bootstrap_s", 0)) / 3600 * r["price_per_hr"]
        / (r["video_minutes"] / 60) / len(r["langs"])
        for r in usable if r["video_minutes"]]
    return {
        "runs": len(usable),
        "total_cost_usd": round(tot_cost, 3),
        "total_video_hours": round(tot_vid_h, 3),
        "usd_per_video_hour_blended": round(tot_cost / tot_vid_h, 3),
        "usd_per_video_hour_per_language": round(sum(per_lang) / len(per_lang), 3),
        "mean_bootstrap_min": round(
            sum(r.get("bootstrap_s", 0) for r in usable) / len(usable) / 60, 1),
    }
