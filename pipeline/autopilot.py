"""Autopilot (AUTOPILOT.md M1): deterministic orchestrator loop over the
existing CLI stages. Perceive -> assess vs spec -> mechanical fix -> re-eval
-> PASS or ESCALATE.

M1 fixes are strictly mechanical: re-roll synthesis for segments failing WER
or score (delete the cached take; the hash cache re-synthesizes a fresh one
through the MOS gate). Anything requiring judgment — translation edits,
reference re-pick, taste — escalates with a reason and a suggested command.
The LLM-agent layer (M2+) replaces the fix policy; the perceive/assess/
escalate skeleton stays.

Every (symptom -> fix -> outcome) is appended to FIXES.md — the playbook
future runs (and future agents) read first.

After a re-roll, QC re-runs only for the segments whose takes changed
(_reroll deletes their qc_* keys; _ensure_qc passes exactly the missing ids
to backcheck/evaluate). The unchanged segments keep their scores, so _assess
still judges the whole language. A segment that fails to improve on a
re-roll is marked stuck and never re-rolled again — fresh dice through the
same MOS gate rarely rescue it, and each wasted round costs a full Whisper
pass over the re-rolled takes.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
import yaml

from . import manifest as M
from . import (s1_extract, s2_transcribe, s3_translate, s4_synthesize,
               s5_fit, s6_mix, s7_subtitles, s8_mux)

log = logging.getLogger("dubadabidu.autopilot")
DEFAULT_SPEC = "specs/batch.yaml"
FIXES = Path("FIXES.md")


def load_spec(path: str | None) -> dict:
    p = Path(path or DEFAULT_SPEC)
    if not p.exists():
        raise SystemExit(f"no spec at {p} — the autopilot refuses to run "
                         f"without acceptance criteria (see AUTOPILOT.md).")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _ensure_stages(cfg: dict, video: str, lang: str, mux: bool = True) -> None:
    """Run whatever the manifest says is missing, via the normal stage code.
    mux=False stops after s7 (dubbed audio + subs): the remote GPU path skips
    the mux — a video stream-copy needing the 4K source the pod never gets — and
    the orchestrator muxes locally after sync-back."""
    if not M.manifest_path(cfg, video).exists():
        s1_extract.run(cfg, video)
        s2_transcribe.run(cfg, video)
    else:
        s1_extract.run(cfg, video)  # cached no-op if stems exist
    man = M.load(cfg, video)
    if any("text" not in u["tr"].get(lang, {}) for u in man["utterances"]):
        s3_translate.run(cfg, video, [lang])
    st = M.load(cfg, video).get("stages", {})
    for flag, mod in [(f"s4_{lang}", s4_synthesize), (f"s5_{lang}", s5_fit),
                      (f"s6_{lang}", s6_mix), (f"s7_{lang}", s7_subtitles)]:
        if st.get(flag) != "done":
            mod.run(cfg, video, [lang])
    if mux:
        s8_mux.run(cfg, video, [lang])


def _ensure_qc(cfg: dict, video: str, lang: str) -> None:
    """Run QC for segments missing it OR carrying stale scores. First pass:
    everything is missing, so this is the normal full sweep. After a re-roll:
    _reroll stripped qc_* from exactly the re-rolled segments, so only those
    are re-checked.

    Stale = scored, but on different audio than the segment points at now
    (manifest.stale_qc). _ensure_stages may have re-run s5/s6 — e.g. a changed
    fit/mix setting, or a resumed run whose s6 flag was missing — which rewrites
    the placed wav that QC grades. Without this the loop would assess the new
    audio using the old audio's scores."""
    from qc import backcheck, evaluate
    man = M.load(cfg, video)
    total = len(man["utterances"])
    stale = M.stale_qc(M.video_workdir(cfg, video), man, lang)
    if stale["score"] or stale["wer"]:
        log.info("%s: re-scoring stale segments (score=%d wer=%d) — the placed "
                 "audio changed since they were graded",
                 lang, len(stale["score"]), len(stale["wer"]))
    need_wer = [u["id"] for u in man["utterances"]
                if "qc_wer" not in u["tr"][lang] or u["id"] in stale["wer"]]
    need_score = [u["id"] for u in man["utterances"]
                  if "qc_score" not in u["tr"][lang] or u["id"] in stale["score"]]
    if need_wer:
        backcheck.run(cfg, video, [lang],
                      only=None if len(need_wer) == total else need_wer)
    if need_score:
        evaluate.run(cfg, video, [lang],
                     only=None if len(need_score) == total else need_score)


def _assess(cfg: dict, video: str, lang: str, accept: dict) -> tuple[dict, list[str]]:
    from qc.batch_report import _lang_row
    man = M.load(cfg, video)
    trs = [u["tr"][lang] for u in man["utterances"]]
    row = _lang_row(cfg, Path(video).stem, lang, trs)
    fails = []
    if row["score"] is not None and row["score"] < accept["mean_score_min"]:
        fails.append(f"mean score {row['score']} < {accept['mean_score_min']}")
    if row["segs"] and 100 * row["flagged"] / row["segs"] > accept["flagged_pct_max"]:
        fails.append(f"{row['flagged']}/{row['segs']} segments flagged "
                     f"(> {accept['flagged_pct_max']}%)")
    for key, label in [("wer_bad", "wer_bad"), ("overflow", "overflow"),
                       ("overlap", "overlap")]:
        if row[key] > accept[f"{key}_max"]:
            fails.append(f"{label} {row[key]} > {accept[f'{key}_max']}")
    return row, fails


def _bad_segments(cfg: dict, video: str, lang: str) -> list[str]:
    """Segments fixable mechanically: bad WER or low score -> re-roll take.
    Human verdicts (M3 writeback) override the metrics in both directions:
    an accepted segment is settled and never re-rolled no matter its score;
    a rejected one re-rolls even when the metrics say it's fine — that
    disagreement is exactly the signal the weight re-fit learns from."""
    score_flag = cfg["qc"].get("eval", {}).get("score_flag", 0.55)
    wer_thr = cfg["qc"]["wer_flag_threshold"]
    man = M.load(cfg, video)
    bad = []
    for u in man["utterances"]:
        tr = u["tr"][lang]
        if tr.get("human_verdict") == "accept":
            continue
        if (tr.get("human_verdict") == "reject"
                or tr.get("qc_wer", 0) > wer_thr
                or tr.get("qc_score", 1.0) < score_flag):
            bad.append(u["id"])
    return bad


def _qc_snapshot(cfg: dict, video: str, lang: str,
                 ids: list[str]) -> dict[str, tuple]:
    man = M.load(cfg, video)
    return {u["id"]: (u["tr"][lang].get("qc_wer"), u["tr"][lang].get("qc_score"))
            for u in man["utterances"] if u["id"] in ids}


def _stuck_after(cfg: dict, video: str, lang: str, rolled: list[str],
                 prev: dict[str, tuple]) -> set[str]:
    """ids from `rolled` that are still bad AND whose metrics didn't move:
    neither WER dropped nor composite score rose vs. the deleted take. A
    re-roll is already best-of-N through the MOS gate, so a flat round means
    the distribution itself is the problem (text/ref), not the dice — another
    round would burn a Whisper pass for the same result. Metrics are stored
    rounded to 3 decimals, so strict comparison is a real change, not float
    noise."""
    still_bad = set(_bad_segments(cfg, video, lang))
    man = M.load(cfg, video)
    cur = {u["id"]: u["tr"][lang] for u in man["utterances"]}
    out = set()
    for sid in rolled:
        if sid not in still_bad:
            continue  # cleared — that's progress
        w0, s0 = prev.get(sid, (None, None))
        tr = cur[sid]
        wer_down = (w0 is not None and tr.get("qc_wer") is not None
                    and tr["qc_wer"] < w0)
        score_up = (s0 is not None and tr.get("qc_score") is not None
                    and tr["qc_score"] > s0)
        if not (wer_down or score_up):
            out.add(sid)
    return out


def _reroll(cfg: dict, video: str, lang: str, ids: list[str],
            mux: bool = True) -> None:
    """Delete the cached takes for `ids`; s4/s5/s6 re-synthesize only those.
    WER-flagged segments get tr.reroll_wer: s4 then back-transcribes every
    fresh take and vetoes hallucinated ones — re-rolling on the metric that
    actually failed instead of hoping the MOS-gated dice land differently."""
    wer_thr = cfg["qc"]["wer_flag_threshold"]
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    for u in man["utterances"]:
        if u["id"] not in ids:
            continue
        tr = u["tr"][lang]
        if tr.get("qc_wer", 0) > wer_thr:
            tr["reroll_wer"] = True
        for key in ("synth", "fitted", "placed"):
            if tr.get(key):
                (wd / tr[key]).unlink(missing_ok=True)
        for key in list(tr):
            if key.startswith("qc_"):
                del tr[key]
        # the verdict/rating described the take we just deleted, not the
        # fresh one (a stale reject would re-roll forever); the accumulated
        # row in ratings_<lang>.json keeps the historical pair for the re-fit
        tr.pop("human_verdict", None)
        tr.pop("human_rating", None)
    M.save(cfg, video, man)
    for mod in (s4_synthesize, s5_fit, s6_mix, s7_subtitles):
        mod.run(cfg, video, [lang])
    if mux:
        s8_mux.run(cfg, video, [lang])


def _log_fix(entry: str) -> None:
    header = "# FIXES — autopilot playbook (symptom -> fix -> outcome)\n\n"
    if not FIXES.exists():
        FIXES.write_text(header, encoding="utf-8")
    with FIXES.open("a", encoding="utf-8") as f:
        f.write(entry.rstrip() + "\n")


def _escalations(fails: list[str]) -> list[str]:
    out = []
    for f in fails:
        if "overflow" in f or "overlap" in f:
            out.append(f"{f} -> needs shorter text: edit tr.<lang>.text in "
                       f"the manifest (or check TRANSLATE_API_KEY so s5's "
                       f"LLM rescue can run), then re-run from s4")
        elif "score" in f or "flagged" in f:
            out.append(f"{f} -> re-rolls exhausted; likely ref/param issue: "
                       f"try `dubadabidu preamble <video>` (ref re-pick) or "
                       f"`tune`")
        else:
            out.append(f"{f} -> re-rolls exhausted; inspect the review page")
    return out


def run(cfg: dict, video: str, langs: list[str],
        spec_path: str | None = None, mux: bool = True) -> bool:
    spec = load_spec(spec_path)
    accept, budget = spec["accept"], spec["budget"]
    langs = spec.get("policy", {}).get("langs") or langs
    name = Path(video).stem
    all_ok = True

    for lang in langs:
        _ensure_stages(cfg, video, lang, mux=mux)
        _ensure_qc(cfg, video, lang)
        row, fails = _assess(cfg, video, lang, accept)
        rounds = 0
        stuck: set[str] = set()
        while fails and rounds < int(budget["max_reroll_rounds"]):
            # skip stuck segments: a flat re-roll already proved fresh dice
            # don't help them, so re-rolling again just wastes budget
            bad = [b for b in _bad_segments(cfg, video, lang)
                   if b not in stuck]
            if not bad:  # nothing (left) that's mechanically fixable
                break
            rounds += 1
            log.info("%s/%s round %d: re-rolling %s (fails: %s)",
                     name, lang, rounds, bad, fails)
            before = {f.split(" ")[0] for f in fails}
            prev = _qc_snapshot(cfg, video, lang, bad)
            _reroll(cfg, video, lang, bad, mux=mux)
            _ensure_qc(cfg, video, lang)
            row, fails = _assess(cfg, video, lang, accept)
            newly_stuck = _stuck_after(cfg, video, lang, bad, prev)
            stuck |= newly_stuck
            _log_fix(f"- {name}/{lang} round {rounds}: re-rolled {bad} for "
                     f"{sorted(before)} -> "
                     f"{'PASS' if not fails else 'still: ' + '; '.join(fails)}"
                     + (f" (no progress, now stuck: {sorted(newly_stuck)})"
                        if newly_stuck else ""))

        verdict = "PASS" if not fails else "ESCALATE"
        escalations = _escalations(fails) if fails else []
        if fails and stuck:
            escalations.append(
                f"stuck segments {sorted(stuck)}: re-roll didn't move WER or "
                f"score — the take distribution is the problem, not the dice; "
                f"edit tr.<lang>.text in the manifest or re-pick the ref "
                f"(`dubadabidu preamble <video>`)")
        result = {"video": name, "lang": lang, "verdict": verdict,
                  "row": row, "reroll_rounds": rounds,
                  "stuck": sorted(stuck),
                  "escalations": escalations}
        out = M.video_workdir(cfg, video) / f"autopilot_{lang}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[autopilot] {name}/{lang}: {verdict}  "
              f"(score={row['score']} flagged={row['flagged']}/{row['segs']} "
              f"wer_bad={row['wer_bad']} overflow={row['overflow']} "
              f"overlap={row['overlap']}, {rounds} fix rounds)")
        for e in result["escalations"]:
            print(f"  !! {e}")
        all_ok = all_ok and verdict == "PASS"
    return all_ok


def main(cfg: dict, videos: list[str], langs: list[str],
         spec_path: str | None, mux: bool = True) -> None:
    from qc import batch_report
    ok = True
    for v in videos:
        ok = run(cfg, v, langs, spec_path, mux=mux) and ok
    batch_report.run(cfg, None)
    print(f"\n[autopilot] batch verdict: {'PASS' if ok else 'ESCALATE'}")
    sys.exit(0 if ok else 1)
