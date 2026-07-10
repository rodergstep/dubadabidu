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

Note: after a re-roll, QC re-runs for the whole language (backcheck +
evaluate are per-language sweeps). Fine for short fixtures; per-segment QC
is an M2 optimization.
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


def _ensure_stages(cfg: dict, video: str, lang: str) -> None:
    """Run whatever the manifest says is missing, via the normal stage code."""
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
    s8_mux.run(cfg, video, [lang])


def _ensure_qc(cfg: dict, video: str, lang: str) -> None:
    from qc import backcheck, evaluate
    man = M.load(cfg, video)
    trs = [u["tr"][lang] for u in man["utterances"]]
    if any("qc_wer" not in t for t in trs):
        backcheck.run(cfg, video, [lang])
    if any("qc_score" not in t for t in trs):
        evaluate.run(cfg, video, [lang])


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


def _reroll(cfg: dict, video: str, lang: str, ids: list[str]) -> None:
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
        spec_path: str | None = None) -> bool:
    spec = load_spec(spec_path)
    accept, budget = spec["accept"], spec["budget"]
    langs = spec.get("policy", {}).get("langs") or langs
    name = Path(video).stem
    all_ok = True

    for lang in langs:
        _ensure_stages(cfg, video, lang)
        _ensure_qc(cfg, video, lang)
        row, fails = _assess(cfg, video, lang, accept)
        rounds = 0
        while fails and rounds < int(budget["max_reroll_rounds"]):
            bad = _bad_segments(cfg, video, lang)
            if not bad:  # failures aren't the mechanically fixable kind
                break
            rounds += 1
            log.info("%s/%s round %d: re-rolling %s (fails: %s)",
                     name, lang, rounds, bad, fails)
            before = {f.split(" ")[0] for f in fails}
            _reroll(cfg, video, lang, bad)
            _ensure_qc(cfg, video, lang)
            row, fails = _assess(cfg, video, lang, accept)
            _log_fix(f"- {name}/{lang} round {rounds}: re-rolled {bad} for "
                     f"{sorted(before)} -> "
                     f"{'PASS' if not fails else 'still: ' + '; '.join(fails)}")

        verdict = "PASS" if not fails else "ESCALATE"
        result = {"video": name, "lang": lang, "verdict": verdict,
                  "row": row, "reroll_rounds": rounds,
                  "escalations": _escalations(fails) if fails else []}
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
         spec_path: str | None) -> None:
    from qc import batch_report
    ok = True
    for v in videos:
        ok = run(cfg, v, langs, spec_path) and ok
    batch_report.run(cfg, None)
    print(f"\n[autopilot] batch verdict: {'PASS' if ok else 'ESCALATE'}")
    sys.exit(0 if ok else 1)
