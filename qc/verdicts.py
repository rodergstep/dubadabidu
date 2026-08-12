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
from qc import stress_words as SW  # noqa: E402

log = logging.getLogger("dubadabidu.qc.verdicts")

QC_FEATURES = ("qc_score", "qc_sim2", "qc_sim_cal", "qc_mos", "qc_mos_min",
               "qc_f0st", "qc_wer", "tempo", "fit")


def _seg_hash(utterances: list[dict], lang: str) -> str:
    """Must mirror review_page.py: hash over ID-sorted ids+starts.

    SORTED BY ID, NOT BY SCORE, and the difference is the whole point. Both
    sides used to hash the worst-first DISPLAY order, so the key changed
    whenever any qc_score moved — re-running `evaluate`, re-rolling one
    segment, or editing qc.eval.weights. The ingest then rejected a perfectly
    good export with "the video was re-segmented", which is not what happened
    and sends the reviewer looking in the wrong place. Worse, the review page
    and the stale-metrics error both TELL the user to run `dubadabidu qc` and
    re-review — the exact action that invalidated their ratings.

    Sorting by id makes this what it claims to be: a hash of the utterance
    BOUNDARIES. The genuinely dangerous case — ratings taken against audio that
    has since changed — is caught separately and precisely by M.stale_qc below,
    which compares a content hash of the graded wav.

    `lang` is unused now; kept in the signature because review_page calls the
    same shape and one of the two silently not taking it is how these two
    drifted apart the first time.
    """
    us = sorted(utterances, key=lambda u: u["id"])
    return hashlib.sha1(
        ",".join(u["id"] + str(u["start"]) for u in us).encode()).hexdigest()[:6]


def run(cfg: dict, video: str, export_file: str) -> None:
    data = json.loads(Path(export_file).read_text(encoding="utf-8"))
    key = data.get("key", "")
    ratings = {k: v for k, v in (data.get("ratings") or {}).items()
               if isinstance(v, (int, float))}
    verdicts = data.get("verdicts") or {}
    # Exports predating word marking simply have no `words` key.
    words = {k: v for k, v in (data.get("words") or {}).items()
             if isinstance(v, list) and v}
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
    # Every row carries a QC_FEATURES snapshot — that snapshot IS the re-fit's
    # training input. Ingesting scores that describe audio s5/s6 has since
    # rewritten would teach the weight fit to predict the human's judgment of
    # one take from another take's metrics. Refuse, same as above: the export
    # file is on disk, so nothing is lost by re-scoring and running again.
    # word marks describe a specific TAKE, so they go through the same staleness
    # gate: if the audio changed, the reviewer marked a word in a rendition that
    # no longer exists and the table would learn a defect from the wrong take.
    rated = set(ratings) | set(verdicts) | set(words)
    stale = M.stale_qc(M.video_workdir(cfg, video), man, lang)
    hit = sorted(rated & (set(stale["score"]) | set(stale["wer"])))
    if hit:
        raise SystemExit(
            f"{len(hit)} rated segments carry STALE qc metrics (the audio "
            f"changed after they were scored): "
            f"{hit[:8]}{' ...' if len(hit) > 8 else ''}\n"
            f"The stored metrics describe one take and these ratings describe "
            f"another, so the pair would teach the re-fit a relationship that "
            f"never existed. Re-score AND re-review before ingesting:\n"
            f"  dubadabidu qc {video} --langs {lang}\n"
            f"  dubadabidu review {video} --langs {lang}")

    rows_path = Path(f"ratings_{lang}.json")
    rows = (json.loads(rows_path.read_text(encoding="utf-8"))
            if rows_path.exists() else [])
    # DO NOT re-key the whole file into a dict. It used to be
    #     by_key = {(r["video"], r["id"]): r for r in rows}
    # which silently collapsed every PRE-EXISTING (video, id) duplicate,
    # including rows this ingest never touches. qc/blind.py and qc/compare.py
    # legitimately write several rows per (video, id) — the same segment rated
    # under different variants and takes — so a single `verdicts` run deleted
    # 36 of 114 accumulated rows, each a distinct measurement with its own
    # qc_mos/qc_f0st. Measured 2026-08-11, recovered from git.
    #
    # That is the worst possible failure mode for this file: refit is starved
    # of ratings by design (qc/blind.py), the loss is silent, and the row count
    # printed below still went UP because the ingest added more than it ate.
    #
    # Replace only what this export actually re-rates; leave everything else
    # byte-identical.
    new_rows: dict[tuple[str, str], dict] = {}

    n_man = 0
    marked: list[tuple[str, dict, str]] = []   # (uid, mark, spoken text)
    n_stale_marks = 0
    for u in man["utterances"]:
        uid = u["id"]
        rating, verdict = ratings.get(uid), verdicts.get(uid)
        marks = words.get(uid) or []
        if rating is None and verdict is None and not marks:
            continue
        tr = u["tr"][lang]
        if rating is not None:
            tr["human_rating"] = rating
        if verdict is not None:
            tr["human_verdict"] = verdict
        if marks:
            spoken = tr.get("fitted_text") or tr.get("text", "")
            ok, stale = SW.verify(spoken, marks)
            n_stale_marks += len(stale)
            if stale:
                log.warning("%s: %d word mark(s) no longer match the text "
                            "(edited since review?) — dropped: %s",
                            uid, len(stale), [m.get("w") for m in stale])
            if ok:
                tr["stress_words"] = [m["w"] for m in ok]
                marked += [(uid, m, spoken) for m in ok]
            else:
                tr.pop("stress_words", None)
        n_man += 1
        row = {"video": stem, "lang": lang, "id": uid,
               "rating": rating, "verdict": verdict,
               "text": tr.get("fitted_text", tr.get("text", ""))}
        row.update({k: tr[k] for k in QC_FEATURES if k in tr})
        new_rows[(stem, uid)] = row  # replace: latest verdict wins
    M.save(cfg, video, man)

    n_replaced = sum(1 for r in rows if (r["video"], r["id"]) in new_rows)
    rows = [r for r in rows if (r["video"], r["id"]) not in new_rows] \
        + list(new_rows.values())
    rows = sorted(rows, key=lambda r: (r["video"], r["id"]))
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    n_rej = sum(1 for r in rows if r.get("verdict") == "reject")
    print(f"[verdicts] {stem}/{lang}: {n_man} segments written to manifest; "
          f"{rows_path} now holds {len(rows)} rows ({n_rej} rejects) "
          f"for the weight re-fit "
          f"({len(new_rows) - n_replaced} added, {n_replaced} replaced).")
    if marked or n_stale_marks:
        lex_path, n_words, n_new = _update_lexicon(stem, lang, marked)
        print(f"[verdicts] {len(marked)} word mark(s) -> {lex_path} "
              f"({n_words} distinct words, {n_new} new this ingest"
              + (f", {n_stale_marks} dropped as stale" if n_stale_marks else "")
              + ")")


def _update_lexicon(stem: str, lang: str,
                    marked: list[tuple[str, dict, str]]) -> tuple[Path, int, int]:
    """Accumulate marked words into stress_lexicon_<lang>.json.

    THE TABLE IS THE POINT. Automated stress detection is closed (FINDINGS 2.1),
    and §2.1j closed selection too, because the errors are partly SYSTEMATIC —
    for some words qwen is reliably wrong, so the majority placement is wrong
    and no ranking rule can reach them. Systematic also means enumerable: a
    word that is reliably wrong needs fixing ONCE, and this file is where the
    finite set of them accumulates across a 20-video course.

    Counts, not a set: a word marked in eight segments across three videos is a
    different proposition from one marked once, and remediation should start at
    the top. Occurrences carry the sentence so a fix can be judged in context —
    §2.1h found that some words (`цвета`: gen. sg. vs nom. pl.) have no single
    right answer without one.

    Keyed by qc.stress_words.lexicon_key (case-folded, marks stripped) but NOT
    lemmatised: a TTS can be right about one inflected form and wrong about
    another, and the surface form is what every remediation route matches on.
    """
    lex_path = Path(f"stress_lexicon_{lang}.json")
    lex = (json.loads(lex_path.read_text(encoding="utf-8"))
           if lex_path.exists() else {})
    n_new = 0
    for uid, m, spoken in marked:
        key = SW.lexicon_key(m["w"])
        if not key:
            continue
        if key not in lex:
            lex[key] = {"marked": 0, "forms": [], "occurrences": []}
            n_new += 1
        e = lex[key]
        where = {"video": stem, "id": uid, "form": m["w"], "context": spoken}
        # replace rather than append when the same segment is re-reviewed, so a
        # second ingest of the same page cannot inflate the count
        e["occurrences"] = [o for o in e["occurrences"]
                            if not (o.get("video") == stem
                                    and o.get("id") == uid
                                    and o.get("form") == m["w"])] + [where]
        e["marked"] = len(e["occurrences"])
        e["forms"] = sorted({o["form"] for o in e["occurrences"]})
    lex = dict(sorted(lex.items(), key=lambda kv: (-kv[1]["marked"], kv[0])))
    lex_path.write_text(json.dumps(lex, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return lex_path, len(lex), n_new
