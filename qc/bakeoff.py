"""Engine bake-off (IMPROVEMENT_PLAN Phase C, turnkey).

`dubadabidu bakeoff <video> --langs en` synthesizes the SAME subset of segments
with every candidate engine and scores them head-to-head, so the decision is
made by the harness, not by reputation. Output per language:
  work/<video>/bakeoff/bakeoff_<lang>.md    scorecard + PASS/FAIL vs incumbent
  work/<video>/bakeoff/bakeoff_<lang>.html  per-segment audio for ALL engines
                                            side by side + your real UA slice

Methodology — the honest cross-engine metric:
  Per-ref calibrated similarity is biased ACROSS refs/engines (evaluate.py), so
  the bake-off scores each take's ECAPA cosine against a COMMON anchor: the mean
  embedding of your REAL UA voice (ground truth). MOS and f0 are engine-
  independent. `takes` takes per segment are averaged to beat autoregressive
  take-to-take variance (same method that validated the ref A/B).

Decision gate (inherited invariant): a challenger wins a language only if it
beats the incumbent (chatterbox) on sim-to-real AND MOS — or ties and wins the
ear on the HTML page. French additionally needs a native-speaker listen.

Engines that aren't installed (cosyvoice/indextts are GPU-only git-clone deps)
are reported as unavailable and skipped, so a partial bake-off still runs.
"""
from __future__ import annotations
import html
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402

log = logging.getLogger("dubadabidu.qc.bakeoff")

INCUMBENT = "chatterbox"


def beats_incumbent(challenger: dict, incumbent: dict,
                    sim_eps: float = 0.0, mos_eps: float = 0.0) -> bool:
    """True if challenger wins on BOTH sim-to-real and MOS (the adoption gate).
    eps lets a caller demand a margin rather than a bare tie-break."""
    return (challenger["sim"] >= incumbent["sim"] + sim_eps
            and challenger["mos"] >= incumbent["mos"] + mos_eps)


def _engine_cfg(base_tts: dict, engine: str) -> dict:
    """tts config for one candidate: base (merged with the video's tts_overrides
    by the caller) + engine + that engine's defaults."""
    t = {**base_tts, "engine": engine, "engine_by_lang": {}}
    if engine == "cosyvoice":
        t.setdefault("cosyvoice_mode", "cross_lingual")  # UA-ref safe default
    return t


def _mean_stat(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def run(cfg: dict, video: str, langs: list[str]) -> None:
    import torch
    from pipeline.tts_engine import synthesize
    from pipeline.tune import _subset
    from qc import metrics as X
    from qc.evaluate import _ua_slices
    from qc.review_page import _ua_slice

    bcfg = cfg.get("bakeoff", {})
    engines = bcfg.get("engines", [INCUMBENT, "cosyvoice"])
    takes = int(bcfg.get("takes", 3))
    n = int(bcfg.get("subset_size", 6))

    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    base_tts = {**cfg["tts"], **man.get("tts_overrides", {})}
    bo = wd / "bakeoff"
    bo.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        subset = _subset([u for u in man["utterances"]
                          if u["tr"].get(lang, {}).get("text")], n)
        if not subset:
            raise SystemExit(f"no {lang} translations — run s3 first.")
        anchor = torch.stack([X.ecapa_embed(w)
                              for w in _ua_slices(wd, man["utterances"])]).mean(0)

        per_engine: dict[str, dict] = {}
        seg_audio: dict[str, dict[str, str]] = {u["id"]: {} for u in subset}
        for engine in engines:
            t = _engine_cfg(base_tts, engine)
            seg_dir = bo / "seg" / engine / lang
            seg_dir.mkdir(parents=True, exist_ok=True)
            rows, unavailable = [], None
            for u in subset:
                text = u["tr"][lang].get("fitted_text") or u["tr"][lang]["text"]
                sims, moss, f0s = [], [], []
                first = None
                for k in range(takes):
                    w = seg_dir / f"{u['id']}_t{k}.wav"
                    try:
                        synthesize(text, lang, w, t, retries=1)
                    except FileNotFoundError as e:  # engine not installed
                        unavailable = str(e).split(" — ")[0]
                        break
                    first = first or w
                    sims.append(X.cosine(anchor, X.ecapa_embed(w)))
                    moss.append(X.mos_min_window(w))
                    f0s.append(X.f0_semitone_std(w))
                if unavailable:
                    break
                # relative to bo/ (where the .html lives), not wd/
                seg_audio[u["id"]][engine] = str(first.relative_to(bo))
                rows.append({"sim": sum(sims) / len(sims),
                             "mos": sum(moss) / len(moss),
                             "f0": sum(f0s) / len(f0s)})
            if unavailable:
                per_engine[engine] = {"unavailable": unavailable}
                log.warning("%s unavailable: %s", engine, unavailable)
                continue
            per_engine[engine] = {"sim": _mean_stat(rows, "sim"),
                                  "mos": _mean_stat(rows, "mos"),
                                  "f0": _mean_stat(rows, "f0"),
                                  "segs": len(rows)}
            log.info("%s/%s: %s", engine, lang, per_engine[engine])

        _write_reports(bo, video, lang, subset, per_engine, seg_audio, wd,
                       _ua_slice)


def _verdict(engine: str, stats: dict, inc: dict | None) -> str:
    if "unavailable" in stats:
        return "n/a (not installed)"
    if engine == INCUMBENT or not inc or "unavailable" in inc:
        return "incumbent" if engine == INCUMBENT else "no incumbent baseline"
    return "ADOPT" if beats_incumbent(stats, inc) else "keep incumbent"


def _write_reports(bo, video, lang, subset, per_engine, seg_audio, wd,
                   ua_slice_fn) -> None:
    import os
    name = Path(video).stem
    inc = per_engine.get(INCUMBENT)
    engines = list(per_engine)

    lines = [f"# bake-off — {name} / {lang}", "",
             "sim = ECAPA cosine to your REAL UA voice (higher=more like you); "
             "mos = windowed Distill-MOS; f0st = pitch liveliness. "
             f"Averaged over takes on {len(subset)} segments.", "",
             "| engine | sim→real | mos | f0st | verdict |",
             "|---|---|---|---|---|"]
    for e in engines:
        s = per_engine[e]
        if "unavailable" in s:
            lines.append(f"| {e} | - | - | - | {_verdict(e, s, inc)} |")
        else:
            lines.append(f"| {e} | {s['sim']} | {s['mos']} | {s['f0']} "
                         f"| {_verdict(e, s, inc)} |")
    lines += ["", "Adoption gate: a challenger must beat chatterbox on sim→real "
              "AND mos, or tie and win the ear on the .html page. Then set "
              "`tts.engine_by_lang: {" + lang + ": <winner>}`."]
    (bo / f"bakeoff_{lang}.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")

    # side-by-side listening page (the ear test)
    avail = [e for e in engines if "unavailable" not in per_engine[e]]
    segs_html = []
    for u in subset:
        ua = os.path.relpath(ua_slice_fn(wd, u), bo)   # e.g. ../qc_ua/u0001.wav
        players = "".join(
            f'<div><b>{e}</b><br><audio controls preload="none" '
            f'src="{seg_audio[u["id"]].get(e, "")}"></audio></div>'
            for e in avail)
        segs_html.append(
            f'<div class="seg"><div class="txt">{u["id"]}: '
            f'{html.escape((u["tr"][lang].get("fitted_text") or u["tr"][lang]["text"]))}'
            f'</div><div class="row">{players}'
            f'<div><b>you (UA)</b><br><audio controls preload="none" '
            f'src="{ua}"></audio></div></div></div>')
    css = ("body{font:14px/1.5 -apple-system,sans-serif;max-width:1000px;"
           "margin:2rem auto;background:#111;color:#ddd}.seg{border:1px solid "
           "#333;border-radius:8px;padding:.8rem;margin:.8rem 0}.row{display:"
           "flex;gap:1rem;flex-wrap:wrap}.txt{color:#aaa;margin-bottom:.5rem}"
           "audio{height:2rem;max-width:220px}")
    page = (f"<!doctype html><meta charset='utf-8'><title>bakeoff {name} "
            f"{lang}</title><style>{css}</style><body>"
            f"<h1>bake-off — {name} / {lang}</h1>"
            f"<pre>{html.escape(chr(10).join(lines))}</pre>"
            f"{''.join(segs_html)}")
    (bo / f"bakeoff_{lang}.html").write_text(page, encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[bakeoff] listen: {bo / f'bakeoff_{lang}.html'}")
