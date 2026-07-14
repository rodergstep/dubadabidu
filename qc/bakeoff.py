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
  independent. Two more dimensions, because sim/mos/f0 are all blind to them:
  wer (back-transcription vs the requested text — catches hallucination and
  dropped/invented words) and pace (synth_dur / source slot — natural speaking
  rate, the overflow/timing risk s5 has to stretch away). `takes` takes per
  segment are averaged to beat autoregressive take-to-take variance (same method
  that validated the ref A/B) — and the spread that averaging hides is itself
  reported: mos± (take-to-take MOS std = how much best_of an engine needs) and
  s/take (median synth wall-clock = engine speed/cost for the batch).

Decision gate (inherited invariant): a challenger wins a language only if it
beats the incumbent (chatterbox) on sim-to-real AND MOS, AND does not regress
wer (an intelligibility VETO) — or ties and wins the ear on the HTML page. pace
is reported, not gated (s5 retimes within limits). French additionally needs a
native-speaker listen.

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
                    sim_eps: float = 0.0, mos_eps: float = 0.0,
                    wer_eps: float = 0.02) -> bool:
    """The adoption gate. A challenger wins only if it beats the incumbent on
    BOTH sim-to-real and MOS AND does not regress intelligibility: WER is a VETO,
    not a win condition — a fluent, on-voice take that drops or invents words is
    still a bad dub, and sim/mos/f0 are all blind to it. pace is deliberately NOT
    gated (s5 retimes within limits; the ear/HTML page judges timing feel).
    sim/mos/wer_eps let a caller demand a margin. WER veto is skipped when either
    side lacks a `wer` key, so a partial/legacy scorecard still gates on sim+mos."""
    if (challenger.get("wer") is not None and incumbent.get("wer") is not None
            and challenger["wer"] > incumbent["wer"] + wer_eps):
        return False   # intelligibility regressed beyond tolerance -> disqualified
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
    import statistics
    import time
    import torch
    import soundfile as sf
    from pipeline.tts_engine import synthesize
    from pipeline.tune import _subset
    from qc import metrics as X
    from qc.backcheck import segment_wer
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
            synth_secs = []   # raw wall-clock of every synth call (engine speed)
            for u in subset:
                text = u["tr"][lang].get("fitted_text") or u["tr"][lang]["text"]
                # source speech slot: same text goes to every engine, so pace
                # DIFFERENCES between engines isolate the engine's speaking rate
                # (the shared translation-length bias cancels in the comparison).
                slot = max(u["end"] - u["start"], 0.1)
                sims, moss, f0s, wers, paces = [], [], [], [], []
                first = None
                for k in range(takes):
                    w = seg_dir / f"{u['id']}_t{k}.wav"
                    try:
                        t0 = time.perf_counter()
                        synthesize(text, lang, w, t, retries=1)
                        synth_secs.append(time.perf_counter() - t0)
                    except FileNotFoundError as e:  # engine not installed
                        unavailable = str(e).split(" — ")[0]
                        break
                    first = first or w
                    sims.append(X.cosine(anchor, X.ecapa_embed(w)))
                    moss.append(X.mos_min_window(w))
                    f0s.append(X.f0_semitone_std(w))
                    # WER: back-transcribe and compare to the text we asked for —
                    # catches hallucination/dropped words that sim/mos/f0 miss.
                    wers.append(segment_wer(cfg, text, w, lang))
                    # pace: synth_dur / source slot (>1 = slower than source,
                    # overflow-prone; s5 would have to stretch it to fit).
                    paces.append(sf.info(str(w)).duration / slot)
                if unavailable:
                    break
                # relative to bo/ (where the .html lives), not wd/
                seg_audio[u["id"]][engine] = str(first.relative_to(bo))
                rows.append({"sim": sum(sims) / len(sims),
                             "mos": sum(moss) / len(moss),
                             "f0": sum(f0s) / len(f0s),
                             "wer": sum(wers) / len(wers),
                             "pace": sum(paces) / len(paces),
                             # take-to-take MOS spread for THIS segment: how much a
                             # single roll's quality is a dice-throw (drives how much
                             # best_of the engine needs). 0 when takes==1.
                             "mos_sd": statistics.stdev(moss) if len(moss) > 1
                                       else 0.0})
            if unavailable:
                per_engine[engine] = {"unavailable": unavailable}
                log.warning("%s unavailable: %s", engine, unavailable)
                continue
            per_engine[engine] = {"sim": _mean_stat(rows, "sim"),
                                  "mos": _mean_stat(rows, "mos"),
                                  "f0": _mean_stat(rows, "f0"),
                                  "wer": _mean_stat(rows, "wer"),
                                  "pace": _mean_stat(rows, "pace"),
                                  "mos_sd": _mean_stat(rows, "mos_sd"),
                                  # MEDIAN synth wall-clock per take — median so the
                                  # one-time model load on the first call (and any
                                  # occasional slow roll) doesn't skew steady-state
                                  # engine speed.
                                  "s_take": round(statistics.median(synth_secs), 2)
                                            if synth_secs else 0.0,
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
             "mos = windowed Distill-MOS; f0st = pitch liveliness (anti-monotony); "
             "wer = back-transcription WER, LOWER=fewer hallucinations/dropped "
             "words; pace = synth_dur / source-slot, >1 = speaks slower than the "
             "source (overflow-prone; same text per engine, so it's the engine); "
             "mos± = take-to-take MOS std (reliability — LOWER = a single roll is "
             "less of a dice-throw, needs less best_of); s/take = median seconds "
             "per synthesis (engine speed/cost, model-load excluded via median). "
             f"Averaged over takes on {len(subset)} segments.", "",
             "| engine | sim→real | mos | f0st | wer | pace | mos± | s/take "
             "| verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in engines:
        s = per_engine[e]
        if "unavailable" in s:
            lines.append(f"| {e} | - | - | - | - | - | - | - "
                         f"| {_verdict(e, s, inc)} |")
        else:
            lines.append(f"| {e} | {s['sim']} | {s['mos']} | {s['f0']} "
                         f"| {s['wer']} | {s['pace']} | {s['mos_sd']} "
                         f"| {s['s_take']} | {_verdict(e, s, inc)} |")
    lines += ["", "Adoption gate: a challenger must beat chatterbox on sim→real "
              "AND mos, AND not regress wer beyond tolerance (intelligibility "
              "veto), or tie and win the ear on the .html page. pace, mos± and "
              "s/take are informational (timing feel / reliability / cost) — they "
              "inform the choice but do not gate it. Then set "
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
