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

Tuning parity (IMPROVEMENT_PLAN Phase C: "tune-lite ... on BOTH engines"):
before the head-to-head, every engine runs a small parameter sweep and enters
the comparison at ITS OWN best point. Without this the incumbent arrived with a
tuned cfg_weight/exaggeration from a real tune run while every challenger got
library defaults — an untuned-vs-tuned comparison whose predictable "keep
incumbent" would have looked like evidence. Grids are config-driven
(bakeoff.grids); an engine whose grid is a single point costs nothing extra and
is reported as such, so the comparison stays auditable either way.

Decision gate (inherited invariant): a challenger wins a language only if it
beats the incumbent (chatterbox) on sim-to-real AND MOS, AND does not regress
wer (an intelligibility VETO) — or ties and wins the ear on the HTML page. pace
is reported, not gated (s5 retimes within limits). French additionally needs a
native-speaker listen.

Engines that aren't installed (qwen is a GPU-only git-clone dep)
are reported as unavailable and skipped, so a partial bake-off still runs.
"""
from __future__ import annotations
import html
import json
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


def variant_key(engine: str, t: dict, lang: str) -> str:
    """Scorecard/audio key for an engine AS CONFIGURED.

    Rows were keyed on the engine NAME alone, so testing the same engine with a
    different capability enabled REPLACED the earlier row instead of sitting
    beside it — and overwrote its audio. That is right for re-running an
    identical config and wrong for a variant, which is exactly the comparison
    worth seeing: qwen's CUDA-graph decode path against its stock one.

    Only settings that materially change what the engine DOES earn a suffix."""
    from pipeline.manifest import resolve_engine
    eng = resolve_engine(t, lang)
    suffix = ""
    # qwen_fast swaps the decode implementation (CUDA graphs). The whole point
    # of the A/B is to see both rows AND keep both sets of audio for a listen —
    # without a suffix the second run overwrites the first, which is the exact
    # failure this function was written for.
    if t.get("qwen_fast") and eng == "qwen":
        suffix = "+fast"
    # ICL clone (ref audio + its transcript in context) vs x_vector_only
    # (speaker embedding alone). Documented to improve speaker similarity —
    # our weakest metric — at the risk of source-accent bleed, so both rows
    # must survive for the ear to arbitrate.
    if eng == "qwen" and t.get("qwen_x_vector_only") is False:
        suffix += "+icl"
    # model SIZE changes what the engine is, not how it is configured. Without
    # this a 0.6B run overwrites the 1.7B row and its audio — the same failure
    # that cost us the plain-indextts recordings and nearly cost the
    # qwen fast-vs-stock comparison.
    if eng == "qwen" and "0.6B" in str(t.get("qwen_model_dir", "")):
        suffix += "+0.6B"
    return engine + suffix


def _engine_cfg(base_tts: dict, engine: str) -> dict:
    """tts config for one candidate: base (merged with the video's tts_overrides
    by the caller) + engine + that engine's defaults."""
    return {**base_tts, "engine": engine, "engine_by_lang": {}}


# Per-engine tune-lite grids. Only knobs that change a single take's OUTPUT
# belong here (they must be in manifest.synth_hash, or the cache would serve
# one grid point's audio for another). Take-SELECTION params (best_of,
# min_f0st, ...) are deliberately absent: the bake-off measures engines, not
# selection policy, and every engine gets the same selection downstream.
#
# Everything here is UA-REFERENCE SAFE. Modes needing a target-language ref
# transcript are excluded on purpose: qwen's full clone mode cannot tokenize a
# Ukrainian ref (see tts_engine), so sweeping it would only produce failures.
#
# A single-point grid means "already tuned, nothing to sweep" and costs nothing:
#   chatterbox — cfg_weight is FIXED at the config value (0.0 is mandatory for a
#     UA ref -> non-UA target, re-validated by tune R2), and exaggeration does
#     not reliably move quality (measured 2026-07-14), so the incumbent enters
#     at its tune-selected point. Widen via bakeoff.grids to re-open it.
#   qwen — x_vector_only is forced by the UA ref; nothing left to sweep.
ENGINE_GRIDS: dict[str, dict[str, list]] = {
    "chatterbox": {},
    "qwen": {},
    "edge": {},
}


def _grid_points(grid: dict[str, list]) -> list[dict]:
    """Grid -> list of override dicts (cartesian product). Empty grid -> one
    empty override, i.e. 'run at the configured defaults'."""
    import itertools
    keys = sorted(k for k, v in grid.items() if v)
    if not keys:
        return [{}]
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(grid[k] for k in keys))]


def _fmt_point(over: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(over.items())) or "(defaults)"


def _tune_engine(engine: str, base_t: dict, subset: list[dict], lang: str,
                 anchor, seg_root, takes: int, grid: dict) -> tuple:
    """tune-lite for ONE engine: score every grid point on a small subset and
    return its best. Scored on the two axes the adoption gate judges — cosine
    to the REAL UA voice and MOS, weighted equally — so the point that wins
    here is the point that maximizes what the gate measures.

    WER is not scored during tuning (a Whisper pass per grid point per take is
    the expensive part of the run, and these knobs are samplers/styles, not text
    edits). The tuned point still faces the gate's WER veto afterwards.

    Returns (best_overrides, trial_rows, unavailable_or_None).
    """
    from pipeline.tts_engine import synthesize
    from qc import metrics as X

    points = _grid_points(grid)
    trials = []
    last_err = None
    for i, over in enumerate(points):
        t = {**base_t, **over}
        sims, moss = [], []
        failed = None
        for u in subset:
            text = u["tr"][lang].get("fitted_text") or u["tr"][lang]["text"]
            for k in range(takes):
                w = seg_root / f"p{i}_{u['id']}_t{k}.wav"
                try:
                    synthesize(text, lang, w, t, retries=1)
                except (FileNotFoundError, RuntimeError) as e:
                    # FileNotFoundError = package/venv missing (the documented
                    # engine-unavailable contract). RuntimeError = the worker died
                    # or synthesis failed every retry — a broken install, an OOM,
                    # a full disk, or a parameter this engine build rejects.
                    #
                    # Skip THIS GRID POINT and keep going: a grid exists to
                    # explore, so a single unsupported value must not disqualify
                    # the engine — otherwise widening a grid is a gamble and the
                    # safe move is never to widen it. The engine is only
                    # unavailable if EVERY point fails (checked after the loop).
                    failed = str(e).split(" — ")[0][:120]
                    last_err = failed
                    break
                sims.append(X.cosine(anchor, X.ecapa_embed(w)))
                moss.append(X.mos_min_window(w))
            if failed:
                break
        if failed:
            log.warning("tune-lite %s/%s %s FAILED (%s) — skipping this point",
                        engine, lang, _fmt_point(over), failed)
            continue
        sim = sum(sims) / len(sims)
        mos = sum(moss) / len(moss)
        # equal weight on the gate's two axes; mos normalized 1..5 -> 0..1 so
        # neither term can dominate the other by scale alone
        score = 0.5 * sim + 0.5 * max(0.0, min(1.0, (mos - 1.0) / 4.0))
        trials.append({"point": over, "sim": round(sim, 3),
                       "mos": round(mos, 3), "score": round(score, 4)})
        log.info("tune-lite %s/%s %s -> sim %.3f mos %.2f (score %.4f)",
                 engine, lang, _fmt_point(over), sim, mos, score)
    if not trials:      # every point failed -> the engine itself is unusable
        return {}, [], last_err or "no grid point produced audio"
    best = max(trials, key=lambda r: r["score"])
    if len(trials) > 1:
        log.info("tune-lite %s/%s: %s wins %d points",
                 engine, lang, _fmt_point(best["point"]), len(trials))
    return best["point"], trials, None


def _mean_stat(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def run(cfg: dict, video: str, langs: list[str]) -> None:
    import statistics
    import time
    import torch
    import soundfile as sf
    from pipeline import engine_client
    from pipeline.tts_engine import release_models, synthesize
    from pipeline.tune import _subset
    from qc import metrics as X
    from qc.backcheck import segment_wer
    from qc.evaluate import _ua_slices
    from qc.review_page import _ua_slice

    bcfg = cfg.get("bakeoff", {})
    engines = bcfg.get("engines", [INCUMBENT, "qwen"])
    takes = int(bcfg.get("takes", 3))
    n = int(bcfg.get("subset_size", 6))
    # tune-lite: per-engine grids, overridable per engine (a config grid
    # REPLACES the default for that engine — that is how you widen or close
    # one). Its subset/takes are separate and smaller: the sweep is
    # points x segments x takes, so it grows fastest.
    grids = {**ENGINE_GRIDS, **(bcfg.get("grids") or {})}
    tcfg = bcfg.get("tune") or {}
    tune_on = bool(tcfg.get("enabled", True))
    tune_n = int(tcfg.get("subset_size", 3))
    tune_takes = int(tcfg.get("takes", 2))

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
        # The anchor IS the cross-engine metric (mean embedding of the speaker's
        # REAL voice), so unlike evaluate.calibration there is no sensible
        # fallback to degrade to — fail fast with something actionable instead of
        # torch's "stack expects a non-empty TensorList". _ua_slices skips spans
        # under 1s, so this fires when every sampled utterance is shorter.
        # Matters because it would otherwise surface AFTER the engine installs,
        # i.e. at the most expensive point of a billed pod run.
        embs = [X.ecapa_embed(w) for w in _ua_slices(wd, man["utterances"])]
        if not embs:
            raise SystemExit(
                f"no usable reference slices from {wd}/vocals.wav — every "
                f"sampled utterance is under 1s, so there is no real-voice "
                f"anchor to score engines against. Check this video's "
                f"segmentation in the manifest (over-split?) and re-run s2.")
        anchor = torch.stack(embs).mean(0)

        per_engine: dict[str, dict] = {}
        tuning: dict[str, dict] = {}
        seg_audio: dict[str, dict[str, str]] = {u["id"]: {} for u in subset}
        for engine in engines:
            t = _engine_cfg(base_tts, engine)
            # key rows AND audio by the configured variant, so an
            # emotion-enabled run sits beside the plain one instead
            # of replacing it (and keeps its own wavs)
            vkey = variant_key(engine, t, lang)
            seg_dir = bo / "seg" / vkey / lang
            seg_dir.mkdir(parents=True, exist_ok=True)
            rows, unavailable = [], None

            # tune-lite FIRST, so the comparison below runs this engine at its
            # own best point rather than at library defaults. Same worker/model
            # stays loaded for both phases; shutdown happens once, after.
            grid = grids.get(engine, {})
            points = _grid_points(grid)
            if tune_on and len(points) > 1:
                tune_dir = bo / "tune" / vkey / lang
                tune_dir.mkdir(parents=True, exist_ok=True)
                over, trials, unavailable = _tune_engine(
                    engine, t, _subset(subset, tune_n), lang, anchor,
                    tune_dir, tune_takes, grid)
                if not unavailable:
                    t = {**t, **over}
                    tuning[vkey] = {"winner": over, "trials": trials,
                                      "n_points": len(points)}
            else:
                tuning[vkey] = {"winner": {}, "trials": [],
                                  "n_points": len(points),
                                  "skipped": "tuning off" if not tune_on
                                             else "single-point grid"}
            synth_secs = []   # raw wall-clock of every synth call (engine speed)
            for u in subset:
                if unavailable:   # tune-lite already proved it isn't installed
                    break
                text = u["tr"][lang].get("fitted_text") or u["tr"][lang]["text"]
                # source speech slot: same text goes to every engine, so pace
                # DIFFERENCES between engines isolate the engine's speaking rate
                # (the shared translation-length bias cancels in the comparison).
                slot = max(u["end"] - u["start"], 0.1)
                tu = t
                sims, moss, f0s, wers, paces = [], [], [], [], []
                first = None
                for k in range(takes):
                    w = seg_dir / f"{u['id']}_t{k}.wav"
                    try:
                        t0 = time.perf_counter()
                        synthesize(text, lang, w, tu, retries=1)
                        synth_secs.append(time.perf_counter() - t0)
                    except (FileNotFoundError, RuntimeError) as e:
                        # see _tune_engine: a dead worker / failed synth must
                        # disqualify THIS engine, not the whole comparison
                        unavailable = str(e).split(" — ")[0][:120]
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
                seg_audio[u["id"]][vkey] = str(first.relative_to(bo))
                rows.append({"id": u["id"],   # kept so the report can show WHICH
                                              # segments are weak, not just means
                             "sim": sum(sims) / len(sims),
                             "mos": sum(moss) / len(moss),
                             "f0": sum(f0s) / len(f0s),
                             "wer": sum(wers) / len(wers),
                             "pace": sum(paces) / len(paces),
                             # take-to-take MOS spread for THIS segment: how much a
                             # single roll's quality is a dice-throw (drives how much
                             # best_of the engine needs). 0 when takes==1.
                             "mos_sd": statistics.stdev(moss) if len(moss) > 1
                                       else 0.0})
            # free this engine before the next loads: stop its isolated-venv
            # worker (if any) and drop in-process singletons + CUDA cache —
            # otherwise engines pile up in VRAM and a 16 GB card OOMs by the
            # third challenger. qc models (ECAPA/MOS/whisper) stay resident.
            engine_client.shutdown(engine)
            release_models()
            if unavailable:
                per_engine[vkey] = {"unavailable": unavailable}
                log.warning("%s unavailable: %s", engine, unavailable)
                continue
            per_engine[vkey] = {"sim": _mean_stat(rows, "sim"),
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
                                  "segs": len(rows),
                                  # per-segment detail: the means hide WHICH
                                  # segments drag an engine down, and a 4-of-6
                                  # sample is only obvious if you can see the gaps
                                  "per_seg": {r["id"]: {k: round(r[k], 3)
                                                        for k in ("sim", "mos", "f0",
                                                                  "wer", "pace")}
                                              for r in rows}}
            log.info("%s/%s: %s", vkey, lang,
                     {k: v for k, v in per_engine[vkey].items()
                      if k != "per_seg"})

        # MERGE with earlier runs before rendering. Engines are now tested one per
        # run (each needs its own pod-sized disk), so without this every run
        # overwrote the previous engine's scorecard and the page could never show
        # two engines side by side — which defeats the point of a bake-off.
        merged, merged_tuning = _merge_results(bo, lang, per_engine, tuning)
        _write_reports(bo, video, lang, subset, merged,
                       _audio_on_disk(bo, lang, merged, subset), wd,
                       _ua_slice, merged_tuning)


def _verdict(engine: str, stats: dict, inc: dict | None) -> str:
    if "unavailable" in stats:
        return "n/a (not installed)"
    if engine == INCUMBENT or not inc or "unavailable" in inc:
        return "incumbent" if engine == INCUMBENT else "no incumbent baseline"
    return "ADOPT" if beats_incumbent(stats, inc) else "keep incumbent"


RESULTS = "results_{lang}.json"      # accumulated across runs, next to the reports


def _merge_results(bo: Path, lang: str, per_engine: dict,
                   tuning: dict) -> tuple[dict, dict]:
    """Fold this run's engines into the accumulated results and persist them.

    Engines are tested one per run (each wants the whole container disk for its
    own torch), so a run only ever holds one row. Rendering from just that row
    overwrote the previous engine's scorecard every time — the audio survived on
    disk but the comparison page could never show two engines together, which is
    the one thing a bake-off is for. This run's numbers WIN for the engines it
    measured (a re-test supersedes an older one) and every other engine is
    carried forward untouched.
    """
    p = bo / RESULTS.format(lang=lang)
    prev, prev_tune = {}, {}
    if p.exists():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            prev = saved.get("engines", {})
            prev_tune = saved.get("tuning", {})
        except json.JSONDecodeError:
            log.warning("%s unreadable — starting a fresh scorecard", p)
    merged = {**prev, **per_engine}
    merged_tuning = {**prev_tune, **tuning}
    p.write_text(json.dumps({"engines": merged, "tuning": merged_tuning},
                            indent=2, ensure_ascii=False), encoding="utf-8")
    fresh = sorted(per_engine)
    carried = sorted(set(merged) - set(per_engine))
    log.info("scorecard: measured %s this run%s", fresh,
             f", carried forward {carried}" if carried else "")
    return merged, merged_tuning


def _audio_on_disk(bo: Path, lang: str, merged: dict,
                   subset: list[dict]) -> dict:
    """{segment_id: {engine: relpath}} for every engine whose audio is still on
    disk — including engines measured by EARLIER runs, so the listening page can
    A/B them. Derived from the filesystem rather than this run's bookkeeping,
    which only knows about the engine it just tested."""
    out: dict[str, dict[str, str]] = {u["id"]: {} for u in subset}
    for engine in merged:
        for u in subset:
            w = bo / "seg" / engine / lang / f"{u['id']}_t0.wav"
            if w.exists():
                out[u["id"]][engine] = str(w.relative_to(bo))
    return out


def _per_segment_section(merged: dict, engines: list) -> list[str]:
    """Per-segment metrics. The means alone hide which segments drag an engine
    down, and hide a partial sample: qwen scored on 4 of 6 segments and the only
    way to see that is a table with gaps in it."""
    have = [e for e in engines if merged.get(e, {}).get("per_seg")]
    if not have:
        return []
    ids = sorted({sid for e in have for sid in merged[e]["per_seg"]})
    out = ["", "## per-segment metrics", "",
           "Means hide which segments are weak, and hide a PARTIAL sample — a "
           "blank cell means that engine produced no usable take for that "
           "segment, so its column average is over fewer segments than the "
           "others. Compare columns only where both have a value.", ""]
    for metric, label in [("sim", "sim→real"), ("mos", "mos"),
                          ("f0", "f0st"), ("wer", "wer"), ("pace", "pace")]:
        out += ["", f"**{label}**", "",
                "| segment | " + " | ".join(have) + " |",
                "|---" * (len(have) + 1) + "|"]
        for sid in ids:
            cells = [str(merged[e]["per_seg"].get(sid, {}).get(metric, "—"))
                     for e in have]
            out.append(f"| {sid} | " + " | ".join(cells) + " |")
    return out


def _tuning_section(tuning: dict, engines: list) -> list[str]:
    """What each engine was tuned to, and what it beat. The adoption decision
    has to be reproducible from the report alone — a scorecard that hides which
    settings produced it is exactly the untuned-vs-tuned trap this pass fixes."""
    if not tuning:
        return []
    out = ["", "## tune-lite — each engine at its own best point", "",
           "Every engine sweeps its own grid (bakeoff.grids) BEFORE the "
           "comparison and enters at its winner, scored on the gate's two axes "
           "(0.5*sim→real + 0.5*normalized mos). A single-point grid means "
           "'already tuned, nothing to sweep' and costs nothing. Widen a grid "
           "in config to re-open an engine's parameters.", "",
           "| engine | points | ran at | sim→real | mos |",
           "|---|---|---|---|---|"]
    for e in engines:
        tn = tuning.get(e)
        if not tn:
            out.append(f"| {e} | - | (not reached) | - | - |")
            continue
        win = next((r for r in tn["trials"] if r["point"] == tn["winner"]), None)
        note = f" — {tn['skipped']}" if tn.get("skipped") else ""
        out.append(f"| {e} | {tn['n_points']}{note} | "
                   f"{_fmt_point(tn['winner'])} | "
                   f"{win['sim'] if win else '-'} | {win['mos'] if win else '-'} |")
    losers = [(e, r) for e in engines for r in tuning.get(e, {}).get("trials", [])
              if r["point"] != tuning[e]["winner"]]
    if losers:
        out += ["", "<details><summary>grid points that lost</summary>", "",
                "| engine | point | sim→real | mos | score |", "|---|---|---|---|---|"]
        out += [f"| {e} | {_fmt_point(r['point'])} | {r['sim']} | {r['mos']} "
                f"| {r['score']} |" for e, r in losers]
        out += ["", "</details>"]
    return out


def _write_reports(bo, video, lang, subset, per_engine, seg_audio, wd,
                   ua_slice_fn, tuning=None) -> None:
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
              "`tts.engine_by_lang: {" + lang + ": <winner>}` — AND the winning "
              "engine's tuned parameters from the table below, or production "
              "will run it at defaults the bake-off did not measure."]
    lines += _tuning_section(tuning or {}, engines)
    lines += _per_segment_section(per_engine, engines)
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
