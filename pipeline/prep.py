"""prep: per-video preamble — extract reference-clip candidates from the
video's OWN clean vocals (the echo-free ref source validated on sketch60).

Requires s1 (separated stem) + s2 (utterance timings). Spans in the target
length band are cut from the 44.1 kHz vocals stem, passed through a QUALITY
GATE that rejects genuinely broken candidates (clipped, mostly-silent, noisy),
and the survivors are kept in ref/ with work/<video>/refs.json mapping each ref
to its transcript (CosyVoice zero-shot cloning needs the transcript; tune reads
it too).

The gate is a FILTER, not a ranker. A focused tune R1 (2026-07-14) showed the
old acoustic "quality score" did NOT predict clone similarity — it mildly
inverted it — so nothing here claims to pick the best clone source. That call
belongs to `tune` R1, which measures real ECAPA speaker similarity on
synthesized output and scores every ref in the pool. prep's only job is to
keep obviously-bad spans out and hand tune a wide, clean pool; survivors are
ordered by length (a neutral tie-break), and tune overrides that order anyway.

After prep: `dubadabidu tune <video> --langs en` picks the winning ref (R1),
then skim text_uk / terms_*.json and run the pipeline.
"""
from __future__ import annotations
import json
import logging
import subprocess
from pathlib import Path
from . import manifest as M

log = logging.getLogger("dubadabidu.prep")

PREP_DEFAULTS = {
    "min_s": 12.0,
    "max_s": 20.0,
    "n_refs": 3,            # keep the pool WIDE — tune R1 makes the real call
    "max_candidates": 12,   # cap gated spans (each = a cut + MOS/f0 pass)
    # quality GATE (reject broken spans only — deliberately lenient so a good
    # ref is never filtered out; tune R1 does the actual selecting):
    "clip_peak": 0.99,      # peak at/above this = clipped, rejected
    "min_voiced": 0.25,     # <25% voiced = mostly silence, rejected
    "min_mos": 3.0,         # DistillMOS below this = noisy/artifacty, rejected
}


def _spread(items: list, n: int) -> list:
    """Up to `n` items evenly spaced across `items`, keeping its order.

    Used wherever a length-sorted list gets capped. Truncating a sorted list is
    not a neutral cap — it silently picks one end. prep sorts spans longest-first
    and used to do exactly that in two places (the candidate cap and the kept
    refs), so `tune` R1 only ever scored long references and could not measure
    the 10-15 s window upstream calls optimal. prep ranks nothing; it hands over
    a pool that SPANS the band and lets R1 decide."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return items
    if n == 1:
        return [items[0]]
    idx = dict.fromkeys(round(i * (len(items) - 1) / (n - 1)) for i in range(n))
    return [items[i] for i in idx]


def _voiced_density(path: Path, thr: float = 0.05) -> float:
    """Fraction of 30 ms frames above thr*peak — how much of the span is actual
    speech vs silence/pauses. A near-empty span is a bad ref; used as a floor,
    not a score."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(1)
    peak = float(np.abs(x).max()) or 1.0
    fr = int(sr * 0.03)
    if len(x) < fr:
        return 0.0
    rms = np.array([np.sqrt(np.mean(x[i:i + fr] ** 2))
                    for i in range(0, len(x) - fr, fr)])
    return float((rms > thr * peak).mean())


def _peak(path: Path) -> float:
    import numpy as np
    import soundfile as sf
    x, _ = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(1)
    return float(np.abs(x).max())


def _quality(path: Path, clip_peak: float) -> dict:
    """Gate metrics + diagnostics for a candidate span (NOT a selection score).
    Reuses the same MOS/f0 metrics tune/evaluate use."""
    from qc import metrics as X
    pk = _peak(path)
    return {"mos": round(X.mos(str(path)), 2),
            "voiced": round(_voiced_density(path), 3),
            "f0st": round(X.f0_semitone_std(str(path)), 2),
            "peak": round(pk, 3),
            "clipped": pk >= clip_peak}


def _gate_reason(q: dict, p: dict) -> str:
    """'' if the span passes the gate, else why it was rejected."""
    if q["clipped"]:
        return f"clipped (peak {q['peak']})"
    if q["voiced"] < p["min_voiced"]:
        return f"too silent (voiced {q['voiced']} < {p['min_voiced']})"
    if q["mos"] < p["min_mos"]:
        return f"noisy (mos {q['mos']} < {p['min_mos']})"
    return ""


def run(cfg: dict, video: str) -> None:
    p = {**PREP_DEFAULTS, **cfg.get("prep", {})}
    min_s, max_s, n_refs = p["min_s"], p["max_s"], p["n_refs"]
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    stem = Path(video).stem

    # refs must come from the FULL-RATE stem, not the 16k ASR downmix — try
    # whichever separation backend s1 ran (roformer output name embeds the
    # model, so glob), then demucs, then the unseparated original.
    cand_srcs = sorted((wd / "roformer").glob("*(Vocals)*.wav")) + \
        [wd / "demucs" / cfg["separation"]["demucs_model"] / "audio_full" / "vocals.wav"]
    src = next((c for c in cand_srcs if c.exists()), None)
    if src is None:
        src = wd / "audio_full.wav"
        log.warning("no separated vocals stem — cutting refs from %s "
                    "(fine only if the source has no music/noise)", src.name)

    # candidate spans, longest first (a neutral order — the gate keeps/drops,
    # it does not rank clone quality; tune R1 does)
    spans = [u for u in man["utterances"] if min_s <= u["end"] - u["start"] <= max_s]
    spans.sort(key=lambda u: u["end"] - u["start"], reverse=True)
    if not spans:  # fall back to the longest available utterances
        spans = sorted(man["utterances"],
                       key=lambda u: u["end"] - u["start"], reverse=True)
        log.warning("no %d-%ds utterances; using the longest available",
                    int(min_s), int(max_s))
    # SPREAD, don't truncate. `spans` is sorted longest-first, so
    # `spans[:max_candidates]` gated only the longest N — on 2026-08-03 that
    # left every candidate >= 14.5 s in a 10-20 s band, so widening min_s to
    # reach the 10-15 s optimum changed nothing. The cap is a COST limit (each
    # candidate is a cut plus a MOS/f0 pass), not a length preference.
    spans = _spread(spans, p["max_candidates"])

    cand_dir = wd / "refcand"
    cand_dir.mkdir(parents=True, exist_ok=True)
    passed, rejected = [], []   # (span, cut, quality)
    for u in spans:
        dur = min(u["end"] - u["start"], max_s)
        cut = cand_dir / f"cand_{u['id']}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ss", str(u["start"]), "-t", str(dur),
                        str(cut)], check=True)
        q = _quality(cut, p["clip_peak"])
        reason = _gate_reason(q, p)
        if reason:
            rejected.append((u, cut, q))
            log.info("candidate %s (%.1fs): REJECT — %s", u["id"], dur, reason)
        else:
            passed.append((u, cut, q, dur))
            log.info("candidate %s (%.1fs): pass  mos=%.2f voiced=%.2f f0st=%.2f "
                     "peak=%.2f", u["id"], dur, q["mos"], q["voiced"],
                     q["f0st"], q["peak"])

    # Survivors are STRATIFIED across the length band, not topped off from the
    # long end. `passed` is sorted longest-first, so `passed[:n_refs]` handed
    # tune R1 the n longest spans and nothing else — on 2026-08-03 that meant
    # 17.6/19.5/19.8 s out of a 12-20 s band, and R1 had therefore never scored
    # a reference in the 10-15 s range upstream documents as optimal ("quality
    # scales roughly linearly from 3 to 15 s, then plateaus and eventually
    # degrades"). The old comment called length a "neutral tie-break"; taking
    # the top N of a sorted list is not a tie-break, it is a bias, and it
    # silently decided an axis this module says belongs to tune.
    #
    # Evenly spaced indices give shortest / middle / longest, so R1 measures
    # length instead of inheriting it. prep still ranks nothing — it just hands
    # over a pool that spans the band.
    keep = _spread(passed, n_refs)
    if not keep:
        log.warning("no span passed the quality gate — keeping the %d longest "
                    "anyway; inspect ref/ before trusting the clone", n_refs)
        keep = [(u, cut, q, min(u["end"] - u["start"], max_s))
                for u, cut, q in rejected[:n_refs]]

    kept_cuts = {id(x[1]) for x in keep}
    refs = {}
    for i, (u, cut, q, dur) in enumerate(keep, 1):
        out = Path("ref") / f"{stem}_ref_{i:02d}.wav"
        cut.replace(out)
        refs[out.name] = {"text_uk": u["text_uk"], "start": u["start"],
                          "end": round(u["start"] + dur, 2), "quality": q}
        print(f"[prep] {out}  ({dur:.1f}s)  mos={q['mos']} voiced={q['voiced']}  "
              f"{u['text_uk'][:48]}")
    for _, cut, _ in rejected:   # drop losing cuts (kept ones were moved out)
        if id(cut) not in kept_cuts:
            cut.unlink(missing_ok=True)

    rp = wd / "refs.json"
    rp.write_text(json.dumps(refs, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"[prep] {len(spans)} candidates, {len(passed)} passed the quality "
          f"gate, {len(refs)} kept — `tune` R1 selects the clone winner")
    print(f"[prep] transcripts -> {rp}\n[prep] next: dubadabidu tune {video} "
          f"--langs en   (tune.refs_glob may need ref/{stem}_ref_*.wav)")
