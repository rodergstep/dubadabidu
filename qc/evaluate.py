"""Eval layer: per-segment perceptual metrics -> manifest + printed scorecard.

Writes per segment (tr.<lang>):
  qc_sim2     raw ECAPA cosine vs reference
  qc_sim_cal  calibrated 0..1 inside this ref's [floor..ceiling] band
  qc_mos      Distill-MOS naturalness proxy 1..5
  qc_f0st     f0 std in semitones (monotony indicator, informational)
  qc_score    composite objective (tune loop optimizes this)

Calibration band, computed once per run and stored in manifest["qc_calibration"]:
  ceiling = mean ECAPA sim(reference, real UA vocal slices)  — "same speaker"
  floor   = ECAPA sim(reference, an edge-TTS voice)          — "different speaker"
"""
from __future__ import annotations
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import manifest as M  # noqa: E402
from qc import metrics as X  # noqa: E402

log = logging.getLogger("dubadabidu.qc.evaluate")

FLOOR_SENTENCES = {
    "en": "This is a calibration sentence for speaker similarity.",
    "fr": "Ceci est une phrase de calibration pour la similarité vocale.",
    "de": "Dies ist ein Kalibrierungssatz für die Sprecherähnlichkeit.",
    "es": "Esta es una frase de calibración para la similitud del hablante.",
    "ru": "Это калибровочное предложение для оценки похожести голоса.",
    "pl": "To jest zdanie kalibracyjne do oceny podobieństwa mówcy.",
}


def _ua_slices(wd: Path, utterances: list[dict], max_slices: int = 6):
    """Real-voice 16 kHz slices from vocals.wav (already 16k mono from s1)."""
    import soundfile as sf
    import torch
    path = wd / "vocals.wav"
    info = sf.info(str(path))
    step = max(1, len(utterances) // max_slices)
    for u in utterances[::step][:max_slices]:
        start = int(u["start"] * info.samplerate)
        stop = min(int(u["end"] * info.samplerate), info.frames)
        if stop - start < info.samplerate:  # skip <1s slices
            continue
        data, sr = sf.read(str(path), start=start, stop=stop, dtype="float32")
        wav = torch.tensor(data).unsqueeze(0)
        if sr != X.SAMPLE_RATE:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, X.SAMPLE_RATE)
        yield wav


def _floor_wav(cfg: dict, lang: str) -> Path | None:
    """Cached different-voice sample (edge-tts). None if synth fails (offline)."""
    from pipeline.tts_engine import synthesize
    out = Path("work/.models") / f"floor_{lang}.wav"
    if out.exists():
        return out
    t = dict(cfg["tts"], engine="edge")
    try:
        synthesize(FLOOR_SENTENCES.get(lang, FLOOR_SENTENCES["en"]), lang, out, t,
                   retries=0)
        return out
    except Exception as e:
        log.warning("floor synth failed (%s); using sim_floor_default", e)
        return None


def calibration(cfg: dict, wd: Path, man: dict, lang: str) -> dict:
    ref_emb = X.ecapa_embed(cfg["tts"]["reference_wav"])
    sims = [X.cosine(ref_emb, X.ecapa_embed(w))
            for w in _ua_slices(wd, man["utterances"])]
    ceiling = sum(sims) / len(sims) if sims else 1.0
    fw = _floor_wav(cfg, lang)
    ecfg = cfg["qc"].get("eval", {})
    floor = (X.cosine(ref_emb, X.ecapa_embed(fw)) if fw
             else ecfg.get("sim_floor_default", 0.35))
    return {"lang": lang, "floor": round(floor, 3), "ceiling": round(ceiling, 3),
            "ref": cfg["tts"]["reference_wav"]}


def run(cfg: dict, video: str, langs: list[str],
        only: list[str] | None = None) -> None:
    """only: utterance ids to (re)score; None = all. Subset runs reuse the
    stored calibration band (same ref => same band; recomputing it costs an
    ECAPA embed of the ref + 6 vocal slices per round for nothing)."""
    man = M.load(cfg, video)
    # similarity must be judged against the ref the video was actually synthesized
    # with — honor the per-video override written by `preamble`
    if man.get("tts_overrides"):
        cfg = {**cfg, "tts": {**cfg["tts"], **man["tts_overrides"]}}
    wd = M.video_workdir(cfg, video)
    ecfg = cfg["qc"].get("eval", {})
    weights = ecfg.get("weights", {"sim": 0.5, "mos": 0.35, "tempo": 0.15})
    max_tempo = cfg["fit"]["max_tempo"]
    ref_emb = X.ecapa_embed(cfg["tts"]["reference_wav"])

    for lang in langs:
        cal = man.get("qc_calibration", {}).get(lang)
        if only is None or not cal or cal.get("ref") != cfg["tts"]["reference_wav"]:
            cal = calibration(cfg, wd, man, lang)
            man.setdefault("qc_calibration", {})[lang] = cal
        log.info("%s calibration: floor=%.3f ceiling=%.3f",
                 lang, cal["floor"], cal["ceiling"])
        rows = []
        for u in man["utterances"]:
            if only is not None and u["id"] not in only:
                continue
            tr = u["tr"][lang]
            wav = M.scored_path(wd, tr)
            raw = X.cosine(ref_emb, X.ecapa_embed(wav))
            sim_cal = X.calibrate_sim(raw, cal["floor"], cal["ceiling"])
            m = X.mos(wav)
            pen = X.tempo_penalty(tr.get("tempo", 1.0), max_tempo)
            f0st = X.f0_semitone_std(wav)
            tr["qc_sim2"] = round(raw, 3)
            tr["qc_sim_cal"] = round(sim_cal, 3)
            tr["qc_mos"] = round(m, 2)
            tr["qc_mos_min"] = round(X.mos_min_window(wav), 2)
            tr["qc_f0st"] = round(f0st, 2)
            tr["qc_score"] = X.composite_score(sim_cal, m, pen, weights, f0st)
            # stamp WHICH audio these scores describe, so a later s5/s6 re-run
            # (which rewrites the placed wav) is detectable instead of silent
            M.stamp_qc(wd, tr, "score")
            rows.append((u["id"], tr))
        M.save(cfg, video, man)

        scope = f"  [{len(rows)}/{len(man['utterances'])} segments]" \
            if only is not None else ""
        print(f"\n[evaluate] {lang}  (floor={cal['floor']}  "
              f"ceiling={cal['ceiling']}){scope}")
        hdr = f"{'id':6} {'score':>6} {'sim2':>6} {'simcal':>7} {'mos':>5} " \
              f"{'mosmin':>6} {'f0st':>5} {'tempo':>6} {'fit':>9}"
        print(hdr); print("-" * len(hdr))
        for uid, tr in sorted(rows, key=lambda r: r[1]["qc_score"]):
            print(f"{uid:6} {tr['qc_score']:>6.3f} {tr['qc_sim2']:>6.3f} "
                  f"{tr['qc_sim_cal']:>7.3f} {tr['qc_mos']:>5.2f} "
                  f"{tr['qc_mos_min']:>6.2f} "
                  f"{tr['qc_f0st']:>5.2f} {tr.get('tempo', 1.0):>6.3f} "
                  f"{tr.get('fit', '-'):>9}")
        mean = lambda k: sum(r[1][k] for r in rows) / len(rows)  # noqa: E731
        print(f"means: score={mean('qc_score'):.3f} sim_cal={mean('qc_sim_cal'):.3f} "
              f"mos={mean('qc_mos'):.2f}")
