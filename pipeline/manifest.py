"""Central per-video manifest (pattern from Softcatala/open-dubbing utterance_metadata).

work/<video>/manifest.json:
{
  "video": "input/lesson01.mp4",
  "duration": 3601.2,
  "utterances": [
    {
      "id": "u0001",
      "start": 7.61, "end": 12.03,
      "text_uk": "Сьогодні ми змішуємо ...",
      "tr": {
        "en": {
          "text": "Today we are mixing ...",
          "variants": ["Today we mix ...", "We mix ..."],   # progressively shorter
          "synth": "seg/en/u0001_9f3a.wav",                  # content-hash cached
          "synth_dur": 4.71,
          "tempo": 1.07,                                     # applied stretch
          "fit": "ok"                                        # ok | stretched | shortened | overflow
        }, ...
      }
    }, ...
  ]
}
Hand-edit text_uk or tr.<lang>.text, delete the segment's "synth" key (or just re-run
s4 — hash mismatch re-synthesizes automatically), and re-run from s4.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from .text_norm import NORM_VERSIONS, NUM_VERSIONS


def video_workdir(cfg: dict, video: str | Path) -> Path:
    d = Path(cfg["work_dir"]) / Path(video).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(cfg: dict, video: str | Path) -> Path:
    return video_workdir(cfg, video) / "manifest.json"


def load(cfg: dict, video: str | Path) -> dict:
    p = manifest_path(cfg, video)
    if not p.exists():
        raise SystemExit(f"No manifest at {p} — run s2_transcribe first.")
    return json.loads(p.read_text(encoding="utf-8"))


def save(cfg: dict, video: str | Path, man: dict) -> None:
    p = manifest_path(cfg, video)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


# tr[lang] keys written by the synth/fit/mix/qc stages (s4..s6 + qc). clear_synth
# drops these; it deliberately KEEPS text/variants/adequacy (translation work) and
# human_verdict/human_rating (human labels — recoverable, and preserved so a forced
# re-synth doesn't silently discard review effort).
_SYNTH_KEYS = ("takes", "synth_engine", "synth", "synth_dur",
               "fitted", "fitted_text", "tempo", "fit", "placed",
               "placed_start", "placed_end", "drift", "drift_exceeded",
               "norm_gain_db", "reroll_wer")


def clear_synth(cfg: dict, video: str | Path, langs: list[str]) -> int:
    """Discard synthesized takes + fit/mix/qc state for `langs` so a re-run of
    s4..s7 re-synthesizes from scratch. Needed after changing a take-SELECTION
    param — best_of, rank_takes, retake_mos_below, min_f0st, f0_reroll_max,
    best_of_early_accept, early_accept_* — because synth_hash intentionally omits
    those: they don't change a single take's inputs, only which of several takes
    WINS. The content-hash cache would otherwise serve the previously-picked take
    and silently ignore the new setting, contaminating any A/B. Returns the number
    of segment wavs removed."""
    import shutil
    wd = video_workdir(cfg, video)
    man = load(cfg, video)
    removed = 0
    for lang in langs:
        seg_dir = wd / "seg" / lang
        if seg_dir.exists():
            removed += sum(1 for _ in seg_dir.rglob("*.wav"))
            shutil.rmtree(seg_dir)
        for u in man["utterances"]:
            tr = u["tr"].get(lang)
            if not tr:
                continue
            for k in _SYNTH_KEYS:
                tr.pop(k, None)
            for k in [k for k in tr if k.startswith("qc_")]:
                del tr[k]
        for st in ("s4", "s5", "s6", "s7"):
            man["stages"].pop(f"{st}_{lang}", None)
    save(cfg, video, man)
    return removed


# ---------- QC staleness ----------
# evaluate/backcheck grade the PLACED segment (post-trim/normalize), but s5/s6
# rewrite that file whenever fit or mix settings change — and nothing re-scores
# it. The scores then describe audio that no longer exists, and every consumer
# (report, batch_report, review pages, autopilot._assess, and the ratings rows
# the qc-weight re-fit trains on) silently believes them. Each scoring pass
# stamps WHICH audio it graded, so staleness is detectable instead of invisible.
#
# Stamps are `qc_`-prefixed, so clear_synth and autopilot._reroll already drop
# them with the rest of the qc keys. A segment scored before stamping existed
# reads as stale — correct: those scores are unverifiable.
_QC_STAMPS = {
    "score": ("qc_of", ("qc_score", "qc_sim2", "qc_sim_cal", "qc_mos",
                        "qc_mos_min", "qc_f0st")),
    "wer":   ("qc_wer_of", ("qc_wer",)),
}


def scored_path(wd: Path, tr: dict) -> Path | None:
    """The audio QC grades: the placed segment once s6 has run, else the fitted
    one. Single source of truth — evaluate, backcheck and the staleness check
    must all resolve the same file or the stamp means nothing."""
    rel = tr.get("placed") or tr.get("fitted")
    return (wd / rel) if rel else None


def audio_sig(path: Path) -> str:
    """Content hash of a scored wav. Content, not mtime: work/ is rsync'd to and
    from the GPU pod, and timestamps do not survive that reliably."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def stamp_qc(wd: Path, tr: dict, kind: str) -> None:
    """Record which audio the `kind` ('score' | 'wer') results describe."""
    p = scored_path(wd, tr)
    if p and p.exists():
        tr[_QC_STAMPS[kind][0]] = audio_sig(p)


def stale_qc(wd: Path, man: dict, lang: str) -> dict[str, list[str]]:
    """{'score': [ids], 'wer': [ids]} whose stored results were computed on
    different audio than the segment points at now. Segments with nothing
    scored yet are MISSING, not stale, and are omitted. Each file is hashed
    once per call."""
    out: dict[str, list[str]] = {"score": [], "wer": []}
    for u in man["utterances"]:
        tr = u["tr"].get(lang)
        if not tr:
            continue
        p = scored_path(wd, tr)
        sig = audio_sig(p) if p and p.exists() else None
        for kind, (stamp, score_keys) in _QC_STAMPS.items():
            if not any(k in tr for k in score_keys):
                continue                    # nothing scored — missing, not stale
            # `sig is None` (the graded file is gone) must not compare equal to
            # an absent stamp, or a segment whose audio vanished would read as
            # current. Scores with no audio behind them are always stale.
            if sig is None or tr.get(stamp) != sig:
                out[kind].append(u["id"])
    return out


def edge_langs(man: dict, langs: list[str]) -> list[str]:
    """Languages whose synthesis used the edge fallback (generic MS voices,
    no cloning). Edge output validates plumbing ONLY — it must never be
    mistaken for judgeable dubbing, so mux renames it, the review page
    banners it, and verdict ingestion refuses it. Legacy manifests without
    synth_engine tags are treated as real (they predate the edge fixtures)."""
    return [lang for lang in langs
            if any(u["tr"].get(lang, {}).get("synth_engine") == "edge"
                   for u in man["utterances"])]


def resolve_engine(tts_cfg: dict, lang: str) -> str:
    """Effective engine for a language: per-lang override, else the default."""
    return tts_cfg.get("engine_by_lang") and \
        tts_cfg["engine_by_lang"].get(lang) or tts_cfg["engine"]


def synth_hash(text: str, lang: str, tts_cfg: dict) -> str:
    """Content hash → cache key. Any change in text/ref/params triggers
    re-synthesis. The engine is resolved per-language; extra keys are added
    ONLY for engines that use them, so pre-existing caches stay valid."""
    engine = resolve_engine(tts_cfg, lang)
    # cfg_weight / exaggeration were CHATTERBOX knobs and left with it, but they
    # stay in the key at their old defaults so every previously cached take
    # keeps its hash. Removing them outright would silently invalidate every
    # seg/ cache for no benefit.
    key_data = {"t": text, "l": lang, "ref": tts_cfg["reference_wav"],
                "cfg": tts_cfg.get("cfg_weight", 0.0),
                "ex": tts_cfg.get("exaggeration", 0.55),
                "engine": engine}
    if engine == "qwen":  # clone mode (+ ref transcript when used) change output
        x_only = bool(tts_cfg.get("qwen_x_vector_only", True)) \
            or not tts_cfg.get("reference_text")
        key_data["xv"] = x_only
        if not x_only:
            key_data["rt"] = tts_cfg.get("reference_text", "")
        # qwen_fast swaps the inference implementation (faster-qwen3-tts:
        # CUDA Graphs + StaticCache). It SHOULD be numerically identical to the
        # stock decode loop, but "should" is not "is" — salt the key so an A/B
        # re-synthesizes instead of comparing new settings against cached audio
        # produced by the other implementation. Absent/false adds no key, so
        # pre-existing caches stay valid.
        if tts_cfg.get("qwen_fast"):
            key_data["qf"] = True
        # a trimmed reference is a DIFFERENT reference — the adapter swaps the
        # path internally, so "ref" above would not see it and the cache would
        # serve 18 s-reference audio for a 12 s-reference config.
        if tts_cfg.get("reference_max_s"):
            key_data["rmax"] = tts_cfg["reference_max_s"]
        # sampling params change the output, so they must change the key
        if tts_cfg.get("qwen_gen_kwargs"):
            key_data["gk"] = sorted(tts_cfg["qwen_gen_kwargs"].items())
        # WHICH MODEL produced the audio. Missing until 2026-08-02: switching
        # qwen_model_dir (1.7B <-> 0.6B, or a local checkpoint) left the cache
        # serving takes from the OTHER model, so an A/B would have compared a
        # model against itself. The bake-off never hit it because it writes to
        # per-variant seg/ paths rather than hash-named ones; s4 would have.
        # Non-default only, so existing caches stay valid.
        mdl = tts_cfg.get("qwen_model_dir")
        if mdl and mdl != "Qwen/Qwen3-TTS-12Hz-1.7B-Base":
            key_data["mdl"] = mdl
    # NORM_VERSIONS salted the key for chatterbox's acute RU stress marks. That
    # normalisation was chatterbox-only and left with it (2026-08-02); no
    # surviving engine applies it, so nothing salts on it.
    # number localization is applied to EVERY engine, so its version salts all
    # engines' hashes (v1/absent adds no key -> pre-existing caches stay valid)
    numv = NUM_VERSIONS.get(lang, 1)
    if numv > 1:
        key_data["numv"] = numv
    key = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(key.encode()).hexdigest()[:8]
