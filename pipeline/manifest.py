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
    key_data = {"t": text, "l": lang, "ref": tts_cfg["reference_wav"],
                "cfg": tts_cfg["cfg_weight"], "ex": tts_cfg["exaggeration"],
                "engine": engine}
    if engine == "cosyvoice":  # output depends on mode (+ transcript / instruct)
        key_data["mode"] = tts_cfg.get("cosyvoice_mode", "cross_lingual")
        key_data["rt"] = tts_cfg.get("reference_text", "")
        if tts_cfg.get("instruct_text"):
            key_data["ins"] = tts_cfg["instruct_text"]
    if engine == "indextts":  # emotion prompt / alpha / duration change output
        if tts_cfg.get("emotion_wav"):
            key_data["emo"] = tts_cfg["emotion_wav"]
            key_data["ea"] = tts_cfg.get("emo_alpha", 1.0)
        if tts_cfg.get("instruct_text"):
            key_data["ins"] = tts_cfg["instruct_text"]
        if tts_cfg.get("indextts_duration_ratio"):
            key_data["dr"] = tts_cfg["indextts_duration_ratio"]
    if engine == "qwen":  # clone mode (+ ref transcript when used) change output
        x_only = bool(tts_cfg.get("qwen_x_vector_only", True)) \
            or not tts_cfg.get("reference_text")
        key_data["xv"] = x_only
        if not x_only:
            key_data["rt"] = tts_cfg.get("reference_text", "")
    if engine == "voxcpm":  # style prefix + sampler params change output
        if tts_cfg.get("instruct_text"):
            key_data["ins"] = tts_cfg["instruct_text"]
        if tts_cfg.get("voxcpm_cfg_value") is not None:
            key_data["vcfg"] = tts_cfg["voxcpm_cfg_value"]
        if tts_cfg.get("voxcpm_timesteps") is not None:
            key_data["vts"] = tts_cfg["voxcpm_timesteps"]
    if engine == "chatterbox":  # only chatterbox applies normalize_for_tts
        nv = NORM_VERSIONS.get(lang, 1)
        if nv > 1:  # v1 adds no key so pre-existing caches stay valid
            key_data["nv"] = nv
    # number localization is applied to EVERY engine, so its version salts all
    # engines' hashes (v1/absent adds no key -> pre-existing caches stay valid)
    numv = NUM_VERSIONS.get(lang, 1)
    if numv > 1:
        key_data["numv"] = numv
    key = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(key.encode()).hexdigest()[:8]
