"""s4: per-segment TTS through pipeline.tts_engine with content-hash caching.

Auto-repair: autoregressive takes vary (measured spread ~0.74-0.80 on identical
config); best_of takes are scored on windowed MOS + f0 liveliness + ECAPA
similarity and the composite-best wins (tts.rank_takes; see synth_best_of).
Per-take metrics land in the manifest (tr.takes) — diagnostic memory for the
autopilot and features for the qc-weight re-fit. Cached segments are never
re-judged — the hash-named file is the accepted take.

Autopilot hook: a segment marked tr.reroll_wer (set by autopilot._reroll for
WER-flagged segments) is re-synthesized with a per-take back-transcription
veto, so a hallucination re-roll can't be won by another hallucination."""
from __future__ import annotations
import logging
from pathlib import Path
import soundfile as sf
from . import manifest as M
from .tts_engine import synth_best_of


def _rank_w(cfg: dict) -> dict | None:
    """qc.eval.weights, which is what ranks takes (tts_engine._take_rank).

    Threaded explicitly rather than read inside the engine: the engine takes a
    TTS-config dict, and these live under qc — hiding the lookup in there is how
    the two drifted apart in the first place."""
    return (cfg.get("qc", {}).get("eval", {}) or {}).get("weights")

log = logging.getLogger("dubadabidu.s4")

# How far BELOW s5's soft_tempo threshold to start pre-making variants. s5
# measures the primary against its retimed slot, which can be tighter than the
# source slot s4 sees, so 0.85 buys ~15% of headroom. Lower = more spare takes
# synthesized (costs GPU); higher = more chances s5 needs an engine it has not
# got. Cheap insurance either way: a missed variant strands a whole run.
VARIANT_MARGIN = 0.85


def run(cfg: dict, video: str, langs: list[str]) -> None:
    man = M.load(cfg, video)
    # per-video overrides (e.g. this video's own ref picked by `preamble`)
    t = {**cfg["tts"], **man.get("tts_overrides", {})}
    wd = M.video_workdir(cfg, video)

    for lang in langs:
        seg_dir = wd / "seg" / lang
        seg_dir.mkdir(parents=True, exist_ok=True)
        n_new = 0
        n_var = 0
        total = len(man["utterances"])
        for k, u in enumerate(man["utterances"], 1):
            tr = u["tr"].get(lang)
            if not tr or not tr.get("text"):
                raise SystemExit(f"{u['id']} missing {lang} translation — run s3.")
            tu = t
            h = M.synth_hash(tr["text"], lang, tu)
            out = seg_dir / f"{u['id']}_{h}.wav"
            fresh = not out.exists()
            if fresh:
                takes: list[dict] = []
                verify = bool(tr.get("reroll_wer"))
                synth_best_of(tr["text"], lang, out, tu, meta=takes,
                              verify_cfg=cfg if verify else None,
                              verify_text=tr["text"] if verify else None,
                              rank_weights=_rank_w(cfg),
                              # source slot: rank takes toward the speaker's
                              # own pace and prefer takes that fit it
                              target_dur=u["end"] - u["start"])
                tr["takes"] = takes
                tr["synth_engine"] = M.resolve_engine(t, lang)
                tr.pop("reroll_wer", None)
                n_new += 1
            info = sf.info(str(out))
            tr["synth"] = str(out.relative_to(wd))
            tr["synth_dur"] = round(info.frames / info.samplerate, 3)

            # Pre-synthesize the VARIANTS s5 would otherwise generate itself.
            #
            # s5_fit calls seg_wav() for candidates[1:] whenever the primary
            # needs a hard stretch, i.e. it can SYNTHESIZE. That silently
            # assumed an engine reachable wherever s5 runs, which held only
            # while the engine was edge (a network service). With qwen the
            # split is s4-on-a-GPU-pod / s5-on-the-laptop, and s5 died on
            # `No module named 'faster_qwen3_tts'` after four languages of pod
            # synthesis had already been paid for (2026-08-02). course.py's
            # whole design — phase C is local and free — depends on s5 never
            # needing a GPU.
            #
            # Gate it on the same condition s5 uses so this does not triple
            # s4's cost: only segments whose primary already overruns its slot
            # can need a shorter variant. s5 measures against the RETIMED slot,
            # which this cannot know, so allow a margin and pre-make a few
            # extra rather than miss one and strand the run.
            soft = float(cfg.get("fit", {}).get("soft_tempo", 1.06))
            slot = u["end"] - u["start"]
            ratio = (tr["synth_dur"] / slot) if slot > 0 else float("inf")
            if ratio > soft * VARIANT_MARGIN:
                # STOP AT THE FIRST VARIANT THAT CLEARS THE SAME BAR. Variants
                # are progressively shorter by construction (s3 generates them
                # that way), so once one comfortably fits, every later one is
                # shorter still and s5's ladder — which walks the candidates and
                # takes the first that places — will never reach it. Making all
                # of them meant up to translation.n_short_variants EXTRA
                # best_of units per over-long segment, i.e. up to 3x s4's GPU
                # bill on the segments that trip this gate, to synthesize audio
                # nothing reads.
                #
                # The bar is the SAME margin the gate above uses, not a bare
                # fit: s5 measures against the RETIMED slot, which can be
                # tighter than this one, and VARIANT_MARGIN is exactly the
                # headroom that buys. So this drops variants that are provably
                # surplus and keeps the insurance that stops s5 needing a GPU.
                keep_under = slot * soft * VARIANT_MARGIN if slot > 0 else 0.0
                for c in (tr.get("variants") or []):
                    vh = M.synth_hash(c, lang, tu)
                    vout = seg_dir / f"{u['id']}_{vh}.wav"
                    if not vout.exists():
                        synth_best_of(c, lang, vout, tu,
                                      target_dur=slot or None,
                                      rank_weights=_rank_w(cfg))
                        n_var += 1
                        M.save(cfg, video, man)
                    vinfo = sf.info(str(vout))
                    if keep_under and vinfo.frames / vinfo.samplerate <= keep_under:
                        break
            if fresh:  # checkpoint each new synth: long best-of units are minutes each
                M.save(cfg, video, man)
                if n_new % 25 == 0:
                    log.info("%s: %d/%d ...", lang, k, total)
        man["stages"][f"s4_{lang}"] = "done"
        M.save(cfg, video, man)
        log.info("%s: %d synthesized, %d cached, %d fit-variant(s) pre-made",
                 lang, n_new, total - n_new, n_var)
