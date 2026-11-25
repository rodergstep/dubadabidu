# SPDX-License-Identifier: AGPL-3.0-or-later
"""`dubadabidu preamble`: the per-video prep pass (s1 + s2 + reference pick)."""
from __future__ import annotations

import logging
from pathlib import Path

from dubadabidu.core import manifest as M
from dubadabidu.lab import prep as prep_mod
from dubadabidu.lab import tune as tune_mod
from dubadabidu.stages import s1_extract
from dubadabidu.stages import s2_transcribe
from dubadabidu.stages import s3_translate
from dubadabidu.core.config.schema import Config, derive


def preamble(cfg: Config, video: str, langs: list[str]) -> None:
    """Per-video preamble (IMPROVEMENT_PLAN Phase D), resumable:
    pass 1: s1 + s2 + prep, then pause for the text_src hand-review;
    pass 2 (re-run):  s3 on the tune language, tune-lite R1 over this video's
    own refs, winning ref stored in manifest tts_overrides (s4/s5/evaluate
    honor it — no config paste per video)."""
    fresh = not M.manifest_path(cfg, video).exists()
    s1_extract.run(cfg, video)
    if fresh:
        s2_transcribe.run(cfg, video)
    wd = M.video_workdir(cfg, video)
    if not (wd / "refs.json").exists():
        prep_mod.run(cfg, video)
    if fresh:
        print(f"\n[preamble] paused — review text_src in "
              f"{M.manifest_path(cfg, video)} (one fix here propagates to all "
              f"languages), then re-run:\n  dubadabidu preamble {video}")
        return
    lang = langs[0]
    s3_translate.run(cfg, video, [lang])
    # R1 must be able to see the CURATED references, not just this lesson's own
    # cuts. Measured 2026-08-03: scoping the pool to `ref/{stem}_ref_*.wav` made
    # R1 pick this lesson's ref_04 (mos 4.26, the flattest f0st of all 11) while
    # never comparing it against sketch60_ref_03 (mos 4.59) — and the listener
    # rated audio from ref_04 12-of-18 unusable against ~3.5/5 for sketch60.
    # Source quality varies per lesson; the reference is a curated asset, so the
    # pool is this video's cuts PLUS whatever tune.refs_glob already names.
    tcfg = derive(cfg, tune=derive(cfg.tune, refs_glob="ref/*.wav",
                                   subset_size=5, rounds=["R1"]))
    win = tune_mod.run(tcfg, video, [lang])
    man = M.load(cfg, video)
    ref = _pick_reference(cfg, win)
    if ref is None:
        man.pop("tts_overrides", None)
    else:
        over = {"reference_wav": ref}
        rt = M.load_refs(wd).get(Path(ref).name, {}).get("text_src", "")
        if rt:  # ref transcript (prep fills refs.json)
            over["reference_text"] = rt
        man["tts_overrides"] = over
    M.save(cfg, video, man)
    used = ref or cfg.tts.reference_wav
    print(f"\n[preamble] done — reference for this video: {used}"
          + ("  (per-video override)" if ref else "  (curated default)")
          + f"\nNext: skim {wd}/terms_{lang}.json, then:"
          f"\n  dubadabidu run {video} --from s3_translate")


def _pick_reference(cfg: Config, win: dict) -> str | None:
    """R1's winner, or None to keep the curated `tts.reference_wav`.

    FINDINGS 2.2 refuted the per-video reference and recorded the verdict as
    "ENCODED — the reference is now a config default, not a per-video
    override". Only the ROOT CAUSE was actually fixed (refs_glob was widened so
    R1 can see the curated pool at all). `preamble` still wrote
    man["tts_overrides"] unconditionally, and s4, s5, evaluate and bakeoff all
    merge it — so the refuted mechanism stayed live and regenerated itself on
    every new video, including on a pod (REMOTE_TASK["preamble"]).

    Two gates now stand between R1 and the manifest:

      1. tune.per_video_ref_override (default FALSE) — the code finally says
         what FINDINGS says. The machinery is intact behind the flag rather
         than deleted, because the ROOT CAUSE fix means a future sweep is at
         least measuring the right pool.
      2. tune.override_margin — even when enabled, the winner must beat the
         CURATED DEFAULT by a real margin, and the default must have been IN
         the sweep. A sweep that cannot see the incumbent cannot rank against
         it; that is exactly how this lesson's ref_04 (rated 12-of-18 unusable)
         won in the first place.

    Both gates are about trust, not arithmetic: R1's objective is labelled
    UNVALIDATED in tune.DEFAULTS because the ear has contradicted it every time
    it has been checked.
    """
    tn = cfg.tune
    ref, default = win["reference_wav"], cfg.tts.reference_wav
    if not tn.per_video_ref_override:
        logging.info(
            "per-video reference override is OFF (tune.per_video_ref_override) "
            "— keeping the curated %s. R1's pick was %s; see its report for the "
            "ranking.", default, ref)
        return None
    scores = win.get("r1_scores") or {}
    if default not in scores:
        logging.warning(
            "REFUSING the per-video reference: the curated default %s was not "
            "in this sweep, so there is nothing to beat. Widen tune.refs_glob "
            "to include it.", default)
        return None
    margin = float(tn.override_margin)
    gain = scores[ref] - scores[default]
    if gain < margin:
        logging.info("keeping the curated %s — R1 gave %s only %+.3f "
                     "(needs >= %.3f)", default, ref, gain, margin)
        return None
    logging.warning("per-video reference override: %s beats the curated %s by "
                    "%+.3f on an UNVALIDATED objective — confirm by ear before "
                    "trusting the batch", ref, default, gain)
    return ref
