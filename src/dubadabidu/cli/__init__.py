#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""dubadabidu — Ukrainian/Russian -> multilingual video dubbing.

  dubadabidu doctor                          # validate environment before anything
  dubadabidu preamble input/a.mp4            # per-video prep: refs + ref pick (2 passes)
  dubadabidu run input/*.mp4 [--langs en,de] [--from s4] [--engine edge]
  dubadabidu stage s3_translate input/a.mp4
  dubadabidu qc input/a.mp4
  dubadabidu report input/a.mp4              # per-language fit/QC summary table
"""
from __future__ import annotations
import logging, subprocess, sys
from pathlib import Path
import yaml

from dubadabidu.cli.health import doctor
from dubadabidu.cli.parser import build_parser
from dubadabidu.cli.remote import _remote_run_batch, _remote_stages
from dubadabidu.cli.report import report
from dubadabidu.cli.stages import ORDER, STAGES
from dubadabidu.core import manifest as M
from dubadabidu.core.logic import deep_merge

__all__ = ["ORDER", "STAGES", "doctor", "main", "report"]


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # resolve user-supplied paths against the caller's cwd, then anchor to the
    # PROJECT root (core/project.py) so relative resources (config, glossary/,
    # ref/, work/) work no matter where the CLI is invoked from
    import os
    from dubadabidu.core.project import resolve as resolve_project
    project = resolve_project(a.project, os.environ, Path.cwd())
    config_path = Path(a.config).resolve() if Path(a.config).exists() \
        else project.root / a.config
    a.rest = [str(Path(v).resolve()) if Path(v).exists() else v for v in a.rest]
    os.chdir(project.root)

    # .env contract (AUTOPILOT.md): the human's only config touchpoint for
    # credentials. KEY=VALUE lines; the environment always wins over the file.
    env_file = project.env_file
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    for ov in (a.overlay or []):  # stacked left-to-right; later overlays win
        overlay_path = Path(ov) if Path(ov).is_absolute() else project.root / ov
        cfg = deep_merge(cfg, yaml.safe_load(open(overlay_path, encoding="utf-8")))
    if a.engine:
        cfg["tts"]["engine"] = a.engine
    # every key of the MERGED config must be declared (a typo used to be
    # silently ignored and the code ran on its default) — PLAN_v0.5 P7
    from dubadabidu.core.config.schema import validate as _validate_config
    _validate_config(cfg)
    langs = ([x.strip() for x in a.langs.split(",") if x.strip()]
             if a.langs else cfg["languages"])
    # a video is never dubbed into its own language (cfg languages lists ru,
    # which is also a valid source)
    langs = [x for x in langs if x != cfg["source_language"]]

    # where ECAPA / Distill-MOS run. Default 'cpu' keeps every stored score
    # comparable with the ones already in the manifests; 'cuda'/'auto' is the
    # pod-cost lever (see qc.metrics.set_device for how to validate the switch).
    from dubadabidu.qc import metrics as _metrics
    _metrics.set_device(cfg.get("qc", {}).get("metrics_device", "cpu"))

    if a.cmd == "doctor":
        sys.exit(doctor(cfg, f"{project.root} (found by {project.found_by})"))
    if a.cmd == "batch":  # no args = all of work/; else the given videos
        from dubadabidu.review import batch_report
        batch_report.run(cfg, a.rest or None)
        return
    if a.cmd == "refit":  # M4: propose qc.eval.weights from accumulated ratings
        from dubadabidu.review import refit
        sys.exit(refit.run(cfg, langs))
    if not a.rest:
        ap.error("missing video path(s)")

    if a.cmd == "report":
        for v in a.rest:
            report(cfg, v)
        return
    if a.cmd == "stage":
        stage, videos = a.rest[0], a.rest[1:]
        if stage not in STAGES:
            sys.exit(f"unknown stage {stage}; choose from {ORDER}")
        for v in videos:
            if a.force and stage == "s3_translate":
                man = M.load(cfg, v)
                # drop translations AND the terms base — a new provider should
                # rebuild terminology too. Old synth wavs die by hash mismatch.
                for u in man["utterances"]:
                    for lang in langs:
                        u["tr"].pop(lang, None)
                for lang in langs:
                    man["stages"].pop(f"s3_{lang}", None)
                    (M.video_workdir(cfg, v) / f"terms_{lang}.json") \
                        .unlink(missing_ok=True)
                M.save(cfg, v, man)
                logging.info("--force: cleared %s translations for %s",
                             langs, v)
            elif a.force and stage == "s4_synthesize":
                # take-SELECTION params (best_of, min_f0st, ...) are NOT in
                # synth_hash, so the cache would serve the old winning take and
                # ignore the new setting; clear the takes so s4 re-synthesizes.
                n = M.clear_synth(cfg, v, langs)
                logging.info("--force: cleared %d cached takes for %s %s — "
                             "take-selection params re-apply", n, langs, v)
            STAGES[stage](cfg, v, langs)
        return
    if a.cmd == "qc":
        from dubadabidu.qc import evaluate
        from dubadabidu.qc import backcheck
        for v in a.rest:
            backcheck.run(cfg, v, langs)
            evaluate.run(cfg, v, langs)
            report(cfg, v)
        return
    if a.cmd == "evaluate":
        from dubadabidu.qc import evaluate
        for v in a.rest:
            evaluate.run(cfg, v, langs)
        return
    if a.cmd == "review":
        from dubadabidu.review import review_page
        for v in a.rest:
            review_page.run(cfg, v, langs)
        return
    if a.cmd == "verdicts":  # dubadabidu verdicts <video> <exported.json>
        from dubadabidu.review import verdicts
        if len(a.rest) != 2:
            ap.error("verdicts takes exactly: <video> <exported-ratings.json>")
        verdicts.run(cfg, a.rest[0], a.rest[1])
        return
    if a.cmd == "tune":
        from dubadabidu.lab import tune as tune_mod
        for v in a.rest:
            tune_mod.run(cfg, v, langs)
        return
    if a.cmd == "bakeoff":   # engine head-to-head (Phase C); needs s3 done
        from dubadabidu.lab import bakeoff
        for v in a.rest:
            bakeoff.run(cfg, v, langs)
        return
    if a.cmd == "remote":    # M2: RunPod lifecycle. `remote <task> [video]`
        from dubadabidu.infra.remote import runpod_infra as rpi
        task = a.rest[0] if a.rest else "status"
        if task == "status":
            rpi.status(); return
        if task == "kill":
            rpi.sweep_orphans(); print("[remote] swept orphaned pods"); return
        if task == "smoke":
            sys.exit(0 if rpi.smoke_test(cfg) else 1)
        if task in ("setup-check", "setup_check"):
            # dry-run the bake-off install path on a cheap pod (no comparison):
            # reports which engine_setup snippets import. Run before a real
            # `remote bakeoff` to avoid debugging installs on a billing run.
            sys.exit(0 if rpi.setup_check(cfg, a.budget) else 1)
        vids = a.rest[1:]
        if not vids:
            ap.error(f"remote {task} needs a video path")
        if task == "run":
            sys.exit(0 if _remote_run_batch(cfg, vids, langs, a, rpi) else 1)
        ok = all(rpi.remote_run(cfg, v, langs, task, a.budget,
                                keep_alive=a.keep_alive,
                                reuse=a.reuse,
                                overlays=a.overlay or [],
                                stages=_remote_stages(a)) for v in vids)
        sys.exit(0 if ok else 1)
    if a.cmd == "prep":
        from dubadabidu.lab import prep as prep_mod
        for v in a.rest:
            prep_mod.run(cfg, v)
        return
    if a.cmd == "preamble":
        from dubadabidu.workflows.preamble import preamble
        for v in a.rest:
            preamble(cfg, v, langs)
        return
    if a.cmd == "autopilot":
        from dubadabidu.workflows import autopilot as autopilot_mod
        autopilot_mod.main(cfg, a.rest, langs, a.spec, mux=a.mux)
        return

    s4_idx = ORDER.index("s4_synthesize")
    failed: list[str] = []
    for v in a.rest:  # run
        # allow running without the source video when s1/s2 are already cached
        # (the remote GPU path uploads work/ + ref/, never the 4K video)
        if not Path(v).exists() and not M.manifest_path(cfg, v).exists():
            sys.exit(f"not found: {v} (and no cached manifest — run s1+s2 first)")
        # --force on a run that reaches s4: discard cached takes so changed
        # take-selection params (best_of, min_f0st, ...) actually re-apply
        # (synth_hash omits them by design). Guarded so it can't wipe state a
        # from-a-later-stage run depends on, or run before s1/s2 exist.
        if a.force and M.manifest_path(cfg, v).exists() \
                and ORDER.index(a.from_stage) <= s4_idx:
            n = M.clear_synth(cfg, v, langs)
            logging.info("--force: cleared %d cached takes for %s %s — "
                         "take-selection params re-apply", n, langs, v)
        logging.info("=== %s ===", v)
        started = False
        try:
            for name in ORDER:
                started = started or name == a.from_stage
                if started:
                    try:
                        STAGES[name](cfg, v, langs)
                    except subprocess.CalledProcessError as e:
                        sys.exit(f"[{name}] external command failed: {e}")
                if name == a.to_stage:
                    break
        except (Exception, SystemExit) as e:
            # --keep-going: one bad video must not cost the rest of a batch
            # its run (the remote batch does every video in one process)
            if not a.keep_going:
                raise
            logging.error("=== %s FAILED (%s) — continuing with the next "
                          "video ===", v, e)
            failed.append(v)
            continue
        logging.info("=== %s done ===", v)
    if failed:
        sys.exit(f"{len(failed)} video(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
