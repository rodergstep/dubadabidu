#!/usr/bin/env python3
"""dubadabidu — Ukrainian -> multilingual video dubbing.

  dubadabidu doctor                          # validate environment before anything
  dubadabidu preamble input/a.mp4            # per-video prep: refs + ref pick (2 passes)
  dubadabidu run input/*.mp4 [--langs en,de] [--from s4] [--engine edge]
  dubadabidu stage s3_translate input/a.mp4
  dubadabidu qc input/a.mp4
  dubadabidu report input/a.mp4              # per-language fit/QC summary table
"""
from __future__ import annotations
import argparse, logging, shutil, subprocess, sys
from pathlib import Path
import yaml

from pipeline import s1_extract, s2_transcribe, s3_translate, s4_synthesize, \
    s5_fit, s6_mix, s7_subtitles, s8_mux, prep as prep_mod, tune as tune_mod, \
    autopilot as autopilot_mod, manifest as M
from pipeline.device import torch_device, whisper_device
from pipeline.logic import deep_merge
from qc import backcheck, bakeoff, batch_report, evaluate, refit, review_page, \
    verdicts

ROOT = Path(__file__).resolve().parent

STAGES = {
    "s1_extract":    lambda c, v, l: s1_extract.run(c, v),
    "s2_transcribe": lambda c, v, l: s2_transcribe.run(c, v),
    "s3_translate":  lambda c, v, l: s3_translate.run(c, v, l),
    "s4_synthesize": lambda c, v, l: s4_synthesize.run(c, v, l),
    "s5_fit":        lambda c, v, l: s5_fit.run(c, v, l),
    "s6_mix":        lambda c, v, l: s6_mix.run(c, v, l),
    "s7_subtitles":  lambda c, v, l: s7_subtitles.run(c, v, l),
    "s8_mux":        lambda c, v, l: s8_mux.run(c, v, l),
}
ORDER = list(STAGES)


def doctor(cfg: dict) -> int:
    ok = True

    def check(name, cond, hint=""):
        nonlocal ok
        print(f"  [{'OK' if cond else '!!'}] {name}" + ("" if cond else f" — {hint}"))
        ok = ok and bool(cond)

    print("dubadabidu doctor")
    check("ffmpeg", shutil.which("ffmpeg"), "install ffmpeg and add to PATH")
    check("ffprobe", shutil.which("ffprobe"), "comes with ffmpeg")
    dev = torch_device()
    from pipeline.asr import resolve_backend
    backend = resolve_backend(cfg["asr"])
    if backend == "mlx":
        print(f"  [i ] torch device: {dev} | whisper: mlx-whisper (Metal/ANE)")
    else:
        wdev, wct = whisper_device(cfg["asr"].get("device", "auto"))
        print(f"  [i ] torch device: {dev} | whisper: faster-whisper {wdev}/{wct}")
        if dev == "mps":
            print("  [i ] ASR on CPU int8: `pip install .[mac]` for mlx-whisper "
                  "(~4-5x on Apple Silicon).")
    if dev != "cuda":
        print("  [i ] no CUDA: use tts.engine=edge for pipeline validation; "
              "run the Chatterbox batch on a CUDA machine (RunPod/vast.ai).")
    for mod in ["faster_whisper", "pydub", "soundfile", "srt", "yaml", "openai"]:
        try:
            __import__(mod); check(f"python: {mod}", True)
        except ImportError:
            check(f"python: {mod}", False, "pip install -r requirements.txt")
    sep = cfg.get("separation", {})
    if sep.get("enabled") and sep.get("backend", "roformer") == "roformer":
        try:
            __import__("audio_separator"); check("python: audio_separator", True)
        except ImportError:
            check("python: audio_separator", False,
                  'pip install "audio-separator[cpu]" (or separation.backend: demucs)')
    engines = {cfg["tts"]["engine"], *cfg["tts"].get("engine_by_lang", {}).values(),
               *cfg.get("bakeoff", {}).get("engines", [])}
    # each engine is probed where synthesis would actually run it: inside
    # venvs/<engine> when that isolated venv exists (engine_client routes
    # through a worker there), otherwise in THIS venv (in-process).
    for eng, mod, hint in [
        ("chatterbox", "chatterbox", "pip install chatterbox-tts==0.1.7"),
        ("cosyvoice", "cosyvoice",
         "git clone --recursive FunAudioLLM/CosyVoice into its own venv "
         "(THIRD_PARTY.md)"),
        ("indextts", "indextts",
         "git clone index-tts/index-tts + checkpoints into its own venv "
         "(THIRD_PARTY.md)"),
        ("voxcpm", "voxcpm",
         "pip install voxcpm==2.0.3 in its own venv (THIRD_PARTY.md)"),
        ("qwen", "qwen_tts",
         "git clone QwenLM/Qwen3-TTS + pip install -e . in its own venv "
         "(THIRD_PARTY.md)"),
    ]:
        if eng not in engines:
            continue
        vpy = ROOT / "venvs" / eng / "bin" / "python"
        if vpy.exists():
            r = subprocess.run([str(vpy), "-c", f"import {mod}"],
                               capture_output=True)
            check(f"python: {mod} (venvs/{eng})", r.returncode == 0, hint)
        else:
            try:
                __import__(mod); check(f"python: {mod}", True)
            except ImportError:
                check(f"python: {mod}", False, hint)
    if engines - {"edge"}:
        check("reference_wav", Path(cfg["tts"]["reference_wav"]).exists(),
              f"put a 15-20s clean voice clip at {cfg['tts']['reference_wav']} "
              f"or run `dubadabidu prep <video>`")
    import os
    check(f"env {cfg['translation']['api_key_env']}",
          os.environ.get(cfg["translation"]["api_key_env"]),
          "export it, or point translation.base_url at a local Ollama")
    # remote (M2) readiness — informational; only needed for `dubadabidu remote`
    if os.environ.get("RUNPOD_API_KEY"):
        rp = cfg.get("runpod", {})
        key_path = Path(rp.get("ssh_key", "~/.ssh/id_ed25519_runpod")).expanduser()
        check("remote: ssh_key", key_path.exists(),
              f"private key {key_path} missing; register its .pub with RunPod")
        print(f"  [i ] remote ready: RUNPOD_API_KEY set, budget cap "
              f"${rp.get('budget_usd', 10)}")
    else:
        print("  [i ] remote (`dubadabidu remote`) off: set RUNPOD_API_KEY in "
              ".env to enable GPU pod automation")
    print("doctor:", "all good" if ok else "fix items above")
    return 0 if ok else 1


def preamble(cfg: dict, video: str, langs: list[str]) -> None:
    """Per-video preamble (IMPROVEMENT_PLAN Phase D), resumable:
    pass 1: s1 + s2 + prep, then pause for the text_uk hand-review;
    pass 2 (re-run):  s3 on the tune language, tune-lite R1 over this video's
    own refs, winning ref stored in manifest tts_overrides (s4/s5/evaluate
    honor it — no config paste per video)."""
    import json
    stem = Path(video).stem
    fresh = not M.manifest_path(cfg, video).exists()
    s1_extract.run(cfg, video)
    if fresh:
        s2_transcribe.run(cfg, video)
    wd = M.video_workdir(cfg, video)
    if not (wd / "refs.json").exists():
        prep_mod.run(cfg, video)
    if fresh:
        print(f"\n[preamble] paused — review text_uk in "
              f"{M.manifest_path(cfg, video)} (one fix here propagates to all "
              f"languages), then re-run:\n  dubadabidu preamble {video}")
        return
    lang = langs[0]
    s3_translate.run(cfg, video, [lang])
    tcfg = deep_merge(cfg, {"tune": {"refs_glob": f"ref/{stem}_ref_*.wav",
                                     "subset_size": 5, "rounds": ["R1"]}})
    win = tune_mod.run(tcfg, video, [lang])
    man = M.load(cfg, video)
    over = {"reference_wav": win["reference_wav"]}
    refs = json.loads((wd / "refs.json").read_text(encoding="utf-8"))
    rt = refs.get(Path(win["reference_wav"]).name, {}).get("text_uk", "")
    if rt:  # ref transcript — required by the cosyvoice engine
        over["reference_text"] = rt
    man["tts_overrides"] = over
    M.save(cfg, video, man)
    print(f"\n[preamble] done — this video's ref: {win['reference_wav']} "
          f"(stored in manifest tts_overrides)."
          f"\nNext: skim {wd}/terms_{lang}.json, then:"
          f"\n  dubadabidu run {video} --from s3_translate")


def report(cfg: dict, video: str) -> None:
    man = M.load(cfg, video)
    wd = M.video_workdir(cfg, video)
    print(f"\n{video} — {len(man['utterances'])} utterances, "
          f"{man['duration']:.0f}s | stages: {man.get('stages', {})}")
    langs = sorted({l for u in man["utterances"] for l in u["tr"]})
    stale_note = []
    score_flag = cfg["qc"].get("eval", {}).get("score_flag", 0.55)
    adeq_flag = cfg["translation"].get("adequacy_flag", 3)
    hdr = f"{'lang':5} {'engine':>10} {'ok':>4} {'stretch':>8} {'shorten':>8} " \
          f"{'overflow':>9} {'overlap':>8} {'wer>thr':>8} {'score<f':>8} {'adeq<f':>7}"
    print(hdr); print("-" * len(hdr))
    for lang in langs:
        trs = [u["tr"].get(lang, {}) for u in man["utterances"]]
        st = M.stale_qc(wd, man, lang)
        if st["score"] or st["wer"]:
            stale_note.append(f"{lang} ({len(st['score'])} score, "
                              f"{len(st['wer'])} wer)")
        engine = "⚠EDGE" if M.edge_langs(man, [lang]) else \
            next((t["synth_engine"] for t in trs if t.get("synth_engine")), "-")
        fits = [t.get("fit") for t in trs]
        wer_bad = sum(1 for t in trs
                      if t.get("qc_wer", 0) > cfg["qc"]["wer_flag_threshold"])
        score_bad = sum(1 for t in trs if "qc_score" in t
                        and t["qc_score"] < score_flag)
        # translation faithfulness flags (s3 adequacy judge), if scored
        adeq_bad = sum(1 for t in trs if "adequacy" in t
                       and t["adequacy"].get("score", 5) < adeq_flag)
        # drift_exceeded superseded overrun_s when s5 went soft-anchor
        # (overlap in the mix is impossible now); old manifests keep overrun_s
        overlaps = sum(1 for t in trs
                       if t.get("overrun_s") or t.get("drift_exceeded"))
        print(f"{lang:5} {engine:>10} {fits.count('ok'):>4} {fits.count('stretched'):>8} "
              f"{fits.count('shortened'):>8} {fits.count('overflow'):>9} "
              f"{overlaps:>8} {wer_bad:>8} {score_bad:>8} {adeq_bad:>7}")
    if stale_note:
        print(f"\n  !! STALE QC — scores describe audio that s5/s6 has since "
              f"rewritten: {', '.join(stale_note)}."
              f"\n     The numbers above (and any review page or ratings row "
              f"built from them) are not current."
              f"\n     Re-score:  dubadabidu qc {video}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="dubadabidu")
    ap.add_argument("cmd", choices=["run", "stage", "qc", "doctor", "report",
                                    "evaluate", "review", "tune", "prep",
                                    "preamble", "batch", "autopilot",
                                    "verdicts", "bakeoff", "remote", "refit"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--langs", default=None)
    ap.add_argument("--from", dest="from_stage", default="s1_extract",
                    choices=ORDER)
    ap.add_argument("--to", dest="to_stage", default="s8_mux", choices=ORDER,
                    help="stop after this stage (e.g. --to s7_subtitles to "
                         "produce audio+subs but skip mux — the remote GPU path "
                         "muxes locally so the source video never uploads)")
    ap.add_argument("--engine", default=None, choices=["chatterbox", "edge"])
    ap.add_argument("--spec", default=None,
                    help="acceptance spec for `autopilot` "
                         "(default specs/batch.yaml)")
    ap.add_argument("--no-mux", dest="mux", action="store_false",
                    help="`autopilot`: stop at s7 (audio+subs), skip the mux — "
                         "the remote GPU path muxes locally so the source video "
                         "never uploads")
    ap.add_argument("--force", action="store_true",
                    help="discard a cache so a changed setting re-applies. "
                         "`stage s3_translate`: drop translations + terms for "
                         "--langs (provider switch). `stage s4_synthesize` or a "
                         "`run` reaching s4: drop cached takes for --langs so "
                         "take-SELECTION params (best_of, min_f0st, ...) re-apply "
                         "— they're not in synth_hash, so the cache ignores them "
                         "otherwise. Keeps translations + human verdicts.")
    ap.add_argument("--budget", type=float, default=None,
                    help="hard USD cap for `remote` (auto-terminates the pod); "
                         "default runpod.budget_usd")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--overlay", action="append", default=None,
                    help="yaml deep-merged over --config; REPEATABLE, applied "
                         "left-to-right (later wins). Compose independent axes, "
                         "e.g. GPU/TTS + translation provider: --overlay "
                         "config.gpu.yaml --overlay config.deepseek.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # resolve user-supplied paths against the caller's cwd, then anchor to the
    # repo root so relative resources (config, glossary/, prompts/, ref/, work/)
    # work no matter where the CLI is invoked from
    import os
    config_path = Path(a.config).resolve() if Path(a.config).exists() \
        else ROOT / a.config
    a.rest = [str(Path(v).resolve()) if Path(v).exists() else v for v in a.rest]
    os.chdir(ROOT)

    # .env contract (AUTOPILOT.md): the human's only config touchpoint for
    # credentials. KEY=VALUE lines; the environment always wins over the file.
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    for ov in (a.overlay or []):  # stacked left-to-right; later overlays win
        overlay_path = Path(ov) if Path(ov).is_absolute() else ROOT / ov
        cfg = deep_merge(cfg, yaml.safe_load(open(overlay_path, encoding="utf-8")))
    if a.engine:
        cfg["tts"]["engine"] = a.engine
    langs = a.langs.split(",") if a.langs else cfg["languages"]

    if a.cmd == "doctor":
        sys.exit(doctor(cfg))
    if a.cmd == "batch":  # no args = all of work/; else the given videos
        batch_report.run(cfg, a.rest or None)
        return
    if a.cmd == "refit":  # M4: propose qc.eval.weights from accumulated ratings
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
        for v in a.rest:
            backcheck.run(cfg, v, langs)
            evaluate.run(cfg, v, langs)
            report(cfg, v)
        return
    if a.cmd == "evaluate":
        for v in a.rest:
            evaluate.run(cfg, v, langs)
        return
    if a.cmd == "review":
        for v in a.rest:
            review_page.run(cfg, v, langs)
        return
    if a.cmd == "verdicts":  # dubadabidu verdicts <video> <exported.json>
        if len(a.rest) != 2:
            ap.error("verdicts takes exactly: <video> <exported-ratings.json>")
        verdicts.run(cfg, a.rest[0], a.rest[1])
        return
    if a.cmd == "tune":
        for v in a.rest:
            tune_mod.run(cfg, v, langs)
        return
    if a.cmd == "bakeoff":   # engine head-to-head (Phase C); needs s3 done
        for v in a.rest:
            bakeoff.run(cfg, v, langs)
        return
    if a.cmd == "remote":    # M2: RunPod lifecycle. `remote <task> [video]`
        from pipeline import runpod_infra as rpi
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
        ok = all(rpi.remote_run(cfg, v, langs, task, a.budget) for v in vids)
        sys.exit(0 if ok else 1)
    if a.cmd == "prep":
        for v in a.rest:
            prep_mod.run(cfg, v)
        return
    if a.cmd == "preamble":
        for v in a.rest:
            preamble(cfg, v, langs)
        return
    if a.cmd == "autopilot":
        autopilot_mod.main(cfg, a.rest, langs, a.spec, mux=a.mux)
        return

    s4_idx = ORDER.index("s4_synthesize")
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
        for name in ORDER:
            started = started or name == a.from_stage
            if started:
                try:
                    STAGES[name](cfg, v, langs)
                except subprocess.CalledProcessError as e:
                    sys.exit(f"[{name}] external command failed: {e}")
            if name == a.to_stage:
                break
        logging.info("=== %s done ===", v)


if __name__ == "__main__":
    main()
