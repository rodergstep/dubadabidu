# SPDX-License-Identifier: AGPL-3.0-or-later
"""The argparse tree of the `dubadabidu` command."""
from __future__ import annotations

import argparse

from dubadabidu.cli.stages import ORDER


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--engine", default=None, choices=["qwen", "edge"])
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
    ap.add_argument("--keep-alive", dest="keep_alive", action="store_true",
                    help="`remote`: do NOT terminate the pod when the task "
                         "ends, so a failing engine install can be debugged "
                         "IN PLACE over SSH instead of paying a fresh ~4min "
                         "bootstrap per attempt. Prints the ssh command. The "
                         "pod-side watchdog still self-destructs at the "
                         "deadline; `remote kill` ends it sooner.")
    ap.add_argument("--reuse", action="store_true",
                    help="`remote`: attach to the pod a previous --keep-alive "
                         "run left running instead of provisioning a new one. "
                         "Skips the bootstrap, whose torch download is 4-6 GB "
                         "and took anywhere from 4 to 60 min depending on the "
                         "pod's route to PyPI. Falls back to provisioning if "
                         "no live pod is tracked.")
    ap.add_argument("--keep-going", dest="keep_going", action="store_true",
                    help="`run` over several videos: log a failing video and "
                         "continue with the next instead of stopping")
    ap.add_argument("--budget", type=float, default=None,
                    help="hard USD cap for `remote` (auto-terminates the pod); "
                         "default runpod.budget_usd")
    ap.add_argument("--project", default=None,
                    help="project directory (config.yaml, glossary/, ref/, work/, "
                         ".env); default $DUBADABIDU_HOME, else the nearest "
                         "directory above the cwd that holds config.yaml")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--overlay", action="append", default=None,
                    help="yaml deep-merged over --config; REPEATABLE, applied "
                         "left-to-right (later wins). Compose independent axes, "
                         "e.g. GPU/TTS + translation provider: --overlay "
                         "config.gpu.yaml --overlay config.deepseek.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap
