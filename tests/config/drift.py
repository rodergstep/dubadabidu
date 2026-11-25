# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every place a code default disagrees with the schema, and what happens to it.

Keys are inventory rows ("path kind default"); test_equivalence fails on a row
that drifts without an entry here, and on an entry that no longer drifts.
Decisions are the ones approved in ARCHITECTURE_PLAN.md §3.3 (Q8). Tags:

  SCHEMA      move the code's value into the schema as the field default
              (effective value unchanged; the code default is deleted)
  SCHEMA-WINS latent: shipped configs always set the key, so the schema's
              documented default becomes the only one
  CONST       not config: a named module constant (the key is undeclared, so
              no config could ever set it)
  HASH        a persisted cache-key format constant, kept inside synth_hash
  ABSENT      differs only when the whole block is missing from the config
  EQUIV       the site's comparison makes the two values equivalent
  BUG         preserved during the refactor; fixed separately if approved
"""
from __future__ import annotations

_RUNPOD_SCHEMA = ("SCHEMA (Q8b): runpod.DEFAULTS values become Runpod field "
                  "defaults; applies when the key or the whole runpod block is "
                  "absent (remote status/kill, doctor, --reuse without the gpu "
                  "overlay). Relaxes validation of a PARTIAL runpod block.")
_UNDECLARED = ("CONST (Q8d): read with a default but rejected by extra=forbid, "
               "so it is a constant dressed as config")

DRIFT: dict[str, str] = {
    # --- asr ---
    "asr.compression_ratio_threshold get 2.4": _UNDECLARED,
    "asr.condition_on_previous_text get False": _UNDECLARED,
    "asr.no_speech_threshold get 0.6": _UNDECLARED,
    "asr.temperature_fallback get (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)": _UNDECLARED,
    "asr.vad_filter get <falsy>": "SCHEMA-WINS: asr.py:121 reads absent as off; "
                                  "config.yaml sets true",
    # --- separation / mix / mux ---
    "separation.enabled get <falsy>": "SCHEMA-WINS: doctor reads absent as off",
    "separation.device get ''": "EQUIV: s1 compares str(value).lower() == 'cpu'",
    "mix.master.enabled get <falsy>": "SCHEMA-WINS: s6 and doctor read absent as off",
    "mux.per_language get <falsy>": "SCHEMA-WINS: s8 reads absent as off",
    # --- tts ---
    "tts.qwen_fast get True": "BUG (§10.2 #1): engine_available assumes True, "
                              "the loader and synth_hash False; preserved "
                              "until a separate fix: commit",
    "tts.qwen_warmup_len get 100": "SCHEMA: 100 becomes the field default",
    "tts.qwen_x_vector_only get <falsy>": "EQUIV: bakeoff.variant_key tests "
                                          "`is False`, so absent == True",
    "tts.cfg_weight get 0.0": "HASH: chatterbox-era salt kept at its old default "
                              "so existing seg/ caches keep their keys",
    "tts.exaggeration get 0.55": "HASH: chatterbox-era salt, as cfg_weight",
    # --- tune ---
    "tune.rounds defaults ['R1', 'R2', 'R3']": "SCHEMA: becomes the field default",
    "tune.r1_takes defaults 3": "SCHEMA: 3 (tune.DEFAULTS, which applies) "
                                "replaces the schema's 1",
    # --- bakeoff (block present only with the gpu overlay) ---
    "bakeoff get {}": "ABSENT (Q8c): Config.bakeoff stays optional; each site "
                      "keeps its absent-block meaning through one helper",
    "bakeoff.engines get []": "ABSENT (Q8c)",
    "bakeoff.scoring_workers get 1": "SCHEMA: 1 becomes the field default",
    "bakeoff.subset_size get 6": "SCHEMA: 6 becomes the field default",
    "bakeoff.takes get 3": "SCHEMA: 3 becomes the field default",
    "bakeoff.tune.enabled get True": "ABSENT: bakeoff.tune is optional; "
                                     "BakeoffTune.enabled is already True",
    "bakeoff.tune.subset_size get 3": "SCHEMA: 3 becomes the field default",
    "bakeoff.tune.takes get 2": "SCHEMA: 2 becomes the field default",
    # --- runpod ---
    "runpod get {}": _RUNPOD_SCHEMA,
    "runpod.allowed_cuda_versions defaults []": _RUNPOD_SCHEMA,
    "runpod.assumed_price_per_hr defaults 0.25": _RUNPOD_SCHEMA,
    "runpod.assumed_price_per_hr get 0": "ABSENT: provision() logging fallback; "
                                         "dead once the field has a default",
    "runpod.budget_usd defaults 10.0": _RUNPOD_SCHEMA,
    "runpod.capacity_retry_s defaults 60": _RUNPOD_SCHEMA,
    "runpod.capacity_retry_s get 60": _RUNPOD_SCHEMA,
    "runpod.capacity_wait_min defaults 15": _RUNPOD_SCHEMA,
    "runpod.capacity_wait_min get 15": _RUNPOD_SCHEMA,
    "runpod.cloud_type defaults 'SECURE'": _RUNPOD_SCHEMA,
    "runpod.container_disk_gb defaults 30": _RUNPOD_SCHEMA,
    "runpod.direct_ssh_wait_s get 150": _UNDECLARED,
    "runpod.engine_setup defaults {}": _RUNPOD_SCHEMA,
    "runpod.engine_setup get {}": _RUNPOD_SCHEMA,
    "runpod.gpu_count defaults 1": _RUNPOD_SCHEMA,
    "runpod.gpu_type_ids defaults ['NVIDIA RTX A5000', 'NVIDIA GeForce RTX 3090', "
    "'NVIDIA RTX A4500', 'NVIDIA RTX A4000']": _RUNPOD_SCHEMA,
    "runpod.hold_on_failure_min defaults 30": _RUNPOD_SCHEMA,
    "runpod.image defaults 'runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404'": _RUNPOD_SCHEMA,
    "runpod.interruptible defaults False": _RUNPOD_SCHEMA,
    "runpod.max_runtime_hours defaults 3.0": _RUNPOD_SCHEMA,
    "runpod.min_cuda_version defaults '13.0'": _RUNPOD_SCHEMA,
    "runpod.name get 'dubadabidu'": _RUNPOD_SCHEMA,
    "runpod.ports defaults ['22/tcp']": _RUNPOD_SCHEMA,
    "runpod.provision_attempts get 3": _UNDECLARED,
    "runpod.provision_timeout_s defaults 420": _RUNPOD_SCHEMA,
    "runpod.remote_dir defaults '~/dubadabidu'": _RUNPOD_SCHEMA,
    "runpod.ssh_key defaults '~/.ssh/id_ed25519_runpod'": _RUNPOD_SCHEMA,
    "runpod.ssh_key get '~/.ssh/id_ed25519_runpod'": _RUNPOD_SCHEMA,
    "runpod.ssh_user defaults 'root'": _RUNPOD_SCHEMA,
    "runpod.volume_gb defaults 0": _RUNPOD_SCHEMA,
}
