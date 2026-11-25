# SPDX-License-Identifier: AGPL-3.0-or-later
"""Typed schema for the merged config (PLAN_v0.5 P7).

The CI header lists "a config key that never existed" among the bugs that
shipped: a misspelled key in YAML is silently ignored by `cfg.get(...)`, so the
code runs on its default and nobody notices. Every block here is
`extra="forbid"`, and `validate()` runs on the MERGED config (base + overlays)
in dub.py, so a typo fails at load with its full path.

`validate()` returns the typed, frozen Config the CLI hands to every stage
(ARCHITECTURE_PLAN.md §3). While the stages move onto attribute access, the
model also answers raw reads — cfg["tts"], cfg.get(...), "runpod" in cfg —
from the merged dict it was built from, with exactly the semantics of that
dict (a missing key is missing, so a code default still applies where code
still reads one). tests/config/ proves the two views agree and lists every
place a code default differs from a default declared here. The raw view is
deleted once nothing reads through it.

Adding a config key: declare it here (with its type) in the same change.
"""
from __future__ import annotations

from typing import Literal, TypeVar

from collections.abc import ItemsView, KeysView

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

Number = float | int


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- transition: the merged dict, for code not yet on attribute access ---
    _raw: dict[str, object] = PrivateAttr(default_factory=dict)

    def __getitem__(self, key: str) -> object:
        return self._raw[key]

    def get(self, key: str, default: object = None) -> object:
        return self._raw.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def keys(self) -> KeysView[str]:
        return self._raw.keys()

    def items(self) -> ItemsView[str, object]:
        return self._raw.items()


class Asr(_Block):
    model: str
    device: str = "auto"
    backend: Literal["auto", "faster", "mlx"] = "auto"
    mlx_model: str | None = None
    vad_filter: bool = True
    max_chars: int
    max_seconds: Number
    pause_split_s: Number
    glossary_prompt: bool = True
    initial_prompt: str = ""


class Separation(_Block):
    enabled: bool = True
    backend: Literal["roformer", "demucs"] = "roformer"
    device: str | None = None
    roformer_model: str
    demucs_model: str


class Translation(_Block):
    base_url: str
    model: str
    reflect_model: str | None = None
    adequacy_model: str | None = None
    api_key_env: str
    response_format: Literal["json_object", "json_schema", "text"] = "json_object"
    batch_size: int
    max_context_chars: int = 12000
    passes: Literal[1, 3] = 1
    adequacy_check: bool = True
    adequacy_flag: int = 3
    avoid_mis_stressed: bool = True    # s3_translate reads .get(..., True)
    avoid_min_marks: int = 1
    isometric_tolerance: Number
    n_short_variants: int
    lang_workers: int | None = None


class Tts(_Block):
    engine: Literal["qwen", "edge"]
    engine_by_lang: dict[str, Literal["qwen", "edge"]] = Field(default_factory=dict)
    qwen_model_dir: str | None = None
    qwen_model_dir_by_lang: dict[str, str] = Field(default_factory=dict)
    qwen_x_vector_only: bool = True
    qwen_fast: bool = False   # the loader's default (tts_engine._load_qwen)
    qwen_flash_attn: bool = False
    qwen_warmup_len: int = 100   # CUDA-graph warmup prefill (faster-qwen3-tts)
    qwen_max_audio_s: Number = 60
    qwen_gen_kwargs: dict[str, Number | bool] = Field(default_factory=dict)
    allow_cpu_fallback: bool = False
    reference_wav: str
    reference_text: str = ""
    reference_max_s: Number | None = None
    ru_stress: bool = False
    ru_stress_force: bool = False
    variant_label: str | None = None
    best_of: int
    rank_takes: bool = True
    min_f0st: Number = 0.0
    f0_reroll_max: int = 0
    retake_mos_below: Number
    best_of_early_accept: bool = True
    early_accept_mos: Number
    early_accept_sim: Number
    sample_rate: int
    edge_voices: dict[str, str]
    # chatterbox-era knobs: no engine reads them; synth_hash keys on their
    # defaults. Declared so old experiment overlays still validate.
    cfg_weight: Number | None = None
    exaggeration: Number | None = None


class Fit(_Block):
    max_tempo: Number
    soft_tempo: Number
    stretcher: Literal["auto", "rubberband", "atempo"] = "auto"
    drift_max_s: Number
    min_gap_s: Number


class SegmentNorm(_Block):
    enabled: bool = True
    target_dbfs: Number = -20.0


class Master(_Block):
    enabled: bool = True
    hpf_hz: Number
    comp_threshold_db: Number
    comp_ratio: Number
    comp_attack_ms: Number
    comp_release_ms: Number
    deess_hz: Number
    deess_gain_db: Number
    reverb_wet: Number
    reverb_room: Number
    limiter_db: Number


class Mix(_Block):
    bg_gain_db: Number
    lufs: Number
    duck: bool = True
    duck_threshold: Number
    duck_ratio: Number
    duck_attack_ms: Number
    duck_release_ms: Number
    segment_norm: SegmentNorm
    master: Master


class Mux(_Block):
    subtitle_codec: str
    lang_tags: dict[str, str]
    per_language: bool = True


class EvalWeights(_Block):
    sim: Number
    mos: Number
    f0: Number
    tempo: Number


class QcEval(_Block):
    weights: EvalWeights
    score_flag: Number
    sim_floor_default: Number


class QcRefit(_Block):
    min_rows: int
    step: Number
    folds: int
    min_rho: Number
    max_p: Number
    permutations: int
    adopt_margin: Number


class Qc(_Block):
    wer_flag_threshold: Number
    metrics_device: Literal["cpu", "cuda", "auto"] = "cpu"
    eval: QcEval
    refit: QcRefit


class Prep(_Block):
    """`dubadabidu prep` (comments moved from prep.PREP_DEFAULTS)."""
    min_s: Number
    max_s: Number
    n_refs: int             # keep the pool WIDE — tune R1 makes the real call
    max_candidates: int     # cap gated spans (each = a cut + MOS/f0 pass)
    # quality GATE (reject broken spans only — deliberately lenient so a good
    # ref is never filtered out; tune R1 does the actual selecting):
    clip_peak: Number       # peak at/above this = clipped, rejected
    min_voiced: Number      # <25% voiced = mostly silence, rejected
    min_mos: Number         # DistillMOS below this = noisy/artifacty, rejected


class R1Weights(_Block):
    sim: Number
    mos: Number
    f0: Number


class Tune(_Block):
    """`dubadabidu tune`: the reference sweep (comments moved from tune.DEFAULTS)."""
    subset_size: int
    refs_glob: str
    best_of: int
    rounds: list[Literal["R1", "R2", "R3"]] = Field(   # subset for tune-lite, e.g. ["R1"]
        default_factory=lambda: ["R1", "R2", "R3"])
    # ---- tune's OWN objective. NOT qc.eval.weights, and that is the fix. ----
    # qc.eval.weights ranks TAKES of one segment. It dropped sim to 0.0 on
    # 2026-08-09 because sim does not predict this listener's per-take judgement
    # (en rho -0.075 p 0.42, ru -0.136 p 0.15). Correct there — and silently
    # catastrophic here, because tune R1 selects the CLONE SOURCE, where
    # similarity is the entire question. Measured with the shipped weights:
    #     ref A  sim_cal .80  mos 4.0  f0 2.0  -> 0.662
    #     ref B  sim_cal .10  mos 4.0  f0 2.6  -> 0.715   <- wins
    # A reference that sounds nothing like the speaker beat one that does, on
    # six tenths of a semitone, and `preamble` wrote that winner into the
    # manifest with no human in the loop.
    #
    # UNVALIDATED, and labelled so on purpose. sim_raw + mos is less degenerate
    # than an objective with no identity term at all; it is NOT calibrated
    # against the ear. The one blind test run picked the LOWEST-sim reference
    # (FINDINGS 2.2, 2026-08-04) and the 42-judgement comparison came back
    # 21-21. Treat R1 as a shortlist generator, not an arbiter — the ear
    # decides, which is why the override gate below defaults to off.
    r1_weights: R1Weights
    # Take-to-take spread here is mos +/-0.24-0.49 and f0st +/-0.438 (measured
    # 2026-08-01 by running one config twice). Scoring a reference on ONE roll
    # per segment therefore ranks the dice; qc/listen.py says so in as many
    # words. Average this many rolls per point instead. Cost is
    # refs x subset_size x this, and it is the one place in the project where
    # paying for more samples buys a decision rather than a number.
    r1_takes: int = 3
    # ---- per-video reference override: OFF, matching FINDINGS 2.2 ----------
    # FINDINGS 2.2 records the verdict as "ENCODED — the reference is now a
    # config default, not a per-video override", because source quality varies
    # per lesson and a bad lesson yields a bad reference (its ref_04 was rated
    # 12-of-18 unusable). Only the ROOT CAUSE was fixed (refs_glob was widened
    # so R1 can at least see the curated pool); `preamble` still wrote
    # man["tts_overrides"] on every video, and s4/s5/evaluate/bakeoff all merge
    # it. The refuted mechanism was still live and regenerating itself.
    # true + a margin re-enables it, gated: see dub.preamble.
    per_video_ref_override: bool = False
    override_margin: Number = 0.05


class BakeoffTune(_Block):
    enabled: bool = True
    subset_size: int = 3    # tune-lite's own subset (cost = points x segs x takes)
    takes: int = 2


class Bakeoff(_Block):
    engines: list[Literal["qwen", "edge"]]
    incumbent: Literal["qwen", "edge"] = "qwen"
    subset_size: int = 6      # comparison segments, stratified by length
    takes: int = 3            # averaged per segment to beat take variance
    scoring_workers: int = 1
    text_variant: str | None = None
    tune: BakeoffTune | None = None
    grids: dict[str, dict[str, list[str | Number]]] = Field(default_factory=dict)


class Runpod(_Block):
    """`dubadabidu remote` (comments moved from runpod.DEFAULTS). Every field
    has a default, so `remote status`, `remote kill`, doctor and --reuse work
    without the gpu overlay."""
    name: str | None = "dubadabidu"
    # cheapest adequate community GPUs (>=16GB), in priority order; the API
    # picks the first available. Prices ~$0.16-0.22/hr (checked 2026-07).
    gpu_type_ids: list[str] = Field(default_factory=lambda: [
        "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
        "NVIDIA RTX A4500", "NVIDIA RTX A4000"])
    gpu_count: int | None = 1
    # Ubuntu 24.04 => python3.12 (the old py3.10 image capped numpy at 2.2.6);
    # cu129 base image; the torch in the venv is cu130 and needs a 13.0 driver.
    # Verified present on Docker Hub 2026-08-02 (stable, not an -rc tag).
    image: str = "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404"
    registry_auth_id: str | None = None
    # Filters to hosts whose DRIVER supports these CUDA versions. Without it the
    # scheduler handed us a CUDA 12.4 driver, which caps torch at 2.6.0. Enum is
    # 11.8..13.0 (RunPod OpenAPI). Empty/absent => no filter, old behaviour.
    # TRADE-OFF: every entry removed here shrinks the host pool, and this project
    # has lost runs to capacity before. Widen this list first if provisioning
    # starts failing with "no instances currently available".
    min_cuda_version: str | None = "13.0"   # driver floor >= torch's cu-build (cu130)
    # v1-style exact list; mutually exclusive (400)
    allowed_cuda_versions: list[str] | None = Field(default_factory=list)
    container_disk_gb: int = 30
    volume_gb: int | None = 0               # ephemeral; all state rsync'd back
    network_volume_id: str | None = None
    # COMMUNITY had no on-demand capacity (FINDINGS §3)
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    # REST v2 has NO spot pods. Kept as a key only so an old overlay that sets
    # it is reported instead of silently ignored (create_pod warns).
    interruptible: bool = False
    ports: list[str] | None = Field(default_factory=lambda: ["22/tcp"])
    budget_usd: Number = 10.0           # hard session cap (overridden by --budget / spec)
    assumed_price_per_hr: Number = 0.25 # conservative: deadline = budget / this
    max_runtime_hours: Number = 3.0     # absolute ceiling regardless of budget
    ssh_key: str = "~/.ssh/id_ed25519_runpod"
    ssh_user: str | None = "root"
    remote_dir: str | None = "~/dubadabidu"
    provision_timeout_s: int | None = 420   # wait this long for the pod to reach RUNNING+SSH
    # HOTFIX HOLD: when the task fails on a pod that bootstrapped, keep it up
    # this many minutes (pod-side watchdog re-armed to it) instead of
    # terminating, so the fix is tested on the same pod with no new bootstrap.
    # The next `remote` command reuses it automatically. 0 = terminate always.
    hold_on_failure_min: Number | None = 30
    # no GPU type has capacity: re-sweep the list every capacity_retry_s for
    # up to capacity_wait_min before failing (creating nothing costs nothing)
    capacity_wait_min: Number | None = 15
    capacity_retry_s: Number | None = 60
    # per-engine install snippets for the bake-off (git-clone challengers —
    # THIRD_PARTY.md). Each runs on the pod in that engine's OWN venv
    # (installed into .venv by engine_install_cmd — the per-engine venvs were
    # untouchable by construction); empty => that engine is marked unavailable
    # and skipped. Fill with PINNED commands once validated.
    # e.g. {"qwen": "git clone ... && pip install ..."}
    engine_setup: dict[str, str] = Field(default_factory=dict)


class Config(_Block):
    work_dir: str
    output_dir: str
    languages: list[str]
    source_language: Literal["uk", "ru"]
    asr: Asr
    separation: Separation
    translation: Translation
    tts: Tts
    fit: Fit
    mix: Mix
    mux: Mux
    qc: Qc
    prep: Prep
    tune: Tune
    bakeoff: Bakeoff | None = None
    runpod: Runpod = Field(default_factory=Runpod)



class ConfigError(SystemExit):
    """Raised (as a clean exit) when the merged config does not validate."""


def validate(cfg: dict) -> Config:
    """Validate the MERGED config; exit with every bad key's dotted path."""
    try:
        model = Config.model_validate(cfg)
    except ValidationError as e:
        lines = [f"  {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                 for err in e.errors()]
        raise ConfigError("config does not validate (a typo, or a key this "
                          "schema does not declare — pipeline/config_schema.py):\n"
                          + "\n".join(lines)) from None
    _attach_raw(model, cfg)
    return model


def _attach_raw(block: _Block, raw: dict[str, object]) -> None:
    """Give every block the slice of the merged dict it was validated from."""
    block._raw = raw
    for name in type(block).model_fields:
        sub, sub_raw = getattr(block, name), raw.get(name)
        if isinstance(sub, _Block) and isinstance(sub_raw, dict):
            _attach_raw(sub, sub_raw)


B = TypeVar("B", bound=_Block)


def derive(block: B, **updates: object) -> B:
    """`block` with some fields replaced — for the tts block a manifest's
    tts_overrides or a bake-off grid point produce. Both views move together:
    attribute reads see the update and so do raw reads during the transition."""
    new = block.model_copy(update=updates)
    new._raw = {**block._raw, **updates}
    return new
