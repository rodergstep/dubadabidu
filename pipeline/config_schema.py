"""Typed schema for the merged config (PLAN_v0.5 P7).

The CI header lists "a config key that never existed" among the bugs that
shipped: a misspelled key in YAML is silently ignored by `cfg.get(...)`, so the
code runs on its default and nobody notices. Every block here is
`extra="forbid"`, and `validate()` runs on the MERGED config (base + overlays)
in dub.py, so a typo fails at load with its full path.

Scope: this VALIDATES; stages keep receiving the plain dict (`model_dump()`
of the same values), so no stage changed in this step. Optional fields are the
keys code reads with a default but no shipped config sets — they are declared
so an overlay may set them.

Adding a config key: declare it here (with its type) in the same change.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Number = float | int


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    avoid_mis_stressed: bool = False
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
    qwen_fast: bool = False
    qwen_flash_attn: bool = False
    qwen_warmup_len: int | None = None
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
    min_s: Number
    max_s: Number
    n_refs: int
    max_candidates: int
    clip_peak: Number
    min_voiced: Number
    min_mos: Number


class R1Weights(_Block):
    sim: Number
    mos: Number
    f0: Number


class Tune(_Block):
    subset_size: int
    refs_glob: str
    best_of: int
    rounds: list[Literal["R1", "R2", "R3"]] | None = None
    cfg_weights: list[Number] | None = None
    exaggerations: list[Number] | None = None
    r1_weights: R1Weights
    r1_takes: int = 1
    per_video_ref_override: bool = False
    override_margin: Number = 0.05


class BakeoffTune(_Block):
    enabled: bool = True
    subset_size: int | None = None
    takes: int | None = None


class Bakeoff(_Block):
    engines: list[Literal["qwen", "edge"]]
    incumbent: Literal["qwen", "edge"] = "qwen"
    subset_size: int | None = None
    takes: int | None = None
    scoring_workers: int | None = None
    text_variant: str | None = None
    tune: BakeoffTune | None = None
    grids: dict[str, dict[str, list[str | Number]]] = Field(default_factory=dict)


class Runpod(_Block):
    name: str | None = None
    gpu_type_ids: list[str]
    gpu_count: int | None = None
    image: str
    registry_auth_id: str | None = None
    min_cuda_version: str | None = None
    allowed_cuda_versions: list[str] | None = None
    container_disk_gb: int
    volume_gb: int | None = None
    network_volume_id: str | None = None
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    interruptible: bool = False
    ports: list[str] | None = None
    budget_usd: Number
    assumed_price_per_hr: Number
    max_runtime_hours: Number
    ssh_key: str
    ssh_user: str | None = None
    remote_dir: str | None = None
    provision_timeout_s: int | None = None
    engine_setup: dict[str, str] = Field(default_factory=dict)


class Config(_Block):
    work_dir: str
    output_dir: str
    languages: list[str]
    source_language: str
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
    runpod: Runpod | None = None


class ConfigError(SystemExit):
    """Raised (as a clean exit) when the merged config does not validate."""


def validate(cfg: dict) -> Config:
    """Validate the MERGED config; exit with every bad key's dotted path."""
    try:
        return Config.model_validate(cfg)
    except ValidationError as e:
        lines = [f"  {'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                 for err in e.errors()]
        raise ConfigError("config does not validate (a typo, or a key this "
                          "schema does not declare — pipeline/config_schema.py):\n"
                          + "\n".join(lines)) from None
