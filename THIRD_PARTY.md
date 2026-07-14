# Third-party TTS engines (bake-off candidates)

Chatterbox (the incumbent) and edge install from PyPI. The bake-off challengers
are **git-clone installs, GPU-only**, with their own heavy dependencies. Install
each in the **same venv ONLY after `pip check`** so they cannot silently move the
chatterbox torch pin (torch/torchaudio 2.6.0). If a conflict appears, install the
challenger in a **separate venv** and run the bake-off from there — the manifest
cache is shared via the work/ dir, so engines don't need to coexist in one env.

Pin the commit you validate on (record it below) — these repos move fast and
their inference APIs drift; `pipeline/tts_engine.py` targets the APIs noted here.

## CosyVoice 2/3 — all 5 target languages + cross-lingual (Apache-2.0)

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice third_party/CosyVoice
cd third_party/CosyVoice && git checkout <PIN_COMMIT> && git submodule update --init --recursive
pip check                        # BEFORE installing — must not disturb torch 2.6.0
pip install -r requirements.txt  # (or a fresh venv if pip check complains)
# model weights (pick one):
#   modelscope download iic/CosyVoice2-0.5B --local_dir pretrained_models/CosyVoice2-0.5B
export PYTHONPATH=$PWD:$PWD/third_party/Matcha-TTS:$PYTHONPATH
```

Config (`config.gpu.yaml` → `tts`): `cosyvoice_model_dir`, `cosyvoice_mode`.
- **cross_lingual** (default): audio prompt only — REQUIRED for the Ukrainian
  ref (UA is not a CosyVoice language; its transcript can't be tokenized).
- **zero_shot**: needs `reference_text` (English/target-lang ref only).
- **instruct**: per-segment emotion via `instruct_text` (e.g. "Speak warmly,
  unhurried") — transcript-free, so also UA-ref safe. Uses `inference_instruct2`.

API used: `inference_cross_lingual(text, prompt_16k)`,
`inference_zero_shot(text, prompt_text, prompt_16k)`,
`inference_instruct2(text, instruct_text, prompt_16k)`. Verify on the pinned
commit; adjust `_synth_cosyvoice` if the release differs.

Pinned commit: `__________`  (fill after validation)

## IndexTTS-2 — EN + Mandarin only, dubbing-native (Apache-2.0)

```bash
git clone https://github.com/index-tts/index-tts third_party/index-tts
cd third_party/index-tts && git checkout <PIN_COMMIT>
pip check
pip install -r requirements.txt
# download checkpoints into ./checkpoints (config.yaml + weights) per the repo
export PYTHONPATH=$PWD:$PYTHONPATH
```

Config (`config.gpu.yaml` → `tts`): `indextts_model_dir` (holds `config.yaml`).
- **Emotion transfer**: set `emotion_wav` to the source UA slice for a segment
  to carry YOUR original delivery's emotion (disentangled from timbre);
  `emo_alpha` (default 1.0) scales it. Or set `instruct_text` for a text emotion.
- **Duration**: `indextts_duration_ratio` (0.75–1.25) native pace control.
- Only `en` (and `zh`) are allowed; the adapter refuses other languages — route
  them to chatterbox/cosyvoice via `tts.engine_by_lang`.

API used: `IndexTTS2(cfg_path, model_dir, use_fp16=True).infer(spk_audio_prompt,
text, output_path, emo_audio_prompt=, emo_alpha=, use_emo_text=, emo_text=)`.
Verify on the pinned commit; adjust `_synth_indextts` if the API differs.

Pinned commit: `__________`  (fill after validation)

## VoxCPM2 — 30 languages incl. all 5 targets, 48 kHz (Apache-2.0)

The only challenger that is a plain PyPI install (`voxcpm==2.0.3`) — no git
clone. 2B params, ~8 GB VRAM, language auto-detected from the text, published
speaker-similarity scores top-tier across our targets (see the model README's
Minimax-MLS SIM table). Ukrainian is NOT among its 30 languages, so:
- **clone from the ref audio alone** (`reference_wav_path`) — the adapter's
  only mode; the transcript-based "ultimate cloning" (`prompt_wav_path` +
  `prompt_text`) is unusable with a Ukrainian reference.
- style/emotion per segment: `tts.instruct_text` becomes the `"(...)"` text
  prefix VoxCPM2 parses (e.g. "warm, unhurried teaching tone").

Install — MUST carry the torch pins on the same command line, or its resolver
moves torch 2.6.0 -> 2.13 and pulls torchcodec 0.14 (needs torchaudio>=2.9);
on our stack torchcodec must be 0.2.1 (the torch-2.6-era build):

```bash
pip install voxcpm==2.0.3 torchcodec==0.2.1 torch==2.6.0 torchaudio==2.6.0 "numpy>=1.24,<2"
pip check    # chatterbox pins must be untouched; if it complains, separate venv
```

Config (`config.gpu.yaml` → `tts`): `voxcpm_model_dir` (HF id `openbmb/VoxCPM2`
auto-downloads, or a local dir), `voxcpm_cfg_value` (default 2.0),
`voxcpm_timesteps` (default 10 — more = slower/better).

API used: `VoxCPM.from_pretrained(dir, load_denoiser=False).generate(text=,
reference_wav_path=, cfg_value=, inference_timesteps=)` → numpy wav at
`model.tts_model.sample_rate` (48 kHz; s6 handles mixed rates). No seed is
passed so best_of takes vary. Verify on first pod run; adjust `_synth_voxcpm`
if the 2.0.x API differs.

Validated version: `voxcpm==2.0.3` (resolver-checked vs torch 2.6.0 2026-07-13;
runtime validation pending first GPU run)

## Qwen3-TTS — 10 languages incl. all 5 targets, 3 s clone (Apache-2.0)

Alibaba's open-weights TTS (0.6B/1.7B), ~4 GB VRAM, released 2026-01-22.
git-clone install like CosyVoice/IndexTTS. Covers en/de/fr/es/ru (+ zh/ja/ko/
pt/it). Ukrainian is not a supported language, so:
- **x_vector_only_mode** (`tts.qwen_x_vector_only: true`, default) clones from
  the reference's speaker embedding ALONE — no ref transcript, the UA-ref path.
  Full clone (ref audio + `reference_text`) is opt-in for a target-lang ref.
- the reusable clone prompt is built ONCE per reference and cached in the
  adapter (a full video is hundreds of segments off one ref).

```bash
git clone https://github.com/QwenLM/Qwen3-TTS third_party/Qwen3-TTS
cd third_party/Qwen3-TTS && git checkout <PIN_COMMIT>
pip check                         # must not disturb torch 2.6.0
pip install -e .
# flash-attn is CUDA-only and a heavy build — install it separately and set
# tts.qwen_flash_attn: true only if you want it; the model runs without it:
#   pip install -U flash-attn --no-build-isolation
```

Config (`config.gpu.yaml` → `tts`): `qwen_model_dir` (HF id
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` auto-downloads, or `-0.6B-Base` / a local dir),
`qwen_x_vector_only` (default true), `qwen_flash_attn` (default false).

API used: `Qwen3TTSModel.from_pretrained(model_id, device_map=, dtype=)`,
`create_voice_clone_prompt(ref_audio=, x_vector_only_mode=[, ref_text=])`,
`generate_voice_clone(text=, language=, voice_clone_prompt=)` → `(wavs, sr)`,
`wavs[0]` numpy. Verify on the pinned commit; adjust `_synth_qwen` if the API
differs.

Pinned commit: `__________`  (fill after validation)

## Running the bake-off

Locally (engines installed by hand):
```bash
# translations must exist first (s3). On the pod, with the engine(s) installed:
dubadabidu bakeoff input/sketch60/sketch60.mp4 --langs en --overlay config.gpu.yaml
```

Remotely (M2 installs the challengers automatically): put the PINNED install
command for each engine in `runpod.engine_setup` (config.gpu.yaml). `dubadabidu
remote bakeoff <video>` runs it on the pod before the comparison. Each snippet
must make the engine importable in the venv — either `pip install -e` the cloned
repo or drop a `.pth` file into site-packages (so no PYTHONPATH is needed at run
time). An engine whose snippet is empty or fails is reported unavailable and
skipped, so a partial bake-off still runs.

Writes `work/<video>/bakeoff/bakeoff_<lang>.md` (scorecard + ADOPT/keep verdict
vs chatterbox) and `bakeoff_<lang>.html` (every engine side by side + your real
UA slice, for the ear test). An engine that isn't installed is reported as
unavailable and skipped, so a partial comparison still runs. Adopt a winner by
setting `tts.engine_by_lang`.

## License hygiene (unchanged)

CosyVoice, IndexTTS-2, VoxCPM2 and Qwen3-TTS are Apache-2.0 (commercial OK).
XTTS-v2 (CPML) and Fish-Speech weights (CC-BY-NC) stay OUT. Verify any new
candidate's license before first use.
