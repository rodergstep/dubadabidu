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

CosyVoice and IndexTTS-2 are Apache-2.0 (commercial OK). XTTS-v2 (CPML) and
Fish-Speech weights (CC-BY-NC) stay OUT. Verify any new candidate's license
before first use.
