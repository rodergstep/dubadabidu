# Third-party TTS engines (bake-off candidates)

Chatterbox (the incumbent) and edge install from PyPI into the main venv. The
bake-off challengers are **git-clone installs, GPU-only**, with their own heavy
dependencies — and each installs into its **own venv: `venvs/<engine>`**. The
pipeline routes any engine with a `venvs/<engine>` dir through a persistent
worker subprocess in that venv automatically (`pipeline/engine_client.py` /
`engine_worker.py`), so a challenger's resolver can pick whatever torch it
wants and the chatterbox pin (torch/torchaudio 2.6.0) is untouchable by
construction. On the pod, `remote bakeoff`/`remote setup-check` create these
venvs from the `runpod.engine_setup` snippets; locally, create one by hand
(below) and the routing kicks in the moment the venv exists. The engine venv
needs NOTHING of the project installed — the worker runs `-m
pipeline.engine_worker` from the project root — only the engine's own deps
(plus `soundfile`, which the automated install adds).

Pin the commit you validate on (record it below) — these repos move fast and
their inference APIs drift; `pipeline/tts_engine.py` targets the APIs noted here.

**PENDING — do this immediately after the first `dubadabidu remote setup-check`.**
Three of the four snippets in `config.gpu.yaml` currently track HEAD
(cosyvoice, indextts, qwen — only `voxcpm==2.0.3` is pinned), so the
`Pinned commit:` lines below are still blank. setup-check reports which engines
import; capture the revision each one resolved to and record it here, then
append `&& git -C third_party/<repo> checkout <SHA>` to that engine's snippet
(the `# pin:` line in config.gpu.yaml shows the exact edit).

This is not cosmetic. Until it is done, a bake-off verdict is **not
reproducible**: a re-run weeks later may clone different code, and the numbers
would move for reasons the scorecard cannot show. The whole protocol rests on
"the eval harness decides" — an unpinned harness decides against a moving
target. Cost of doing it: zero. IMPROVEMENT_PLAN.md lists it under Standing
risks.

## CosyVoice 2/3 — all 5 target languages + cross-lingual (Apache-2.0)

```bash
# from the project root — its OWN venv; the pipeline auto-routes through it
python3 -m venv venvs/cosyvoice && . venvs/cosyvoice/bin/activate
pip install soundfile
git clone --recursive https://github.com/FunAudioLLM/CosyVoice third_party/CosyVoice
git -C third_party/CosyVoice checkout <PIN_COMMIT>
git -C third_party/CosyVoice submodule update --init --recursive
pip install -r third_party/CosyVoice/requirements.txt
# model weights (pick one):
#   modelscope download iic/CosyVoice2-0.5B --local_dir pretrained_models/CosyVoice2-0.5B
# make the clone importable in THIS venv with no PYTHONPATH (a .pth file —
# the worker spawns in a fresh process, so an exported PYTHONPATH won't reach it):
python -c 'import sysconfig,os;p=sysconfig.get_paths()["purelib"];r=os.getcwd();open(os.path.join(p,"cosyvoice.pth"),"w").write(r+"/third_party/CosyVoice\n"+r+"/third_party/CosyVoice/third_party/Matcha-TTS\n")'
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

### STATUS 2026-07-30: installs, does NOT yet synthesize. Seven causes found.
The `engine_setup` snippet in config.gpu.yaml now gets as far as
`from cosyvoice.cli.cosyvoice import CosyVoice2` succeeding on a pod. It has
never produced audio. Start from these, do not rediscover them:

1. `openai-whisper==20231117` — its setup.py imports `pkg_resources`, which
   modern setuptools no longer ships. It IS required (cosyvoice imports `whisper`
   at module load), so it cannot simply be dropped.
2. `PIP_CONSTRAINT` does NOT fix (1): pip's build-isolation overlay ignores it.
3. `--no-build-isolation` fixes (1) but breaks `tensorrt-cu12`, whose
   `wheel_stub` PEP 517 backend exists ONLY inside the isolation overlay.
   -> steps need OPPOSITE isolation settings; split the install.
4. `tensorrt-cu12` x3 must be filtered out (TensorRT inference is unused).
5. Do NOT over-filter. `gdown` is imported by `matcha.utils`; dropping it gives
   `ErrorDuringImport: problem in matcha.utils`. Judge deps by what is IMPORTED,
   not by what they sound like. Only tensorrt and deepspeed are proven safe drops.
6. The loader class must match the WEIGHTS, not the newest class the checkout
   exposes: a CosyVoice2-0.5B dir with the CosyVoice3 class gives
   `ValueError: .../cosyvoice3.yaml not found!`. Fixed in `_load_cosyvoice`,
   which now probes for `cosyvoice{3,2}.yaml`.
7. NEXT UNKNOWN: after (5) is restored the import path is untested — the pod's
   venv and clone were deleted by a `free_engine` bug before it could be retried.

Cost so far: ~7 pod attempts. voxcpm covers the same five languages, is faster,
and was chosen by ear — so cosyvoice is a nice-to-have, not a blocker.

## IndexTTS-2 — EN + Mandarin only, dubbing-native (Apache-2.0)

```bash
# from the project root — its OWN venv; the pipeline auto-routes through it
python3 -m venv venvs/indextts && . venvs/indextts/bin/activate
pip install soundfile
git clone https://github.com/index-tts/index-tts third_party/index-tts
git -C third_party/index-tts checkout <PIN_COMMIT>
pip install -e third_party/index-tts
# download checkpoints into ./checkpoints (config.yaml + weights) per the repo
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

Install — in its OWN venv its resolver is free to pick its preferred
torch/torchcodec (the old same-line pins existed only to defend the shared
venv's torch 2.6.0):

```bash
# from the project root — its OWN venv; the pipeline auto-routes through it
python3 -m venv venvs/voxcpm && . venvs/voxcpm/bin/activate
pip install soundfile voxcpm==2.0.3
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
# from the project root — its OWN venv; the pipeline auto-routes through it
python3 -m venv venvs/qwen && . venvs/qwen/bin/activate
pip install soundfile
git clone https://github.com/QwenLM/Qwen3-TTS third_party/Qwen3-TTS
git -C third_party/Qwen3-TTS checkout <PIN_COMMIT>
pip install -e third_party/Qwen3-TTS
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

Locally (engine venvs created by hand as above):
```bash
# translations must exist first (s3). With venvs/<engine> present the pipeline
# routes each challenger through its own venv automatically:
dubadabidu bakeoff input/sketch60/sketch60.mp4 --langs en --overlay config.gpu.yaml
```

Remotely (M2 installs the challengers automatically): put the PINNED install
command for each engine in `runpod.engine_setup` (config.gpu.yaml). `dubadabidu
remote bakeoff <video>` creates `venvs/<engine>` on the pod and runs the snippet
inside it before the comparison. Each snippet must make the engine importable
in ITS venv — either `pip install -e` the cloned repo or drop a `.pth` file
into site-packages (the worker spawns with no PYTHONPATH). An engine whose
snippet is empty or fails is reported unavailable and skipped, so a partial
bake-off still runs — and a failed install can no longer damage any other
engine's environment.

Writes `work/<video>/bakeoff/bakeoff_<lang>.md` (scorecard + ADOPT/keep verdict
vs chatterbox) and `bakeoff_<lang>.html` (every engine side by side + your real
UA slice, for the ear test). An engine that isn't installed is reported as
unavailable and skipped, so a partial comparison still runs. Adopt a winner by
setting `tts.engine_by_lang`.

## License hygiene (unchanged)

CosyVoice, IndexTTS-2, VoxCPM2 and Qwen3-TTS are Apache-2.0 (commercial OK).
XTTS-v2 (CPML) and Fish-Speech weights (CC-BY-NC) stay OUT. Verify any new
candidate's license before first use.
