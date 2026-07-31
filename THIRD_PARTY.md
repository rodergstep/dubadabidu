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

**PINNED 2026-07-31.** `config.gpu.yaml` checks out qwen at a fixed SHA and
installs `faster-qwen3-tts==0.3.2`, so a bake-off verdict is now reproducible.
Until this was done a re-run weeks later could clone different code and the
numbers would move for reasons the scorecard cannot show — the protocol rests on
"the eval harness decides", and an unpinned harness decides against a moving
target.

## Removed engines — cosyvoice, voxcpm, indextts (cut 2026-07-31)

Both adapters, install recipes, config blocks and tests were deleted. Git
history has everything, including CosyVoice's full seven-cause failure log and
the validated-but-unused install recipe: `git show <this commit>^:THIRD_PARTY.md`.

- **CosyVoice 2/3** — installed and imported cleanly after six attempts at the
  build recipe, then never produced a single take in seven runs. Removed because
  three working engines existed and it had consumed more pod time than all of
  them combined.
- **IndexTTS-2** — the best measured mos on the roster (2.739 with per-segment
  emotion transfer, vs qwen+fast's 2.112). Removed anyway: en/zh ONLY, so it
  could never serve four of the five targets, and ~2x qwen's cost with no cheap
  fix — [index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm) reports no
  acceleration for IndexTTS-2's GPT and is a vLLM server rather than a library,
  and [Faster IndexTTS-2](https://arxiv.org/html/2607.21042v1) (TensorRT-LLM) is
  a paper, not a package. Its `emotion_from_source` machinery
  (`with_source_emotion`, the `emo/` slice cutter, the pod-side pre-cut) went
  with it — that was IndexTTS-2-only. CAVEAT: the measured "+0.346 mos from
  emotion transfer" was never actually verified, because `seg/indextts/` and
  `seg/indextts+emo/` turned out to hold identical files. Emotion transfer may
  have been worth more, or nothing; we shipped without finding out.
- **VoxCPM2** — worked well and was the ear's pick on 2026-07-30 (sim→real 0.804,
  the best of any engine). Removed because qwen+fast beat it on both speed
  (2.36 vs 2.92 s/take) and cost once qwen's kernel-launch bottleneck was fixed.
  CAVEAT worth recording: voxcpm's similarity lead (0.804 vs qwen's 0.614) was
  the one gap wider than the noise floor, and it was never settled by ear against
  qwen+fast before the cut. If cloned-voice identity later looks weak, this
  removal is the first thing to revisit.

## Qwen3-TTS — 10 languages incl. all 5 targets, 3 s clone (Apache-2.0)

Alibaba's open-weights TTS (0.6B/1.7B), ~4 GB VRAM, released 2026-01-22.
git-clone install. Covers en/de/fr/es/ru (+ zh/ja/ko/
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
git -C third_party/Qwen3-TTS checkout 022e286b98fbec7e1e916cb940cdf532cd9f488e
pip install -e third_party/Qwen3-TTS
pip install faster-qwen3-tts==0.3.2      # the CUDA-graph path (tts.qwen_fast)
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

Pinned commit: **`022e286b98fbec7e1e916cb940cdf532cd9f488e`** (2026-03-17, HEAD of
`main` at validation — upstream had not moved in four months).
faster-qwen3-tts: **`0.3.2`**.

### SPEED: qwen is kernel-launch bound (`tts.qwen_fast`)

Measured on a 4090, 2026-07-31, after fixing a CPU-only torch in `venvs/qwen`
(126 -> 14.0 s/take). The remaining cost is NOT the model:

| axis | wall/audio | conclusion |
|---|---|---|
| bf16 / fp16 / fp32 | 1.42 / 1.35 / 1.39 | dtype irrelevant |
| sdpa vs eager | 1.42 / 1.57 | attention impl ~10% |
| **1.7B vs 0.6B** | **1.42 / 1.38** | **3x smaller = no speedup** |

Model size not mattering rules out compute. Upstream confirms why: each decode
step dispatches ~500 tiny GPU ops from a Python loop, so the card idles at
10-12% utilisation between kernel launches. flash-attn does NOT fix this.

[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (MIT)
wraps the same weights in CUDA Graphs + `StaticCache` + a vectorized repetition
penalty — no custom kernels. Upstream measures 4.1x on a 4090 (0.75 -> 0.18
wall/audio) and 7.1x on an H100.

**MEASURED HERE 2026-07-31: 14.01 -> 2.36 s/take, a 5.9x win** — better than
upstream's 4.1x, and qwen is now the fastest engine on the roster. 5-language
cost for a 1 h video drops $14.02 -> $2.36. Output is NOT bit-identical to the
stock decode loop (mos 1.942 -> 2.112, f0st 2.591 -> 2.874, sim flat at 0.614);
every shift sits inside qwen's take-to-take mos+- of 0.322 at n=4, so read it as
different takes, not better ones.

```bash
pip install faster-qwen3-tts     # installed alongside the stock clone so the
                                 # A/B needs no reinstall
```
`tts.qwen_fast: true` selects it. API differences handled in `_load_qwen` /
`_qwen_clone_prompt`: `from_pretrained(model_id)` takes NO device/dtype kwargs,
`create_voice_clone_prompt` lives at `.model`, and `warmup(prefill_len=)`
captures the graphs up front (without it early calls pay capture cost lazily —
upstream reports 1.49 vs 0.73 wall/audio on the same setup unwarmed vs warmed).

STATUS: **validated, and the production default.** `qwen_fast` is salted into
`synth_hash` so an A/B re-synthesizes instead of scoring stock-decoded cache. Note that once dispatch overhead is gone the model
becomes genuinely compute-bound, which REOPENS 0.6B vs 1.7B (0.6B scored mos
2.697 vs 2.411 in the dtype probe — inside the noise floor, not worse).

Not adopted: [vLLM-Omni](https://vllm.ai/blog/2026-06-23-vllm-omni-tts) serves
Qwen3-TTS with continuous batching, which suits our throughput-bound workload
(~1200 independent generations per video-hour), but it is a server architecture
and would mean rewriting `engine_client`. Its Code2Wav stage also still
serializes per request upstream (vllm-omni#3163).

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

Qwen3-TTS is Apache-2.0; faster-qwen3-tts is MIT (commercial OK).
XTTS-v2 (CPML) and Fish-Speech weights (CC-BY-NC) stay OUT. Verify any new
candidate's license before first use.
