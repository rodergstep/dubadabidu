# Prebuilt pod image (PLAN_v0.5 P3). Bakes what REMOTE_SETUP + engine_setup
# used to install on EVERY pod — ~11 min of billed bootstrap per run
# (runs.jsonl: 694 s of a 724 s run). The pod still runs REMOTE_SETUP, which
# links this venv in when /opt/venv/.uv-lock-sha256 matches the project's
# uv.lock and then becomes a near no-op; on any mismatch it builds fresh as
# before, so a stale image costs time, never correctness.
#
# Build: .github/workflows/image.yml (tag = lock-<first 12 of sha256(uv.lock)>)
# Local: docker buildx build --platform linux/amd64 -f docker/Dockerfile .

# Same base as config.gpu.yaml runpod.image, so sshd/ports behave as validated.
FROM runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404

ARG UV_VERSION=0.8.15
# keep in sync with runpod.engine_setup.qwen in config.gpu.yaml
ARG QWEN_SHA=022e286b98fbec7e1e916cb940cdf532cd9f488e
ARG FASTER_QWEN=0.3.2
ARG QWEN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_NO_CACHE_DIR=1

# --no-install-recommends is load-bearing: a full ffmpeg install once upgraded
# the C runtime and killed sshd mid-run (FINDINGS §4).
RUN apt-get update \
 && apt-get install -y -q --no-install-recommends rsync ffmpeg git python3-venv \
 && rm -rf /var/lib/apt/lists/*

# 1. the locked environment (torch 2.14.1+cu126 and everything else)
WORKDIR /opt/build
COPY pyproject.toml uv.lock ./
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install -q "uv==${UV_VERSION}" \
 && UV_PROJECT_ENVIRONMENT=/opt/venv /opt/venv/bin/uv sync --frozen --inexact \
        --extra dev --no-install-project --python /opt/venv/bin/python

# 2. the engine, at the SHA engine_setup pins (qwen-tts declares no
#    install_requires, hence the explicit transformers check below)
RUN git clone https://github.com/QwenLM/Qwen3-TTS /opt/third_party/Qwen3-TTS \
 && git -C /opt/third_party/Qwen3-TTS checkout "${QWEN_SHA}" \
 && /opt/venv/bin/pip install -q 'setuptools<80' wheel \
 && /opt/venv/bin/pip install -q -e /opt/third_party/Qwen3-TTS \
 && /opt/venv/bin/pip install -q "faster-qwen3-tts==${FASTER_QWEN}" \
 && /opt/venv/bin/python -c "import qwen_tts, faster_qwen3_tts, transformers; \
assert transformers.__version__ == '4.57.3', transformers.__version__"

# 3. weights into root's DEFAULT HF cache — not a custom HF_HOME, because SSH
#    sessions do not inherit Docker ENV, and the pod runs everything over SSH.
RUN /opt/venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
print(d('${QWEN_MODEL}'))"

# 4. the stamp REMOTE_SETUP compares against the project's uv.lock
RUN sha256sum uv.lock | cut -c1-64 > /opt/venv/.uv-lock-sha256 \
 && echo "${QWEN_SHA}" > /opt/venv/.qwen-sha \
 && rm -rf /opt/build /root/.cache/uv /root/.cache/pip
WORKDIR /
