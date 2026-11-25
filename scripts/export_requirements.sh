#!/usr/bin/env bash
# Regenerate requirements.txt from uv.lock. Never edit requirements.txt by hand:
# tests/test_deps.py fails if it is not exactly this export.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "# GENERATED from uv.lock by scripts/export_requirements.sh — do not edit."
  echo "# Preferred install: uv sync --extra dev   (add --extra sep,mac on the Mac)"
  echo "# torch on Linux comes from the CUDA 13.0 index:"
  echo "--extra-index-url https://download.pytorch.org/whl/cu130"
  uv export --frozen --no-hashes --no-emit-project --extra dev
} > requirements.txt
