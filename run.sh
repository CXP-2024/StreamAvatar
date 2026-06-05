#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u app.py --host "${HOST:-0.0.0.0}" --port "${PORT:-7860}"
