#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DYSTREAM_ASSET_REPO="${DYSTREAM_ASSET_REPO:-robinwitch/DyStream}"
AROD_ASSET_REPO="${AROD_ASSET_REPO:-}"
AROD_ASSET_FILE="${AROD_ASSET_FILE:-blockwise_latest.pt}"

command -v huggingface-cli >/dev/null 2>&1 || {
  echo "huggingface-cli is not installed. Run: pip install huggingface-hub" >&2
  exit 1
}

mkdir -p checkpoints pretrained_model tools/pretrained_model

echo "[1/4] Downloading Wav2Vec2..."
huggingface-cli download facebook/wav2vec2-base-960h \
  --local-dir pretrained_model/wav2vec2-base-960h

echo "[2/4] Downloading DyStream assets from ${DYSTREAM_ASSET_REPO}..."
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
huggingface-cli download "$DYSTREAM_ASSET_REPO" --local-dir "$tmp_dir/dystream"

if [[ -f "$tmp_dir/dystream/checkpoints/last.ckpt" ]]; then
  cp "$tmp_dir/dystream/checkpoints/last.ckpt" checkpoints/last.ckpt
else
  echo "Missing teacher checkpoint: checkpoints/last.ckpt in ${DYSTREAM_ASSET_REPO}" >&2
fi

if [[ -f "$tmp_dir/dystream/tools/pretrained_model/epoch=0-step=312000.ckpt" ]]; then
  cp "$tmp_dir/dystream/tools/pretrained_model/epoch=0-step=312000.ckpt" \
    tools/pretrained_model/epoch=0-step=312000.ckpt
else
  echo "Missing renderer checkpoint: tools/pretrained_model/epoch=0-step=312000.ckpt in ${DYSTREAM_ASSET_REPO}" >&2
fi

echo "[3/4] Checking optional AROD checkpoint..."
mkdir -p outputs/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt
if [[ -n "$AROD_ASSET_REPO" ]]; then
  huggingface-cli download "$AROD_ASSET_REPO" "$AROD_ASSET_FILE" \
    --local-dir outputs/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt
else
  cat <<'EOF'
AROD_ASSET_REPO is not set, so the AROD checkpoint was not downloaded.
Place the student checkpoint manually at:
  outputs/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt/blockwise_latest.pt

If your checkpoint is hosted on Hugging Face, rerun with:
  AROD_ASSET_REPO=<owner/repo> AROD_ASSET_FILE=blockwise_latest.pt bash scripts/download_assets.sh
EOF
fi

echo "[4/4] Asset layout:"
find checkpoints pretrained_model tools/pretrained_model outputs/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt \
  -maxdepth 2 -type f | sort
