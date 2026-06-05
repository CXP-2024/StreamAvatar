#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DYSTREAM_ASSET_REPO="${DYSTREAM_ASSET_REPO:-robinwitch/DyStream}"
AROD_GDRIVE_ID="${AROD_GDRIVE_ID:-1El-2l5GZRfrVLEl2-x6ocPyyT9ILxDJS}"
AROD_GDOWN_PROXY="${AROD_GDOWN_PROXY:-}"
AROD_OUTPUT_DIR="outputs/blockwise_stream_distill_cross_fm_teacher_cache_anchor_pretrain_60k"
AROD_SHA256="${AROD_SHA256:-01893fabb842fcc8e9817a8e2530108d75932aad4f6ac4136e5c22b94702e860}"

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
mkdir -p "$AROD_OUTPUT_DIR"
if command -v gdown >/dev/null 2>&1; then
  gdown_args=()
  if [[ -n "$AROD_GDOWN_PROXY" ]]; then
    gdown_args+=(--proxy "$AROD_GDOWN_PROXY")
  fi
  gdown "${gdown_args[@]}" "$AROD_GDRIVE_ID" \
    -O "$AROD_OUTPUT_DIR/blockwise_latest.pt"
else
  cat <<'EOF'
gdown is not available, so the AROD checkpoint was not downloaded.
Place the student checkpoint manually at:
  outputs/blockwise_stream_distill_cross_fm_teacher_cache_anchor_pretrain_60k/blockwise_latest.pt

Install gdown and rerun this script:
  pip install gdown
EOF
fi

if [[ -f "$AROD_OUTPUT_DIR/blockwise_latest.pt" ]]; then
  echo "Verifying AROD checkpoint SHA256..."
  actual_sha="$(sha256sum "$AROD_OUTPUT_DIR/blockwise_latest.pt" | awk '{print $1}')"
  if [[ "$actual_sha" != "$AROD_SHA256" ]]; then
    echo "AROD checkpoint checksum mismatch." >&2
    echo "Expected: $AROD_SHA256" >&2
    echo "Actual:   $actual_sha" >&2
    exit 1
  fi
fi

echo "[4/4] Asset layout:"
find checkpoints pretrained_model tools/pretrained_model "$AROD_OUTPUT_DIR" \
  -maxdepth 2 -type f | sort
