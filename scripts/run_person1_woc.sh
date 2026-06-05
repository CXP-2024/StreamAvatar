#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/env_local.sh"

AUDIO_SRC="/mnt/pfs/group-jt/changxun.pan/runs/test/woc.mov"
AUDIO_WAV="wav_files/woc.wav"
CONFIG="configs/motion_gen/local_person1_woc.yaml"

if [[ ! -f "$AUDIO_SRC" ]]; then
  echo "Missing audio source: $AUDIO_SRC" >&2
  exit 1
fi

if [[ ! -f "img_files/person1.png" ]]; then
  echo "Missing reference image: img_files/person1.png" >&2
  exit 1
fi

if [[ ! -f "img_files/person1.npz" ]]; then
  echo "Missing reference motion latent: img_files/person1.npz" >&2
  echo "Run one successful person1 inference first, or generate the latent with tools/visualization_0416/img_to_latent.py." >&2
  exit 1
fi

ffmpeg -y -i "$AUDIO_SRC" -vn -ac 1 -ar 16000 "$AUDIO_WAV"

.venv/bin/python main.py \
  --config "$CONFIG" \
  --override \
  exp_name=person1_woc \
  debug=True \
  model.module_name=model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder \
  resume_ckpt=checkpoints/last.ckpt
