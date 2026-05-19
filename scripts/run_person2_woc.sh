#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/env_local.sh"

AUDIO_SRC="/mnt/pfs/group-jt/changxun.pan/runs/test/huaqiang.mov"
AUDIO_WAV="wav_files/huaqiang.wav"
CONFIG="configs/motion_gen/local_person2_woc.yaml"

if [[ ! -f "$AUDIO_SRC" ]]; then
  echo "Missing audio source: $AUDIO_SRC" >&2
  exit 1
fi

if [[ ! -f "img_files/person2.png" ]]; then
  cp /mnt/pfs/group-jt/changxun.pan/runs/test/person2.png img_files/person2.png
fi

ffmpeg -y -i "$AUDIO_SRC" -vn -ac 1 -ar 16000 "$AUDIO_WAV"

.venv/bin/python main.py \
  --config "$CONFIG" \
  --override \
  exp_name=person2_woc \
  debug=True \
  model.module_name=model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder \
  resume_ckpt=checkpoints/last.ckpt
