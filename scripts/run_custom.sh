#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/env_local.sh"

# Edit these two paths for each run. AUDIO_SRC can be .mov, .wav, or any
# ffmpeg-readable media file with an audio stream.
IMAGE_SRC="${IMAGE_SRC:-/mnt/pfs/group-jt/changxun.pan/runs/test/person2.png}"
AUDIO_SRC="${AUDIO_SRC:-/mnt/pfs/group-jt/changxun.pan/runs/test/huaqiang.mov}"

# Keep this stable to overwrite the generated JSON/YAML each run.
RUN_ID="${RUN_ID:-custom_current}"

# Set REFRESH_LATENT=0 to reuse the previous image latent when IMAGE_SRC is unchanged.
REFRESH_LATENT="${REFRESH_LATENT:-1}"

BASE_CONFIG="${BASE_CONFIG:-configs/motion_gen/local_person1_woc.yaml}"
GENERATED_JSON="data_json/${RUN_ID}.json"
GENERATED_CONFIG="configs/motion_gen/${RUN_ID}.yaml"
IMAGE_DST="img_files/${RUN_ID}.png"
AUDIO_WAV="wav_files/${RUN_ID}.wav"
MOTION_NPZ="img_files/${RUN_ID}.npz"

if [[ ! -f "$IMAGE_SRC" ]]; then
  echo "Missing image: $IMAGE_SRC" >&2
  exit 1
fi

if [[ ! -f "$AUDIO_SRC" ]]; then
  echo "Missing audio/media file: $AUDIO_SRC" >&2
  exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing base config: $BASE_CONFIG" >&2
  exit 1
fi

mkdir -p img_files wav_files data_json configs/motion_gen outputs

cp "$IMAGE_SRC" "$IMAGE_DST"
ffmpeg -y -i "$AUDIO_SRC" -vn -ac 1 -ar 16000 "$AUDIO_WAV"

if [[ "$REFRESH_LATENT" == "1" ]]; then
  rm -f "$MOTION_NPZ"
  rm -f "img_files/${RUN_ID}_masked.png"
  rm -f "img_files/${RUN_ID}_resize.png"
fi

.venv/bin/python - <<PY
import json
from pathlib import Path
from omegaconf import OmegaConf

run_id = "${RUN_ID}"
generated_json = Path("${GENERATED_JSON}")
generated_config = Path("${GENERATED_CONFIG}")
base_config = Path("${BASE_CONFIG}")

item = {
    "origin_video_path": None,
    "resampled_video_path": "${IMAGE_DST}",
    "audio_path": "${AUDIO_WAV}",
    "audio_self_path": "${AUDIO_WAV}",
    "audio_other_path": None,
    "motion_self_path": "${MOTION_NPZ}",
    "motion_other_path": None,
    "mode": "test_wild",
    "dataset_type": "dyadic",
    "video_id": run_id,
}

generated_json.write_text(json.dumps([item], indent=4) + "\\n", encoding="utf-8")

cfg = OmegaConf.load(base_config)
cfg.exp_name = run_id
cfg.debug = True
cfg.test = True
cfg.is_test = True
cfg.resume_ckpt = "checkpoints/last.ckpt"
cfg.data.meta_paths = [str(generated_json)]
cfg.data.val_meta_paths = [str(generated_json)]
cfg.data.test_meta_paths = [str(generated_json)]
cfg.data.num_workers = 1
cfg.model.module_name = "model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder"

OmegaConf.save(cfg, generated_config)
print(f"[run_custom] wrote {generated_json}")
print(f"[run_custom] wrote {generated_config}")
PY

.venv/bin/python main.py --config "$GENERATED_CONFIG"
