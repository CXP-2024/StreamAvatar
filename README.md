# StreamAvatar / AROD

StreamAvatar is our DyStream-based audio-driven portrait animation project. The original DyStream teacher is kept as the frozen quality reference and feature provider, while our main model is **AROD**: an **Autoregressive One-step Denoising** student for fast streaming audio-to-motion prediction.

AROD keeps the autoregressive rollout structure over motion blocks, but replaces the teacher's sequential AR+FM motion generation with one forward pass per short future block. The generated motion latents are rendered by the frozen DyStream/LIA portrait renderer.

## What This Repository Contains

- `app.py`: Gradio demo for the AROD student. It takes a reference face image and driving audio, predicts motion with AROD, then renders a talking-head video.
- `train_blockwise_distill.py`: AROD/blockwise student architectures, rollout helpers, and training loop.
- `verify_blockwise_distill.py`: End-to-end teacher/student comparison. It measures teacher and student motion rollout time, renders both videos, and writes side-by-side comparisons.
- `benchmark_arod_speed.py`: Motion-only speed benchmark. It excludes rendering and muxing so the reported speedup isolates the AROD replacement for teacher motion rollout.
- `configs/distill/`: student distillation configs. The current demo default is `blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt.yaml`.
- `report/`: final report source/PDF and figures.
- `proposal/`: reconstructed project proposal source/PDF.

Large checkpoints, cached data, rendered outputs, and local build environments are intentionally kept out of git.

## Environment

Use the existing local virtual environment when available:

```bash
cd /mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream
source .venv/bin/activate
```

If you need to rebuild the environment, install from `requirements.txt` in a local environment under this mounted workspace, not under `/root`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project expects CUDA and the local pretrained assets/checkpoints already present in this workspace, including:

- `checkpoints/last.ckpt`
- `pretrained_model/wav2vec2-base-960h`
- `tools/visualization_0416/`
- AROD checkpoint under `outputs/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt/`

## Run the AROD App

```bash
cd /mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream
source .venv/bin/activate
python -u app.py
```

The app launches on port `7860` by default. The default sample uses:

- Reference image: `img_files/person1.png`
- Audio: `wav_files/test_audio_60s.wav`
- AROD config: `configs/distill/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt.yaml`
- Student checkpoint: `outputs/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt/blockwise_latest.pt`

## Verify Teacher vs AROD

Run the full verification suite:

```bash
source .venv/bin/activate
python verify_blockwise_distill.py \
  --img-path img_files/person1.png \
  --audio-path wav_files/test_audio_60s.wav \
  --train-sample-idx 0
```

This writes metrics and videos to:

```text
outputs/verify_blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt_suite/
```

The important output fields are:

- `teacher_time_sec`: frozen DyStream teacher motion rollout time.
- `student_time_sec`: AROD student motion rollout time.
- `speedup`: teacher motion time divided by student motion time.
- `comparison_video`: side-by-side rendered teacher/student video.

Fresh verification on the current B20Z node for the 60s `person1 + test_audio_60s.wav` case produced:

```text
teacher_time_sec: 29.384
student_time_sec: 3.752
speedup: 7.83x
```

Earlier verification of the same config/output suite recorded `10.24x` on the 60s case. In the report we therefore describe the result as approximately `8x` in the latest fresh run and capable of reaching about `10x` under the same verification setup, rather than claiming a fixed deterministic 10x on every run.

## Motion-Only Speed Benchmark

For a faster timing-only check without rendering videos:

```bash
source .venv/bin/activate
python benchmark_arod_speed.py \
  --config configs/distill/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt.yaml \
  --img-path img_files/person1.png \
  --audio-path wav_files/test_audio_60s.wav \
  --output outputs/arod_speed_benchmark.json
```

Fresh motion-only timing on the same node produced:

```text
teacher_motion_time_sec: 26.007
student_motion_time_sec: 3.253
motion_speedup: 8.00x
```

This benchmark excludes rendering and audio muxing. It measures only the part AROD replaces: DyStream teacher motion rollout.

## Build the Report

```bash
./report/build.sh
```

Output:

```text
report/StreamAvatar_report.pdf
```

## Build the Proposal

```bash
./proposal/build.sh
```

Output:

```text
proposal/StreamAvatar_proposal.pdf
```

## Current Interpretation

AROD is not a conventional flow model and not a standard GPT-style token-by-token generator. It is a blockwise autoregressive denoising model:

```text
audio + motion history + anchor + noisy future tokens -> clean future motion block
```

The student preserves the teacher's rollout structure at the block level, but removes the expensive per-frame AR+FM teacher generation loop. This gives a practical speed-quality trade-off for real-time-capable portrait animation.
