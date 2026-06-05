"""
Benchmark AROD student motion rollout against the frozen DyStream teacher.

This script intentionally measures motion generation only. Rendering and audio
muxing are excluded so the reported speedup isolates the AROD replacement for
the teacher's AR+FM motion rollout.
"""

import argparse
import json
import os
import time

import librosa
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from train_distill import load_teacher
from train_blockwise_distill import (
    blockwise_fm_rollout,
    blockwise_rollout,
    build_blockwise_student,
    extract_audio_features,
    teacher_rollout,
)
from verify_blockwise_distill import load_checkpoint_config, resolve_checkpoint
from verify_distill import get_reference


DEFAULT_CONFIG = "configs/distill/blockwise_stream_distill_cross_fm_mixed_trainval_teacher_gt.yaml"
DEFAULT_IMAGE = "img_files/person1.png"
DEFAULT_AUDIO = "wav_files/test_audio_60s.wav"


def prepare_inputs(audio, ref_npz, ref_image, cfg, device):
    audio_t = torch.from_numpy(audio).float().to(device).unsqueeze(0)
    inpaint_len = int(cfg.model.cbh_window_length) - 2
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    audio_padded = F.pad(audio_t, (inpaint_len * hop, 0))
    audio_other = torch.zeros_like(audio_padded)

    total_frames = audio_padded.shape[1] // hop
    motion_np, _ = get_reference(ref_npz, ref_image)
    motion_latent = torch.from_numpy(motion_np).float().to(device)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.view(1, 1, -1)
    elif motion_latent.dim() == 2:
        motion_latent = motion_latent[:1].unsqueeze(0)

    anchor = motion_latent[:, 0:1, :]
    motion_in = anchor.repeat(1, total_frames, 1)
    target_frames = max(total_frames - inpaint_len, 1)
    return audio_padded, audio_other, motion_in, anchor, inpaint_len, target_frames


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn):
    cuda_sync()
    start = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark AROD vs DyStream teacher motion rollout")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--img-path", default=DEFAULT_IMAGE)
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a meaningful speed benchmark")
    device = torch.device("cuda")

    fallback_cfg = OmegaConf.load(args.config)
    checkpoint_path = resolve_checkpoint(fallback_cfg)
    cfg, checkpoint = load_checkpoint_config(checkpoint_path, fallback_cfg)
    os.environ["DYSTREAM_WAV2VEC_PATH"] = cfg.wav2vec_path

    teacher = load_teacher(cfg).to(device).eval()
    student = build_blockwise_student(cfg).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()
    for param in student.parameters():
        param.requires_grad = False

    audio, _ = librosa.load(args.audio_path, sr=cfg.model.audio_sr)
    stem, _ = os.path.splitext(args.img_path)
    ref_npz = stem + ".npz"
    audio_padded, audio_other, motion_in, anchor, inpaint_len, target_frames = prepare_inputs(
        audio,
        ref_npz,
        args.img_path,
        cfg,
        device,
    )

    with torch.inference_mode():
        (feat_self, feat_other), audio_feature_time = timed_call(
            lambda: extract_audio_features(teacher, audio_padded, audio_other)
        )
        teacher_seq, teacher_time = timed_call(
            lambda: teacher_rollout(teacher, audio_padded, audio_other, motion_in, anchor, cfg, int(cfg.seed))[
                :, inpaint_len:
            ]
        )
        if cfg.student.get("architecture", "additive") == "cross_fm":
            scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler)
            def run_student_fm():
                return blockwise_fm_rollout(
                    student,
                    feat_self,
                    feat_other,
                    anchor,
                    target_frames=teacher_seq.shape[1],
                    inpaint_len=inpaint_len,
                    cfg=cfg,
                    noise_scheduler=scheduler,
                    teacher_seq=None,
                    seed=int(cfg.seed),
                )

            student_seq, student_time = timed_call(run_student_fm)
        else:
            def run_student_ar():
                return blockwise_rollout(
                    student,
                    feat_self,
                    feat_other,
                    anchor,
                    target_frames=teacher_seq.shape[1],
                    inpaint_len=inpaint_len,
                    cfg=cfg,
                )

            student_seq, student_time = timed_call(run_student_ar)

    audio_duration = len(audio) / float(cfg.model.audio_sr)
    min_len = min(student_seq.shape[1], teacher_seq.shape[1])
    metrics = {
        "config": args.config,
        "checkpoint": checkpoint_path,
        "image_path": args.img_path,
        "audio_path": args.audio_path,
        "audio_duration_sec": float(audio_duration),
        "frames": int(min_len),
        "audio_feature_time_sec": float(audio_feature_time),
        "teacher_motion_time_sec": float(teacher_time),
        "student_motion_time_sec": float(student_time),
        "teacher_motion_rtf": float(teacher_time / max(audio_duration, 1.0e-6)),
        "student_motion_rtf": float(student_time / max(audio_duration, 1.0e-6)),
        "motion_speedup": float(teacher_time / max(student_time, 1.0e-6)),
        "mse": float(F.mse_loss(student_seq[:, :min_len], teacher_seq[:, :min_len]).item()),
    }

    output = args.output or os.path.join("outputs", "arod_speed_benchmark.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
