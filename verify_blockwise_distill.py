"""
Verify block-wise streaming AR distillation.

This script compares the frozen DyStream teacher rollout with the block-wise
student rollout, reports latent errors/timing, and optionally renders a
side-by-side video.

Usage:
  .venv/bin/python verify_blockwise_distill.py \
    --checkpoint outputs/blockwise_stream_distill/blockwise_best.pt \
    --audio wav_files/woc.wav \
    --ref-image img_files/person1.png \
    --ref-npz img_files/person1.npz \
    --render
"""

import argparse
import json
import os
import sys
import time
import warnings

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"))

from train_distill import load_teacher
from train_blockwise_distill import (
    BlockARStudent,
    blockwise_rollout,
    extract_audio_features,
    teacher_rollout,
)
from verify_distill import get_reference, make_side_by_side, render_video


def load_blockwise_student(checkpoint_path, cfg, device):
    student = BlockARStudent(
        audio_dim=768,
        motion_dim=cfg.model.vae_codebook_size,
        hidden_dim=cfg.student.hidden_dim,
        block_frames=cfg.student.block_frames,
        history_frames=cfg.student.history_frames,
        layers=cfg.student.layers,
        heads=cfg.student.heads,
        dropout=cfg.student.dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()
    for param in student.parameters():
        param.requires_grad = False
    return student, checkpoint


def prepare_inputs(audio_path, ref_npz, ref_image, cfg, device):
    audio, _ = librosa.load(audio_path, sr=cfg.model.audio_sr)
    audio_t = torch.from_numpy(audio).float().to(device).unsqueeze(0)

    inpaint_len = cfg.model.cbh_window_length - 2
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    audio_padded = F.pad(audio_t, (inpaint_len * hop, 0))
    audio_other = torch.zeros_like(audio_padded)

    total_frames = audio_padded.shape[1] // hop
    motion_np, ref_img_resize = get_reference(ref_npz, ref_image)
    motion_latent = torch.from_numpy(motion_np).float().to(device)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.view(1, 1, -1)
    elif motion_latent.dim() == 2:
        motion_latent = motion_latent[:1].unsqueeze(0)
    anchor = motion_latent[:, 0:1, :]
    motion_in = anchor.repeat(1, total_frames, 1)
    return audio, audio_padded, audio_other, motion_in, anchor, inpaint_len, ref_img_resize


def latent_metrics(student_seq, teacher_seq, block_frames):
    min_len = min(student_seq.shape[1], teacher_seq.shape[1])
    student_seq = student_seq[:, :min_len]
    teacher_seq = teacher_seq[:, :min_len]
    out = {
        "frames": int(min_len),
        "mse": float(F.mse_loss(student_seq, teacher_seq).item()),
        "l1": float((student_seq - teacher_seq).abs().mean().item()),
        "student_mean": float(student_seq.mean().item()),
        "student_std": float(student_seq.std().item()),
        "teacher_mean": float(teacher_seq.mean().item()),
        "teacher_std": float(teacher_seq.std().item()),
    }
    if min_len > 1:
        s_vel = student_seq[:, 1:] - student_seq[:, :-1]
        t_vel = teacher_seq[:, 1:] - teacher_seq[:, :-1]
        out["vel_mse"] = float(F.mse_loss(s_vel, t_vel).item())
        out["student_vel_mag"] = float(s_vel.norm(dim=-1).mean().item())
        out["teacher_vel_mag"] = float(t_vel.norm(dim=-1).mean().item())
    if min_len > block_frames:
        idx = torch.arange(block_frames, min_len, block_frames, device=student_seq.device)
        s_jump = student_seq[:, idx] - student_seq[:, idx - 1]
        t_jump = teacher_seq[:, idx] - teacher_seq[:, idx - 1]
        out["boundary_mse"] = float(F.mse_loss(s_jump, t_jump).item())
        out["student_boundary_mag"] = float(s_jump.norm(dim=-1).mean().item())
        out["teacher_boundary_mag"] = float(t_jump.norm(dim=-1).mean().item())
    return out


def main():
    parser = argparse.ArgumentParser(description="Verify DyStream block-wise streaming distillation")
    parser.add_argument("--config", default="configs/distill/blockwise_stream_distill.yaml")
    parser.add_argument("--checkpoint", default="outputs/blockwise_stream_distill/blockwise_best.pt")
    parser.add_argument("--audio", default="wav_files/woc.wav")
    parser.add_argument("--ref-image", default="img_files/person1.png")
    parser.add_argument("--ref-npz", default="img_files/person1.npz")
    parser.add_argument("--output-dir", default="outputs/verify_blockwise")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--guidance-mode", choices=["full_5way", "uncond_all_2way", "all_only"], default=None)
    parser.add_argument("--no-concat", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DyStream verification")
    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    if args.teacher_steps is not None:
        cfg.teacher.denoising_steps = args.teacher_steps
    if args.guidance_mode is not None:
        cfg.teacher.guidance_mode = args.guidance_mode
    os.environ["DYSTREAM_WAV2VEC_PATH"] = cfg.wav2vec_path
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  DyStream Block-wise Distillation Verification")
    print("=" * 72)
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  audio:      {args.audio}")
    print(f"  ref npz:    {args.ref_npz}")
    print(f"  teacher:    {cfg.teacher.guidance_mode}, {cfg.teacher.denoising_steps} step(s)")
    print(f"  student:    K={cfg.student.block_frames}, H={cfg.student.history_frames}")

    print("\n[1/4] Loading models...")
    teacher = load_teacher(cfg).to(device)
    student, ckpt = load_blockwise_student(args.checkpoint, cfg, device)
    print(f"  student checkpoint step={ckpt.get('step', 'unknown')} loss={ckpt.get('loss', float('nan'))}")

    print("[2/4] Preparing inputs...")
    audio_np, audio_padded, audio_other, motion_in, anchor, inpaint_len, ref_img_resize = prepare_inputs(
        args.audio, args.ref_npz, args.ref_image, cfg, device
    )

    print("[3/4] Running teacher and student...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        feat_self, feat_other = extract_audio_features(teacher, audio_padded, audio_other)
    torch.cuda.synchronize()
    audio_feat_time = time.perf_counter() - t0

    torch.manual_seed(cfg.seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        teacher_out = teacher_rollout(teacher, audio_padded, audio_other, motion_in, anchor, cfg, int(cfg.seed))
        teacher_seq = teacher_out[:, inpaint_len:]
    torch.cuda.synchronize()
    teacher_time = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        student_seq = blockwise_rollout(
            student,
            feat_self,
            feat_other,
            anchor,
            target_frames=teacher_seq.shape[1],
            inpaint_len=inpaint_len,
            cfg=cfg,
        )
    torch.cuda.synchronize()
    student_time = time.perf_counter() - t0

    metrics = latent_metrics(student_seq, teacher_seq, cfg.student.block_frames)
    audio_dur = len(audio_np) / cfg.model.audio_sr
    metrics.update(
        {
            "audio_duration_sec": float(audio_dur),
            "audio_feature_time_sec": float(audio_feat_time),
            "teacher_time_sec": float(teacher_time),
            "student_time_sec": float(student_time),
            "teacher_rtf": float(teacher_time / max(audio_dur, 1e-6)),
            "student_rtf": float(student_time / max(audio_dur, 1e-6)),
            "student_speedup_vs_teacher": float(teacher_time / max(student_time, 1e-6)),
            "checkpoint": args.checkpoint,
            "teacher_steps": int(cfg.teacher.denoising_steps),
            "guidance_mode": str(cfg.teacher.guidance_mode),
            "block_frames": int(cfg.student.block_frames),
            "history_frames": int(cfg.student.history_frames),
        }
    )

    print("\n  Timing:")
    print(f"    audio features: {audio_feat_time:.3f}s")
    print(f"    teacher motion: {teacher_time:.3f}s  rtf={metrics['teacher_rtf']:.3f}")
    print(f"    student motion: {student_time:.3f}s  rtf={metrics['student_rtf']:.3f}")
    print(f"    speedup:        {metrics['student_speedup_vs_teacher']:.2f}x")
    print("\n  Latent metrics:")
    for key in ["frames", "mse", "l1", "vel_mse", "boundary_mse", "student_std", "teacher_std"]:
        if key in metrics:
            print(f"    {key}: {metrics[key]}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "teacher_motion": teacher_seq.detach().cpu(),
            "student_motion": student_seq.detach().cpu(),
            "metrics": metrics,
        },
        os.path.join(args.output_dir, "motion_compare.pt"),
    )
    print(f"\n  Saved metrics: {metrics_path}")

    if args.render:
        print("\n[4/4] Rendering videos...")
        teacher_video = os.path.join(args.output_dir, "teacher.mp4")
        student_video = os.path.join(args.output_dir, "blockwise_student.mp4")
        render_video(teacher_seq, ref_img_resize, args.audio, teacher_video)
        render_video(student_seq, ref_img_resize, args.audio, student_video)
        if not args.no_concat:
            make_side_by_side(
                [teacher_video, student_video],
                [f"teacher {cfg.teacher.guidance_mode}/{cfg.teacher.denoising_steps}step", f"blockwise K={cfg.student.block_frames}"],
                os.path.join(args.output_dir, "comparison_2way.mp4"),
            )
        print(f"  Videos saved in: {args.output_dir}")
    else:
        print("\n[4/4] Render skipped. Add --render to produce videos.")


if __name__ == "__main__":
    main()
