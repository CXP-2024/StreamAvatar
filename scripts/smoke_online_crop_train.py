"""
Smoke test online video crop + motion-encoder training cost.

This script intentionally does not use the precomputed motion cache. Each step:
  1. samples one mp4,
  2. decodes a random short video/audio segment,
  3. detects one fixed crop on that segment,
  4. encodes frames into DyStream motion latents,
  5. runs one current blockwise student train step,
  6. reports per-stage timings.

It is meant for throughput/debug evaluation, not full training.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_lrs3_motion_cache import encode_motion_latents, find_fixed_crop  # noqa: E402
from train_blockwise_distill import (  # noqa: E402
    blockwise_fm_rollout,
    build_blockwise_student,
    distill_loss,
    extract_audio_features,
)
from train_distill import load_teacher  # noqa: E402


def stamp():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def list_videos(input_root, max_videos=None):
    videos = sorted(Path(input_root).rglob("*.mp4"))
    if max_videos:
        videos = videos[: int(max_videos)]
    if not videos:
        raise RuntimeError(f"no mp4 files found under {input_root}")
    return videos


def read_video_segment(video_path, pose_fps, duration_sec, rng):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or pose_fps
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_pose_frames = max(1, int(total_src_frames / max(src_fps, 1e-6) * pose_fps))
    target_frames = max(1, int(round(duration_sec * pose_fps)))
    max_pose_start = max(0, total_pose_frames - target_frames)
    pose_start = rng.randint(0, max_pose_start) if max_pose_start > 0 else 0
    src_start = int(round(pose_start * src_fps / pose_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)
    src_step = max(src_fps / float(pose_fps), 1.0)
    frames = []
    local_idx = 0
    next_keep = 0.0
    while len(frames) < target_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if local_idx + 1e-6 >= next_keep:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            next_keep += src_step
        local_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {video_path}")
    return frames, pose_start, target_frames


def read_audio_segment(video_path, pose_start, target_frames, pose_fps, audio_sr):
    audio, _ = librosa.load(str(video_path), sr=audio_sr, mono=True)
    hop = audio_sr // pose_fps
    sample_start = pose_start * hop
    sample_len = target_frames * hop
    segment = audio[sample_start : sample_start + sample_len].astype(np.float32, copy=False)
    if segment.shape[0] < sample_len:
        segment = np.pad(segment, (0, sample_len - segment.shape[0]))
    return torch.from_numpy(segment).float()


def encode_motion_latents_batched(list_of_frames, crop_infos, batch_size):
    import app
    from scripts.preprocess_lrs3_motion_cache import apply_crop

    app.load_visualization_model()
    transform = app._vis_ctx["transform"]
    motion_encoder = app._vis_ctx["motion_encoder"]

    flat_images = []
    lengths = []
    for frames, crop_info in zip(list_of_frames, crop_infos):
        lengths.append(len(frames))
        flat_images.extend([apply_crop(frame, crop_info).convert("RGB") for frame in frames])

    latents = []
    with torch.no_grad():
        for start in range(0, len(flat_images), batch_size):
            images = flat_images[start : start + batch_size]
            tensor = torch.stack([transform(image) for image in images], dim=0).to(app.DEVICE)
            out = motion_encoder(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            latents.append(out.detach().float().cpu())

    flat_latents = torch.cat(latents, dim=0)
    chunks = []
    offset = 0
    for length in lengths:
        chunks.append(flat_latents[offset : offset + length])
        offset += length
    return torch.stack(chunks, dim=0)


def prepare_rollout_inputs(audio, motion_latent, cfg, device):
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    inpaint_len = cfg.model.cbh_window_length - 2
    pad_samples = inpaint_len * hop
    audio = audio.to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio_padded = F.pad(audio, (pad_samples, 0))
    audio_other = torch.zeros_like(audio_padded)
    target_seq = motion_latent.to(device)
    if target_seq.dim() == 2:
        target_seq = target_seq.unsqueeze(0)
    anchor = target_seq[:, :1]
    return audio_padded, audio_other, target_seq, anchor, inpaint_len


def train_one_step(student, teacher, optimizer, noise_scheduler, audio, motion_latent, cfg, device, seed):
    audio_padded, audio_other, target_seq, anchor, inpaint_len = prepare_rollout_inputs(audio, motion_latent, cfg, device)

    t0 = stamp()
    with torch.no_grad():
        feat_self, feat_other = extract_audio_features(teacher, audio_padded, audio_other)
    audio_encode_time = stamp() - t0

    t0 = stamp()
    return_residual = cfg.loss.get("delta_weight", 0.0) > 0
    with autocast(enabled=bool(cfg.training.fp16)):
        rollout_out = blockwise_fm_rollout(
            student,
            feat_self,
            feat_other,
            anchor,
            target_frames=target_seq.shape[1],
            inpaint_len=inpaint_len,
            cfg=cfg,
            noise_scheduler=noise_scheduler,
            teacher_seq=target_seq,
            seed=seed,
            return_residual=return_residual,
        )
        if return_residual:
            student_seq, base_seq, delta_seq = rollout_out
        else:
            student_seq, base_seq, delta_seq = rollout_out, None, None
        loss, loss_motion, loss_vel, loss_acc, loss_boundary, loss_delta, matched = distill_loss(
            student_seq,
            target_seq,
            cfg,
            base_seq=base_seq,
            delta_seq=delta_seq,
        )
    forward_time = stamp() - t0

    t0 = stamp()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.grad_clip)
    optimizer.step()
    backward_time = stamp() - t0

    metrics = {
        "loss": float(loss.detach().item()),
        "loss_motion": float(loss_motion.detach().item()),
        "loss_velocity": float(loss_vel.detach().item()),
        "loss_acceleration": float(loss_acc.detach().item()),
        "loss_boundary": float(loss_boundary.detach().item()),
        "matched_frames": int(matched),
        "audio_encode": audio_encode_time,
        "student_forward": forward_time,
        "backward": backward_time,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Smoke online crop training cost for DyStream")
    parser.add_argument("--config", default="configs/distill/blockwise_stream_distill_cross_fm_gt_cache_pretrain_24k_scratch.yaml")
    parser.add_argument("--input-root", default="/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/pretrain/pretrain")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--max-videos", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of video clips per train step.")
    parser.add_argument("--motion-batch-size", type=int, default=64, help="Frame batch size for motion_encoder.")
    parser.add_argument("--union-bbox-scale", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")

    cfg = OmegaConf.load(args.config)
    cfg.data.audio_duration_sec = args.duration_sec
    cfg.training.batch_size = 1
    device = torch.device("cuda")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    videos = list_videos(args.input_root, args.max_videos)
    print(json.dumps({
        "event": "setup",
        "videos": len(videos),
        "duration_sec": args.duration_sec,
        "target_frames": int(round(args.duration_sec * cfg.model.pose_fps)),
        "batch_size": args.batch_size,
        "motion_batch_size": args.motion_batch_size,
        "config": args.config,
    }))

    print("[1/3] loading frozen teacher/audio encoders")
    teacher = load_teacher(cfg).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("[2/3] building student")
    student = build_blockwise_student(cfg).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler)

    print("[3/3] online crop training smoke")
    totals = {}
    ok_steps = 0
    pbar = tqdm(range(1, args.steps + 1), desc="online smoke", dynamic_ncols=True)
    # Gradio monkey-patches tqdm in this environment and expects this attr.
    pbar._progress = None
    for step in pbar:
        video_paths = [rng.choice(videos) for _ in range(args.batch_size)]
        step_t0 = stamp()
        try:
            t0 = stamp()
            batch_frames = []
            pose_starts = []
            target_frames = None
            for video_path in video_paths:
                frames, pose_start, current_target_frames = read_video_segment(
                    video_path,
                    cfg.model.pose_fps,
                    args.duration_sec,
                    rng,
                )
                if len(frames) < current_target_frames:
                    frames = frames + [frames[-1]] * (current_target_frames - len(frames))
                batch_frames.append(frames[:current_target_frames])
                pose_starts.append(pose_start)
                target_frames = current_target_frames if target_frames is None else min(target_frames, current_target_frames)
            batch_frames = [frames[:target_frames] for frames in batch_frames]
            video_decode_time = stamp() - t0

            t0 = stamp()
            audio = torch.stack([
                read_audio_segment(video_path, pose_start, target_frames, cfg.model.pose_fps, cfg.model.audio_sr)
                for video_path, pose_start in zip(video_paths, pose_starts)
            ], dim=0)
            audio_decode_time = stamp() - t0

            t0 = stamp()
            crop_infos = [find_fixed_crop(frames, args.union_bbox_scale) for frames in batch_frames]
            crop_time = stamp() - t0

            t0 = stamp()
            motion_latent = encode_motion_latents_batched(batch_frames, crop_infos, args.motion_batch_size)
            motion_encode_time = stamp() - t0

            train_metrics = train_one_step(
                student,
                teacher,
                optimizer,
                noise_scheduler,
                audio,
                motion_latent,
                cfg,
                device,
                seed=args.seed + step,
            )
            total_time = stamp() - step_t0
            row = {
                "event": "step",
                "step": step,
                "batch_size": int(args.batch_size),
                "video_id": video_paths[0].with_suffix("").name,
                "frames": int(motion_latent.shape[1]),
                "pose_start": int(pose_starts[0]),
                "video_decode": video_decode_time,
                "audio_decode": audio_decode_time,
                "crop_detect": crop_time,
                "motion_encode": motion_encode_time,
                "total": total_time,
                **train_metrics,
            }
            ok_steps += 1
            for key, value in row.items():
                if isinstance(value, (int, float)) and key not in {"step", "frames", "pose_start"}:
                    totals[key] = totals.get(key, 0.0) + float(value)
            print(json.dumps(row, ensure_ascii=False))
            pbar.set_postfix({
                "loss": f"{row['loss']:.4f}",
                "total": f"{row['total']:.1f}s",
                "motion": f"{row['motion_encode']:.1f}s",
            })
        except Exception as exc:
            print(json.dumps({
                "event": "error",
                "step": step,
                "video": str(video_paths[0]) if video_paths else None,
                "error": repr(exc),
            }, ensure_ascii=False))

    if ok_steps:
        avg = {key: value / ok_steps for key, value in totals.items()}
        avg["event"] = "average"
        avg["ok_steps"] = ok_steps
        print(json.dumps(avg, ensure_ascii=False))


if __name__ == "__main__":
    main()
