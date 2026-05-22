"""
Verify block-wise streaming AR distillation with two fixed visual cases.

The script always renders two teacher/student comparisons:
  1. one training-set sample using its own first frame and audio;
  2. a user-provided image driven by a user-provided audio file.

Only three user-facing inputs are kept:
  --img-path          reference image for the second case
  --audio-path        driving audio for the second case, default wav_files/woc.wav
  --train-sample-idx  training sample index for the first case
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
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".cache", "matplotlib"))

DEFAULT_CONFIG = "configs/distill/blockwise_stream_distill_cross_fm_gt_cache_full_trainval_100ep_scratch.yaml"
DEFAULT_AUDIO = "wav_files/test_audio_60s.wav"
DEFAULT_IMG = "img_files/person1.png"

from train_distill import LRS3AudioDataset, load_teacher
from train_blockwise_distill import (
    build_blockwise_student,
    blockwise_fm_rollout,
    blockwise_rollout,
    extract_audio_features,
    resolve_lrs3_roots,
    teacher_rollout,
)
from verify_distill import get_reference, make_side_by_side, render_video


def resolve_checkpoint(cfg):
    output_dir = cfg.output_dir
    candidates = [
        os.path.join(output_dir, "blockwise_latest.pt"),
        os.path.join(output_dir, "blockwise_best.pt"),
        os.path.join(output_dir, "blockwise_last.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No blockwise checkpoint found from: {candidates}")


def load_checkpoint_config(checkpoint_path, fallback_cfg):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        cfg = fallback_cfg
    migrate_legacy_config(cfg, fallback_cfg)
    return cfg, checkpoint


def migrate_legacy_config(cfg, fallback_cfg):
    if "lrs3_trainval_root" not in cfg and "lrs3_root" in cfg:
        cfg.lrs3_trainval_root = cfg.lrs3_root
    if "lrs3_pretrain_root" not in cfg and "lrs3_pretrain_root" in fallback_cfg:
        cfg.lrs3_pretrain_root = fallback_cfg.lrs3_pretrain_root
    if "split_mode" not in cfg.data:
        cfg.data.split_mode = fallback_cfg.data.get("split_mode", "trainval")


def default_output_dir(cfg):
    train_name = os.path.basename(str(cfg.output_dir).rstrip("/"))
    return os.path.join("outputs", f"verify_{train_name}_suite")


def load_blockwise_student(checkpoint, cfg, device):
    student = build_blockwise_student(cfg).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()
    for param in student.parameters():
        param.requires_grad = False
    return student


def write_wav(path, audio, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import soundfile as sf

        sf.write(path, audio, sr)
        return path
    except Exception:
        import wave

        audio_i16 = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(audio_i16.tobytes())
        return path


def load_train_sample(cfg, output_dir, sample_idx):
    dataset = LRS3AudioDataset(
        lrs3_root=resolve_lrs3_roots(cfg),
        audio_sr=cfg.model.audio_sr,
        duration_sec=cfg.data.audio_duration_sec,
        max_clips=cfg.data.get("max_clips", None),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No clips found for split_mode={cfg.data.get('split_mode', 'trainval')}")
    idx = sample_idx % len(dataset)
    item = dataset[idx]
    audio = item["audio"].numpy()
    video_path = dataset.clips[idx]
    audio_path = write_wav(os.path.join(output_dir, f"train_sample_{idx:06d}.wav"), audio, cfg.model.audio_sr)
    return audio, audio_path, video_path, idx


def extract_first_frame(video_path, output_path):
    import cv2

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.save(output_path)
    return output_path


def process_reference_image(image_path, output_dir, prefix):
    from app import process_image

    os.makedirs(output_dir, exist_ok=True)
    resized_pil, masked_pil, motion_latent = process_image(Image.open(image_path).convert("RGB"))
    resized_path = os.path.join(output_dir, f"{prefix}_resize.png")
    masked_path = os.path.join(output_dir, f"{prefix}_masked.png")
    npz_path = os.path.join(output_dir, f"{prefix}.npz")
    resized_pil.save(resized_path)
    masked_pil.save(masked_path)
    np.savez(
        npz_path,
        motion_latent=motion_latent.numpy(),
        ref_img_path=image_path,
        mask_img_path=resized_path,
    )
    return image_path, npz_path, resized_path


def reference_from_train_video(video_path, output_dir):
    frame_path = extract_first_frame(video_path, os.path.join(output_dir, "sample_first_frame.png"))
    return process_reference_image(frame_path, output_dir, "sample_first_frame")


def reference_from_image(image_path, output_dir):
    stem, _ = os.path.splitext(image_path)
    adjacent_npz = stem + ".npz"
    if os.path.exists(adjacent_npz):
        return image_path, adjacent_npz, None
    return process_reference_image(image_path, output_dir, "input_image")


def prepare_inputs(audio, ref_npz, ref_image, cfg, device):
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
    return audio_padded, audio_other, motion_in, anchor, inpaint_len, ref_img_resize


def latent_metrics(student_seq, teacher_seq, block_frames):
    min_len = min(student_seq.shape[1], teacher_seq.shape[1])
    student_seq = student_seq[:, :min_len]
    teacher_seq = teacher_seq[:, :min_len]
    out = {
        "frames": int(min_len),
        "mse": float(F.mse_loss(student_seq, teacher_seq).item()),
        "l1": float((student_seq - teacher_seq).abs().mean().item()),
        "student_std": float(student_seq.std().item()),
        "teacher_std": float(teacher_seq.std().item()),
    }
    if min_len > 1:
        s_vel = student_seq[:, 1:] - student_seq[:, :-1]
        t_vel = teacher_seq[:, 1:] - teacher_seq[:, :-1]
        out["vel_mse"] = float(F.mse_loss(s_vel, t_vel).item())
        out["student_vel_mag"] = float(s_vel.norm(dim=-1).mean().item())
        out["teacher_vel_mag"] = float(t_vel.norm(dim=-1).mean().item())
        out["vel_mag_ratio"] = float(out["student_vel_mag"] / max(out["teacher_vel_mag"], 1e-8))
    if min_len > block_frames:
        idx = torch.arange(block_frames, min_len, block_frames, device=student_seq.device)
        s_jump = student_seq[:, idx] - student_seq[:, idx - 1]
        t_jump = teacher_seq[:, idx] - teacher_seq[:, idx - 1]
        out["boundary_mse"] = float(F.mse_loss(s_jump, t_jump).item())
    return out


def run_case(case_name, teacher, student, cfg, checkpoint_path, audio, audio_path, ref_image, ref_npz, output_dir, device, extra_meta=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[{case_name}] audio={audio_path}")

    audio_padded, audio_other, motion_in, anchor, inpaint_len, ref_img_resize = prepare_inputs(
        audio, ref_npz, ref_image, cfg, device
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        feat_self, feat_other = extract_audio_features(teacher, audio_padded, audio_other)
    torch.cuda.synchronize()
    audio_feature_time = time.perf_counter() - t0

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
        if cfg.student.get("architecture", "additive") == "cross_fm":
            student_seq = blockwise_fm_rollout(
                student,
                feat_self,
                feat_other,
                anchor,
                target_frames=teacher_seq.shape[1],
                inpaint_len=inpaint_len,
                cfg=cfg,
                noise_scheduler=FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler),
                teacher_seq=None,
                seed=int(cfg.seed),
            )
        else:
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

    audio_duration = len(audio) / cfg.model.audio_sr
    metrics = latent_metrics(student_seq, teacher_seq, cfg.student.block_frames)
    metrics.update(
        {
            "case": case_name,
            "audio_duration_sec": float(audio_duration),
            "audio_feature_time_sec": float(audio_feature_time),
            "teacher_time_sec": float(teacher_time),
            "student_time_sec": float(student_time),
            "teacher_rtf": float(teacher_time / max(audio_duration, 1e-6)),
            "student_rtf": float(student_time / max(audio_duration, 1e-6)),
            "speedup": float(teacher_time / max(student_time, 1e-6)),
            "checkpoint": checkpoint_path,
            "audio_path": audio_path,
            "ref_image": ref_image,
            "ref_npz": ref_npz,
            "block_frames": int(cfg.student.block_frames),
            "history_frames": int(cfg.student.history_frames),
        }
    )
    if extra_meta:
        metrics.update(extra_meta)

    print(
        f"  frames={metrics['frames']} mse={metrics['mse']:.6f} "
        f"teacher={teacher_time:.3f}s student={student_time:.3f}s speedup={metrics['speedup']:.2f}x"
    )

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "teacher_motion": teacher_seq.detach().cpu(),
            "student_motion": student_seq.detach().cpu(),
            "metrics": metrics,
        },
        os.path.join(output_dir, "motion_compare.pt"),
    )

    teacher_video = os.path.join(output_dir, "teacher.mp4")
    student_video = os.path.join(output_dir, "blockwise_student.mp4")
    comparison_video = os.path.join(output_dir, "comparison_2way.mp4")
    render_video(teacher_seq, ref_img_resize, audio_path, teacher_video)
    render_video(student_seq, ref_img_resize, audio_path, student_video)
    make_side_by_side(
        [teacher_video, student_video],
        [f"teacher {cfg.teacher.guidance_mode}/{cfg.teacher.denoising_steps}step", f"blockwise K={cfg.student.block_frames}"],
        comparison_video,
    )
    metrics["comparison_video"] = comparison_video
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run the fixed blockwise distillation verification suite")
    parser.add_argument("--img-path", default=DEFAULT_IMG)
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO)
    parser.add_argument("--train-sample-idx", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DyStream verification")
    device = torch.device("cuda")

    fallback_cfg = OmegaConf.load(DEFAULT_CONFIG)
    checkpoint_path = resolve_checkpoint(fallback_cfg)
    cfg, checkpoint = load_checkpoint_config(checkpoint_path, fallback_cfg)
    os.environ["DYSTREAM_WAV2VEC_PATH"] = cfg.wav2vec_path
    output_dir = default_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  DyStream Block-wise Distillation Verification")
    print("=" * 72)
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  teacher:    {cfg.teacher.guidance_mode}, {cfg.teacher.denoising_steps} step(s)")
    print(f"  student:    K={cfg.student.block_frames}, H={cfg.student.history_frames}")

    teacher = load_teacher(cfg).to(device).eval()
    checkpoint = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in checkpoint.items()}
    student = load_blockwise_student(checkpoint, cfg, device)

    train_output_dir = os.path.join(output_dir, f"train_sample_{args.train_sample_idx:06d}_own_ref")
    train_audio, train_audio_path, train_video_path, train_idx = load_train_sample(
        cfg,
        train_output_dir,
        args.train_sample_idx,
    )
    train_ref_image, train_ref_npz, train_ref_resize = reference_from_train_video(train_video_path, train_output_dir)

    image_case_dir = os.path.join(output_dir, "input_image_audio")
    image_ref, image_npz, image_resize = reference_from_image(args.img_path, image_case_dir)
    image_audio, _ = librosa.load(args.audio_path, sr=cfg.model.audio_sr)

    suite_metrics = [
        run_case(
            "train_sample_own_ref",
            teacher,
            student,
            cfg,
            checkpoint_path,
            train_audio,
            train_audio_path,
            train_ref_image,
            train_ref_npz,
            train_output_dir,
            device,
            extra_meta={
                "train_sample_idx": train_idx,
                "train_video_path": train_video_path,
                "train_reference_resize": train_ref_resize,
            },
        ),
        run_case(
            "input_image_audio",
            teacher,
            student,
            cfg,
            checkpoint_path,
            image_audio,
            args.audio_path,
            image_ref,
            image_npz,
            image_case_dir,
            device,
            extra_meta={"input_image_resize": image_resize},
        ),
    ]

    with open(os.path.join(output_dir, "suite_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(suite_metrics, f, indent=2)
    print(f"\nDone. Videos and metrics saved in: {output_dir}")


if __name__ == "__main__":
    main()
