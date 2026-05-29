import argparse
import json
import os
import sys
import time
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from train_distill import load_teacher
from train_blockwise_distill import (  # noqa: E402
    blockwise_fm_rollout,
    blockwise_rollout,
    build_blockwise_student,
    extract_audio_features,
    teacher_rollout,
)
from verify_blockwise_distill import load_checkpoint_config, migrate_legacy_config  # noqa: E402
from verify_distill import get_reference, make_side_by_side, render_video  # noqa: E402


def latent_metrics(a, b, name):
    n = min(a.shape[1], b.shape[1])
    a = a[:, :n]
    b = b[:, :n]
    out = {
        f"{name}_frames": int(n),
        f"{name}_mse": float(F.mse_loss(a, b).item()),
        f"{name}_l1": float((a - b).abs().mean().item()),
        f"{name}_a_std": float(a.std().item()),
        f"{name}_b_std": float(b.std().item()),
    }
    if n > 1:
        av = a[:, 1:] - a[:, :-1]
        bv = b[:, 1:] - b[:, :-1]
        out[f"{name}_vel_mse"] = float(F.mse_loss(av, bv).item())
        out[f"{name}_a_vel_mag"] = float(av.norm(dim=-1).mean().item())
        out[f"{name}_b_vel_mag"] = float(bv.norm(dim=-1).mean().item())
    return out


def load_student(checkpoint, cfg, device):
    student = build_blockwise_student(cfg).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()
    for param in student.parameters():
        param.requires_grad = False
    return student


def prepare_audio(audio_path, cfg, device):
    audio, _ = librosa.load(audio_path, sr=cfg.model.audio_sr)
    audio_t = torch.from_numpy(audio).float().to(device).unsqueeze(0)
    inpaint_len = cfg.model.cbh_window_length - 2
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    audio_padded = F.pad(audio_t, (inpaint_len * hop, 0))
    audio_other = torch.zeros_like(audio_padded)
    total_frames = audio_padded.shape[1] // hop
    return audio, audio_padded, audio_other, total_frames, inpaint_len


def make_motion_in(anchor, total_frames):
    return anchor.repeat(1, total_frames, 1)


def run_student(student, feat_self, feat_other, anchor, target_frames, inpaint_len, cfg, seed):
    if cfg.student.get("architecture", "additive") == "cross_fm":
        return blockwise_fm_rollout(
            student,
            feat_self,
            feat_other,
            anchor,
            target_frames=target_frames,
            inpaint_len=inpaint_len,
            cfg=cfg,
            noise_scheduler=FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler),
            teacher_seq=None,
            seed=seed,
        )
    return blockwise_rollout(
        student,
        feat_self,
        feat_other,
        anchor,
        target_frames=target_frames,
        inpaint_len=inpaint_len,
        cfg=cfg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-compare",
        default="outputs/verify_blockwise_stream_distill_cross_fm_suite/input_image_audio/motion_compare.pt",
    )
    parser.add_argument("--fallback-config", default="configs/distill/blockwise_stream_distill_cross_fm.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    motion_compare_path = Path(args.motion_compare)
    original = torch.load(motion_compare_path, map_location="cpu")
    original_metrics = original.get("metrics", {})
    checkpoint_path = original_metrics.get("checkpoint")
    audio_path = original_metrics.get("audio_path")
    ref_image = original_metrics.get("ref_image")
    ref_npz = original_metrics.get("ref_npz")
    if not checkpoint_path or not audio_path or not ref_image or not ref_npz:
        raise ValueError(f"motion_compare metrics does not contain checkpoint/audio/ref paths: {motion_compare_path}")

    fallback_cfg = OmegaConf.load(args.fallback_config)
    cfg, checkpoint = load_checkpoint_config(checkpoint_path, fallback_cfg)
    migrate_legacy_config(cfg, fallback_cfg)
    os.environ["DYSTREAM_WAV2VEC_PATH"] = cfg.wav2vec_path

    output_dir = Path(args.output_dir) if args.output_dir else motion_compare_path.parent / "anchor_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {checkpoint_path}")
    print(f"audio:      {audio_path}")
    print(f"ref_npz:    {ref_npz}")
    print(f"output:     {output_dir}")

    teacher = load_teacher(cfg).to(device).eval()
    checkpoint = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in checkpoint.items()}
    student = load_student(checkpoint, cfg, device)

    audio, audio_padded, audio_other, total_frames, inpaint_len = prepare_audio(audio_path, cfg, device)
    motion_np, ref_img_resize = get_reference(ref_npz, ref_image)
    real_anchor = torch.from_numpy(motion_np).float().to(device)
    if real_anchor.dim() == 1:
        real_anchor = real_anchor.view(1, 1, -1)
    elif real_anchor.dim() == 2:
        real_anchor = real_anchor[:1].unsqueeze(0)
    real_anchor = real_anchor[:, :1, :]

    torch.manual_seed(int(cfg.seed))
    random_anchor = torch.randn_like(real_anchor) * float(cfg.data.get("anchor_std", 0.35))

    with torch.no_grad():
        t0 = time.perf_counter()
        feat_self, feat_other = extract_audio_features(teacher, audio_padded, audio_other)
        torch.cuda.synchronize()
        audio_feature_time = time.perf_counter() - t0

        target_frames = total_frames - inpaint_len
        if args.max_frames > 0:
            target_frames = min(target_frames, args.max_frames)

        # Teacher baseline uses the real reference/image anchor.
        torch.manual_seed(int(cfg.seed))
        t0 = time.perf_counter()
        teacher_out = teacher_rollout(
            teacher,
            audio_padded,
            audio_other,
            make_motion_in(real_anchor, total_frames),
            real_anchor,
            cfg,
            int(cfg.seed),
        )
        teacher_real = teacher_out[:, inpaint_len: inpaint_len + target_frames]
        torch.cuda.synchronize()
        teacher_time = time.perf_counter() - t0

        # Student baseline uses the random-anchor training distribution.
        torch.manual_seed(int(cfg.seed))
        t0 = time.perf_counter()
        student_random = run_student(
            student, feat_self, feat_other, random_anchor, target_frames, inpaint_len, cfg, int(cfg.seed)
        )
        torch.cuda.synchronize()
        student_random_time = time.perf_counter() - t0

        # Hack: same student, but replace random anchor with real image/cache anchor.
        torch.manual_seed(int(cfg.seed))
        t0 = time.perf_counter()
        student_real = run_student(
            student, feat_self, feat_other, real_anchor, target_frames, inpaint_len, cfg, int(cfg.seed)
        )
        torch.cuda.synchronize()
        student_real_time = time.perf_counter() - t0

        # Optional teacher sensitivity diagnostic.
        torch.manual_seed(int(cfg.seed))
        t0 = time.perf_counter()
        teacher_random_out = teacher_rollout(
            teacher,
            audio_padded,
            audio_other,
            make_motion_in(random_anchor, total_frames),
            random_anchor,
            cfg,
            int(cfg.seed),
        )
        teacher_random = teacher_random_out[:, inpaint_len: inpaint_len + target_frames]
        torch.cuda.synchronize()
        teacher_random_time = time.perf_counter() - t0

    metrics = {
        "motion_compare": str(motion_compare_path),
        "checkpoint": checkpoint_path,
        "audio_path": audio_path,
        "ref_image": ref_image,
        "ref_npz": ref_npz,
        "frames": int(target_frames),
        "audio_feature_time_sec": float(audio_feature_time),
        "teacher_real_time_sec": float(teacher_time),
        "teacher_random_time_sec": float(teacher_random_time),
        "student_random_time_sec": float(student_random_time),
        "student_real_anchor_hack_time_sec": float(student_real_time),
        "real_anchor_norm": float(real_anchor.norm(dim=-1).mean().item()),
        "random_anchor_norm": float(random_anchor.norm(dim=-1).mean().item()),
    }
    metrics.update(latent_metrics(student_random, teacher_real, "student_random_vs_teacher_real"))
    metrics.update(latent_metrics(student_real, teacher_real, "student_real_anchor_hack_vs_teacher_real"))
    metrics.update(latent_metrics(student_real, student_random, "student_real_anchor_hack_vs_student_random"))
    metrics.update(latent_metrics(teacher_random, teacher_real, "teacher_random_vs_teacher_real"))

    torch.save(
        {
            "teacher_real_anchor_motion": teacher_real.detach().cpu(),
            "teacher_random_anchor_motion": teacher_random.detach().cpu(),
            "student_random_anchor_motion": student_random.detach().cpu(),
            "student_real_anchor_hack_motion": student_real.detach().cpu(),
            "real_anchor": real_anchor.detach().cpu(),
            "random_anchor": random_anchor.detach().cpu(),
            "metrics": metrics,
        },
        output_dir / "anchor_sensitivity_motion.pt",
    )
    with open(output_dir / "anchor_sensitivity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.render:
        videos = []
        labels = []
        for name, seq, label in [
            ("teacher_real_anchor", teacher_real, "teacher real anchor"),
            ("student_random_anchor", student_random, "student random anchor"),
            ("student_real_anchor_hack", student_real, "student real anchor hack"),
        ]:
            out = str(output_dir / f"{name}.mp4")
            render_video(seq, ref_img_resize, audio_path, out)
            videos.append(out)
            labels.append(label)
        make_side_by_side(videos, labels, str(output_dir / "comparison_teacher_student_anchorhack.mp4"))
        metrics["comparison_video"] = str(output_dir / "comparison_teacher_student_anchorhack.mp4")
        with open(output_dir / "anchor_sensitivity_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
