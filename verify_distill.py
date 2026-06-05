"""
验证蒸馏效果：对比 5步 / 原始1步 / 蒸馏1步 的速度和视频质量

Usage:
  .venv/bin/python verify_distill.py \
      --distilled outputs/distill_rollout/distilled_best.pt
"""

import os, sys, time, warnings, argparse, subprocess, shutil
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["DYSTREAM_WAV2VEC_PATH"] = "pretrained_model/wav2vec2-base-960h"

from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
import librosa

from utils import instantiate_motion_gen
from train_distill import load_teacher

def run_motion_gen(model, audio_padded, audio_other, motion_in, anchor, scheduler_cfg, steps):
    """Run motion generation and return (output, elapsed_time)."""
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_cfg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.inference(
            audio_padded, audio_other=audio_other,
            cond_motion=motion_in, init_motion=motion_in,
            anchor_motion=anchor, noise_scheduler=scheduler,
            num_inference_steps=steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return out, elapsed


def render_video(motion_latent, ref_img_path, audio_path, save_path):
    """Render motion latents to video using the visualization model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    npz_dir = os.path.join(os.path.dirname(save_path), "_tmp_npz")
    os.makedirs(npz_dir, exist_ok=True)

    npz_path = os.path.join(npz_dir, "verify_output.npz")
    np.savez(npz_path,
             motion_latent=motion_latent.cpu().numpy(),
             audio_path=audio_path,
             ref_img_path=ref_img_path,
             video_id="verify")

    video_dir = os.path.join(os.path.dirname(save_path), "_tmp_video")
    cmd = (f"cd tools/visualization_0416 && "
           f"{sys.executable} latent_to_video.py "
           f"--npz_dir {os.path.abspath(npz_dir)} "
           f"--save_dir {os.path.abspath(video_dir)} "
           f"--save_fps 25 --version '0506' 2>/dev/null")
    os.system(cmd)

    # Find and move the output video
    import glob
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    if videos:
        import shutil
        shutil.move(videos[0], save_path)
        print(f"    → Video saved: {save_path}")
    else:
        print(f"    ⚠ Render failed, no video produced")

    # Cleanup
    import shutil
    shutil.rmtree(npz_dir, ignore_errors=True)
    shutil.rmtree(video_dir, ignore_errors=True)


def make_side_by_side(videos, labels, output_path):
    """Create a horizontal comparison video with labels."""
    if len(videos) != len(labels):
        raise ValueError("videos and labels must have the same length")
    if not videos:
        return
    missing = [path for path in videos if not os.path.exists(path)]
    if missing:
        print(f"    ⚠ Cannot create concat; missing videos: {missing}")
        return
    if shutil.which("ffmpeg") is None:
        print("    ⚠ Cannot create concat; ffmpeg not found")
        return

    filter_parts = []
    stack_inputs = []
    for idx, label in enumerate(labels):
        safe_label = label.replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(
            f"[{idx}:v]scale=512:512,"
            f"drawbox=x=0:y=0:w=iw:h=42:color=black@0.55:t=fill,"
            f"drawtext=text='{safe_label}':x=12:y=11:fontsize=22:"
            f"fontcolor=white:box=0[v{idx}]"
        )
        stack_inputs.append(f"[v{idx}]")

    filter_complex = ";".join(filter_parts)
    filter_complex += ";" + "".join(stack_inputs) + f"hstack=inputs={len(videos)}[v]"

    cmd = ["ffmpeg", "-y"]
    for video in videos:
        cmd.extend(["-i", video])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        output_path,
    ])
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print(f"    → Side-by-side video saved: {output_path}")
    else:
        print("    ⚠ ffmpeg concat failed")
        print(result.stderr[-1000:])


def load_distilled_student(cfg, distilled_path, device):
    student = load_teacher(cfg).to(device)
    distilled = torch.load(distilled_path, map_location=device)
    student.diffusion_head.load_state_dict(distilled["diffusion_head"])
    student.time_embed.load_state_dict(distilled["time_embed"])
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    return student, distilled


def get_reference(ref_npz, fallback_image):
    data = np.load(ref_npz, allow_pickle=True)
    motion = data["motion_latent"] if "motion_latent" in data.files else data["random_data"]
    # The renderer must use the same canonical crop used to extract the reference
    # motion latent. img_to_latent.py stores the clean 512 crop in mask_img_path;
    # ref_img_path may point back to the original unaligned upload.
    candidates = []
    if "mask_img_path" in data.files:
        candidates.append(str(data["mask_img_path"]))
    candidates.append(fallback_image.replace(".png", "_resize.png"))
    if "ref_img_path" in data.files:
        candidates.append(str(data["ref_img_path"]))
    candidates.append(fallback_image)

    ref_img_path = None
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            ref_img_path = candidate
            break
    if ref_img_path is None:
        raise FileNotFoundError(f"No valid reference image found from: {candidates}")
    return motion, ref_img_path


def main():
    parser = argparse.ArgumentParser(description="Verify distilled DyStream diffusion head")
    parser.add_argument("--config", default="configs/distill/step_distill.yaml")
    parser.add_argument("--distilled", default="outputs/distill_rollout/distilled_best.pt")
    parser.add_argument("--audio", default="wav_files/woc.wav")
    parser.add_argument("--ref-image", default="img_files/person1.png")
    parser.add_argument("--ref-npz", default="img_files/person1.npz")
    parser.add_argument("--output-dir", default="outputs/verify")
    parser.add_argument("--render", action="store_true", help="Render comparison videos")
    parser.add_argument("--no-concat", action="store_true", help="Do not create side-by-side comparison video")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)

    print("=" * 60)
    print("  DyStream 蒸馏验证")
    print(f"  Audio: {args.audio}")
    print(f"  Ref:   {args.ref_image}")
    print(f"  Distilled: {args.distilled}")
    print("=" * 60)

    print("\n[1] Loading teacher model...")
    teacher = load_teacher(cfg).to(device)

    print("[2] Loading distilled student model...")
    student, distilled = load_distilled_student(cfg, args.distilled, device)
    print(f"    Trained {distilled.get('step', 'unknown')} steps, loss={distilled.get('loss', float('nan')):.6f}")

    # Prepare audio
    audio, _ = librosa.load(args.audio, sr=cfg.model.audio_sr)
    audio_dur = len(audio) / cfg.model.audio_sr
    inpaint_len = cfg.model.cbh_window_length - 2
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    pad_samples = inpaint_len * hop
    audio_t = torch.from_numpy(audio).float().to(device).unsqueeze(0)
    audio_padded = F.pad(audio_t, (pad_samples, 0))
    audio_other = torch.zeros_like(audio_padded)

    total_frames_with_prefix = audio_padded.shape[1] // hop
    motion_np, ref_img_resize = get_reference(args.ref_npz, args.ref_image)
    motion_latent = torch.from_numpy(motion_np).float().to(device)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.view(1, 1, -1)
    elif motion_latent.dim() == 2:
        motion_latent = motion_latent[:1].unsqueeze(0)
    anchor = motion_latent[:, 0:1, :]
    motion_in = anchor.repeat(1, total_frames_with_prefix, 1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Warmup
    print("[3] Warmup...")
    run_motion_gen(teacher, audio_padded, audio_other, motion_in, anchor, cfg.noise_scheduler, 1)
    run_motion_gen(student, audio_padded, audio_other, motion_in, anchor, cfg.noise_scheduler, 1)

    results = {}

    # ─── Test A: 5步 (原始) ───
    print("\n[4] Running: 5-step (teacher baseline)...")
    torch.manual_seed(42)
    out_5, t_5 = run_motion_gen(teacher, audio_padded, audio_other, motion_in, anchor, cfg.noise_scheduler, 5)
    out_5 = out_5[:, inpaint_len:]
    results["5-step"] = {"time": t_5, "output": out_5}
    print(f"    Time: {t_5*1000:.0f}ms, Frames: {out_5.shape[1]}, "
          f"Per-frame: {t_5/out_5.shape[1]*1000:.2f}ms")

    # ─── Test B: 1步 (原始权重，不蒸馏) ───
    print("\n[5] Running: 1-step (teacher weights, no distillation)...")
    torch.manual_seed(42)
    out_1_orig, t_1_orig = run_motion_gen(teacher, audio_padded, audio_other, motion_in, anchor, cfg.noise_scheduler, 1)
    out_1_orig = out_1_orig[:, inpaint_len:]
    results["1-step-orig"] = {"time": t_1_orig, "output": out_1_orig}
    print(f"    Time: {t_1_orig*1000:.0f}ms, Frames: {out_1_orig.shape[1]}, "
          f"Per-frame: {t_1_orig/out_1_orig.shape[1]*1000:.2f}ms")

    # ─── Test C: 1步 (蒸馏权重) ───
    print("\n[6] Running: 1-step (distilled student)...")
    torch.manual_seed(42)
    out_1_dist, t_1_dist = run_motion_gen(student, audio_padded, audio_other, motion_in, anchor, cfg.noise_scheduler, 1)
    out_1_dist = out_1_dist[:, inpaint_len:]
    results["1-step-distilled"] = {"time": t_1_dist, "output": out_1_dist}
    print(f"    Time: {t_1_dist*1000:.0f}ms, Frames: {out_1_dist.shape[1]}, "
          f"Per-frame: {t_1_dist/out_1_dist.shape[1]*1000:.2f}ms")

    # ─── Speed Summary ───
    print(f"\n{'='*60}")
    print(f"  SPEED COMPARISON ({audio_dur:.1f}s audio, {out_5.shape[1]} frames)")
    print(f"{'='*60}")
    print(f"  {'Config':<25s} {'Total':>8s} {'Per-frame':>10s} {'Speedup':>8s}")
    print(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*8}")
    print(f"  {'5-step (baseline)':<25s} {t_5*1000:>6.0f}ms {t_5/out_5.shape[1]*1000:>8.2f}ms {'1.00x':>8s}")
    print(f"  {'1-step (no distill)':<25s} {t_1_orig*1000:>6.0f}ms {t_1_orig/out_1_orig.shape[1]*1000:>8.2f}ms {t_5/t_1_orig:>7.2f}x")
    print(f"  {'1-step (distilled)':<25s} {t_1_dist*1000:>6.0f}ms {t_1_dist/out_1_dist.shape[1]*1000:>8.2f}ms {t_5/t_1_dist:>7.2f}x")

    # ─── Quality Stats ───
    print(f"\n  Motion Latent Statistics:")
    print(f"  {'Config':<25s} {'Mean':>8s} {'Std':>8s} {'Range':>16s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*16}")
    for name, r in results.items():
        o = r["output"]
        print(f"  {name:<25s} {o.mean():.4f}  {o.std():.4f}  [{o.min():.3f}, {o.max():.3f}]")

    # L1 distance vs baseline
    if out_5.shape == out_1_orig.shape:
        l1_orig = (out_5 - out_1_orig).abs().mean().item()
        l1_dist = (out_5 - out_1_dist).abs().mean().item()
        print(f"\n  L1 vs 5-step baseline:")
        print(f"    1-step (no distill): {l1_orig:.4f}")
        print(f"    1-step (distilled):  {l1_dist:.4f}")

    # ─── Render Videos ───
    if args.render:
        print(f"\n{'='*60}")
        print(f"  RENDERING VIDEOS...")
        print(f"{'='*60}")

        print("\n  [A] 5-step baseline:")
        video_5 = os.path.join(args.output_dir, "5step_baseline.mp4")
        render_video(out_5, ref_img_resize, args.audio,
                     video_5)

        print("  [B] 1-step (no distillation):")
        video_1_orig = os.path.join(args.output_dir, "1step_no_distill.mp4")
        render_video(out_1_orig, ref_img_resize, args.audio,
                     video_1_orig)

        print("  [C] 1-step (distilled):")
        video_1_dist = os.path.join(args.output_dir, "1step_distilled.mp4")
        render_video(out_1_dist, ref_img_resize, args.audio,
                     video_1_dist)

        if not args.no_concat:
            print("\n  [D] Side-by-side comparison:")
            make_side_by_side(
                [video_5, video_1_orig, video_1_dist],
                ["5-step teacher", "1-step original", "1-step distilled"],
                os.path.join(args.output_dir, "comparison_3way.mp4"),
            )

        print(f"\n  All videos in: {args.output_dir}/")
    print("  Done!")


if __name__ == "__main__":
    main()
