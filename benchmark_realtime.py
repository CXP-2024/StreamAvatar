"""
DyStream Per-Stage Latency Benchmark
Tests: person2 image + test_audio_60s.wav (60s audio)
Measures each pipeline stage independently for real-time feasibility analysis.
"""

import os, sys, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
import librosa

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".torch_cache")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
os.environ["DYSTREAM_WAV2VEC_PATH"] = "pretrained_model/wav2vec2-base-960h"

from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
from utils import instantiate_motion_gen


# ─── Config ───────────────────────────────────────────────
REF_IMAGE = "img_files/person2.png"
REF_NPZ = "img_files/person2.npz"
AUDIO_PATH = "wav_files/test_audio_60s.wav"
CHECKPOINT = "checkpoints/last.ckpt"
CONFIG_PATH = "configs/motion_gen/custom_current.yaml"


def load_config_and_model():
    """Load config and model from checkpoint."""
    cfg = OmegaConf.load(CONFIG_PATH)
    model_cfg = cfg.model

    print("Loading model...")
    t0 = time.perf_counter()
    model = instantiate_motion_gen(
        module_name=model_cfg.module_name,
        class_name=model_cfg.class_name,
        cfg=model_cfg,
        hfstyle=False
    )

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Strip "model." prefix if present (Lightning wrapping)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[6:]] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model = model.cuda().eval()
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    noise_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler_kwargs)
    return model, cfg, noise_scheduler


@torch.no_grad()
def benchmark():
    print("=" * 60)
    print("  DyStream Real-Time Feasibility Benchmark")
    print("  Input: person2 + test_audio_60s.wav (60s)")
    print("  Target: 40ms/frame (25fps realtime)")
    print("=" * 60)

    # ─── Stage 0: Model Loading ───
    model, cfg, noise_scheduler = load_config_and_model()
    model_cfg = cfg.model

    # ─── Stage 1: Audio Loading & Preparation ───
    print("\n[Stage 1] Audio Loading & Preparation")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    audio, _ = librosa.load(AUDIO_PATH, sr=model_cfg.audio_sr)
    audio_load_time = time.perf_counter() - t0

    additional_motion_seq = model.inpainting_length
    audio_padded = np.concatenate([
        np.zeros((additional_motion_seq * int(model_cfg.audio_sr / model_cfg.pose_fps))),
        audio
    ], axis=0)
    audio_tensor = torch.from_numpy(audio_padded).float().cuda().unsqueeze(0)
    # No listener audio — speaker-only mode
    audio_other = torch.zeros_like(audio_tensor).cuda()

    audio_prep_time = time.perf_counter() - t0 - audio_load_time
    total_frames = audio_tensor.shape[1] // int(model_cfg.audio_sr / model_cfg.pose_fps)
    video_duration = total_frames / model_cfg.pose_fps

    print(f"  Audio load: {audio_load_time*1000:.1f}ms")
    print(f"  Audio prep: {audio_prep_time*1000:.1f}ms")
    print(f"  Total frames: {total_frames} ({video_duration:.1f}s video)")

    # ─── Stage 2: Motion Latent Loading (reference) ───
    print("\n[Stage 2] Reference Motion Latent")
    t0 = time.perf_counter()
    motion_data = np.load(REF_NPZ, allow_pickle=True)
    motion_latent = torch.from_numpy(motion_data["motion_latent"]).cuda().unsqueeze(0)
    t_total = total_frames
    motion_latent_in = motion_latent[:, 0:1, :].repeat(1, t_total, 1)
    anchor_motion = motion_latent[:, 0:1, :]
    latent_load_time = time.perf_counter() - t0
    print(f"  Latent load: {latent_load_time*1000:.1f}ms")
    print(f"  Motion dim: {motion_latent.shape[-1]}, expanded to {motion_latent_in.shape}")

    # ─── Stage 3: Audio Encoding (Wav2Vec2 × 2) ───
    print("\n[Stage 3] Audio Encoding (Wav2Vec2)")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Replicate what model.inference() does internally for audio encoding
    audio_list = [audio_tensor[0].cpu().numpy()]
    inputs = model.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Speaker audio
    audio_fea = model.audio_encoder_face(inputs["input_values"], attention_mask=inputs.get("attention_mask"))["high_level"]
    audio_fea = F.interpolate(audio_fea.permute(0, 2, 1), scale_factor=(model_cfg.pose_fps / 50), mode='linear', align_corners=True).permute(0, 2, 1)
    torch.cuda.synchronize()
    speaker_enc_time = time.perf_counter() - t0

    # Listener audio (zeros)
    t0 = time.perf_counter()
    audio_other_list = [audio_other[0].cpu().numpy()]
    inputs_other = model.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs_other = {k: v.cuda() for k, v in inputs_other.items()}
    audio_other_fea = model.audio_encoder_face_other(inputs_other["input_values"], attention_mask=inputs_other.get("attention_mask"))["high_level"]
    audio_other_fea = F.interpolate(audio_other_fea.permute(0, 2, 1), scale_factor=(model_cfg.pose_fps / 50), mode='linear', align_corners=True).permute(0, 2, 1)
    torch.cuda.synchronize()
    listener_enc_time = time.perf_counter() - t0

    print(f"  Speaker Wav2Vec2: {speaker_enc_time*1000:.1f}ms")
    print(f"  Listener Wav2Vec2: {listener_enc_time*1000:.1f}ms")
    print(f"  Audio feature shape: {audio_fea.shape}")

    # ─── Stage 4: Motion Generation (GPT + Diffusion) ───
    print("\n[Stage 4] Motion Generation (GPT autoregressive + Flow Matching)")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    motion_latent_pred = model.inference(
        audio_tensor,
        cond_motion=motion_latent_in,
        audio_other=audio_other,
        init_motion=motion_latent_in,
        anchor_motion=anchor_motion,
        noise_scheduler=noise_scheduler,
        num_inference_steps=cfg.validation.denoising_steps,
    )
    motion_latent_pred = motion_latent_pred[:, additional_motion_seq:]

    torch.cuda.synchronize()
    motion_gen_time = time.perf_counter() - t0
    output_frames = motion_latent_pred.shape[1]
    ms_per_frame_motion = motion_gen_time / output_frames * 1000

    print(f"  Total motion gen: {motion_gen_time:.2f}s")
    print(f"  Output frames: {output_frames}")
    print(f"  Per-frame: {ms_per_frame_motion:.1f}ms/frame")
    print(f"  Denoising steps: {cfg.validation.denoising_steps}")

    # Save motion latent for rendering benchmark
    os.makedirs("./results", exist_ok=True)
    npz_path = "./results/benchmark_motion_output.npz"
    np.savez(npz_path,
             motion_latent=motion_latent_pred.cpu().numpy(),
             audio_path=AUDIO_PATH,
             ref_img_path="img_files/person2_resize.png",
             video_id="benchmark_person2_60s")
    print(f"  Saved motion to: {npz_path}")

    # ─── Stage 5: Rendering (latent_to_video) ───
    print("\n[Stage 5] Video Rendering (flow estimator + face generator)")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    render_cmd = (
        f"{sys.executable} tools/visualization_0416/latent_to_video.py "
        f"--npz_dir ./results --save_dir ./results/benchmark_video "
        f"--save_fps 25 --version '0506'"
    )
    print(f"  Running: {render_cmd}")
    os.system(render_cmd)

    torch.cuda.synchronize()
    render_time = time.perf_counter() - t0
    ms_per_frame_render = render_time / output_frames * 1000

    print(f"  Total render: {render_time:.2f}s")
    print(f"  Per-frame: {ms_per_frame_render:.1f}ms/frame")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Video duration:        {video_duration:.1f}s ({output_frames} frames)")
    print(f"  Target budget:         40.0ms/frame (25fps realtime)")
    print(f"")
    print(f"  {'Stage':<30s} {'Total':>8s} {'Per-frame':>12s} {'% of budget':>12s}")
    print(f"  {'─'*30} {'─'*8} {'─'*12} {'─'*12}")

    stages = [
        ("Audio encoding (speaker)", speaker_enc_time, speaker_enc_time / output_frames * 1000),
        ("Audio encoding (listener)", listener_enc_time, listener_enc_time / output_frames * 1000),
        ("Motion generation (GPT+FM)", motion_gen_time, ms_per_frame_motion),
        ("Video rendering", render_time, ms_per_frame_render),
    ]
    total_per_frame = 0
    for name, total_t, per_frame_ms in stages:
        pct = per_frame_ms / 40 * 100
        total_per_frame += per_frame_ms
        print(f"  {name:<30s} {total_t:>7.2f}s {per_frame_ms:>9.1f}ms   {pct:>9.0f}%")

    print(f"  {'─'*30} {'─'*8} {'─'*12} {'─'*12}")
    total_inference = motion_gen_time + render_time
    print(f"  {'TOTAL (motion+render)':<30s} {total_inference:>7.2f}s {total_per_frame:>9.1f}ms   {total_per_frame/40*100:>9.0f}%")
    print(f"")
    print(f"  Realtime ratio: {total_inference / video_duration:.2f}x")
    print(f"  Need {total_per_frame/40:.1f}x acceleration for realtime")
    print(f"")
    print(f"  Note: Audio encoding is one-time (not per-frame in streaming).")
    print(f"  The real per-frame cost = Motion Gen + Render = {ms_per_frame_motion + ms_per_frame_render:.1f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    benchmark()
