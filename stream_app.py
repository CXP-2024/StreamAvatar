"""
Minimal DyStream stream-style demo.

This is a small custom-input app for timing the current inference pipeline on a
single reference image and one audio file. It intentionally reuses app.py's
preprocessing, model loading, rendering, and video muxing utilities.
"""

import argparse
import hashlib
import os
import tempfile
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr
import librosa
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

import app as base

_REF_CACHE = {}
_REF_CACHE_LIMIT = 8


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stamp():
    sync_cuda()
    return time.perf_counter()


def fmt_timings(timings, frames, pose_fps):
    duration = frames / pose_fps if pose_fps > 0 else 0.0
    total = timings.get("total", 0.0)
    rt = total / duration if duration > 0 else 0.0
    fps = frames / total if total > 0 else 0.0

    lines = [
        "| Stage | Time |",
        "|---|---:|",
    ]
    for key in [
        "load_models",
        "image_preprocess",
        "audio_prepare",
        "motion_inference",
        "render",
        "mux",
        "total",
    ]:
        lines.append(f"| {key} | {timings.get(key, 0.0):.3f}s |")
    if "motion_gpt" in timings:
        lines += [
            "| motion_gpt | " + f"{timings.get('motion_gpt', 0.0):.3f}s |",
            "| motion_fm | " + f"{timings.get('motion_fm', 0.0):.3f}s |",
            "| motion_audio_encoder | " + f"{timings.get('motion_audio_encoder', 0.0):.3f}s |",
            "| motion_other | " + f"{timings.get('motion_other', 0.0):.3f}s |",
        ]
    lines += [
        "",
        f"Reference cache: `{'hit' if timings.get('ref_cache_hit') else 'miss'}`",
        f"Guidance mode: `{timings.get('guidance_mode', 'full_5way')}`",
        f"Motion chunk frames: `{timings.get('motion_chunk_frames', 1)}`",
        f"Mux audio: `{'on' if timings.get('mux_audio') else 'off'}`",
        f"Frames: `{frames}`",
        f"Video duration: `{duration:.2f}s`",
        f"Pipeline FPS: `{fps:.2f}`",
        f"Realtime ratio: `{rt:.2f}x`",
    ]
    return "\n".join(lines)


def image_cache_key(image_pil):
    image = image_pil.convert("RGB")
    h = hashlib.sha1()
    h.update(str(image.size).encode("utf-8"))
    h.update(image.tobytes())
    return h.hexdigest()


def preprocess_reference_cached(image_pil):
    key = image_cache_key(image_pil)
    cached = _REF_CACHE.get(key)
    if cached is not None:
        resized_pil, masked_pil, motion_latent_cpu = cached
        return resized_pil.copy(), masked_pil.copy(), motion_latent_cpu.clone(), True

    resized_pil, masked_pil, motion_latent_cpu = base.process_image(image_pil)
    if len(_REF_CACHE) >= _REF_CACHE_LIMIT:
        _REF_CACHE.pop(next(iter(_REF_CACHE)))
    _REF_CACHE[key] = (resized_pil.copy(), masked_pil.copy(), motion_latent_cpu.cpu().clone())
    return resized_pil, masked_pil, motion_latent_cpu, False


def repeat_batch(value, batch_size):
    if torch.is_tensor(value):
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.repeat(batch_size, *([1] * (value.dim() - 1)))
        return value[:batch_size]
    if isinstance(value, tuple):
        return tuple(repeat_batch(v, batch_size) for v in value)
    if isinstance(value, list):
        return [repeat_batch(v, batch_size) for v in value]
    if isinstance(value, dict):
        return {k: repeat_batch(v, batch_size) for k, v in value.items()}
    return value


@torch.no_grad()
def latents_to_video_frames_batched(motion_latents, ref_image_pil, render_batch_size=16):
    base.load_visualization_model()

    transform = base._vis_ctx["transform"]
    face_encoder = base._vis_ctx["face_encoder"]
    flow_estimator = base._vis_ctx["flow_estimator"]
    face_generator = base._vis_ctx["face_generator"]

    ref_img_tensor = transform(ref_image_pil.convert("RGB")).unsqueeze(0).to(base.DEVICE)
    if motion_latents.dim() == 3:
        motion_latents = motion_latents.squeeze(0)
    motion_latents = motion_latents.to(base.DEVICE).float()

    render_batch_size = max(1, int(render_batch_size))
    with torch.inference_mode():
        face_feat = face_encoder(ref_img_tensor)
        source = motion_latents[0:1]
        recon_chunks = []
        for start in range(0, motion_latents.shape[0], render_batch_size):
            target = motion_latents[start:start + render_batch_size]
            batch_size = target.shape[0]
            source_batch = source.repeat(batch_size, 1)
            face_feat_batch = repeat_batch(face_feat, batch_size)
            tgt = flow_estimator(source_batch, target)
            recon_chunks.append(face_generator(tgt, face_feat_batch).detach().cpu())

    recon = torch.cat(recon_chunks, dim=0).float()
    video_np = recon.permute(0, 2, 3, 1).numpy()
    return np.clip((video_np + 1) / 2 * 255, 0, 255).astype("uint8")


@torch.no_grad()
def run_stream_lite(
    image_input,
    audio_path,
    denoising_steps,
    cfg_audio,
    cfg_audio_other,
    cfg_anchor,
    cfg_all,
    render_batch_size,
    guidance_mode,
    motion_chunk_frames,
    mux_audio,
    profile_motion,
    progress=gr.Progress(track_tqdm=True),
):
    if image_input is None:
        raise gr.Error("Please upload a reference image.")
    if audio_path is None:
        raise gr.Error("Please upload an audio/video file containing audio.")

    timings = {}
    timings["guidance_mode"] = guidance_mode
    timings["motion_chunk_frames"] = int(motion_chunk_frames)
    timings["mux_audio"] = bool(mux_audio)
    t_total = stamp()

    progress(0.02, desc="Loading models")
    t0 = stamp()
    base.load_dystream_model()
    base.load_visualization_model()
    timings["load_models"] = stamp() - t0

    progress(0.12, desc="Processing image")
    t0 = stamp()
    image_pil = image_input if isinstance(image_input, Image.Image) else Image.fromarray(image_input)
    resized_pil, masked_pil, motion_latent_cpu, ref_cache_hit = preprocess_reference_cached(image_pil)
    timings["image_preprocess"] = stamp() - t0
    timings["ref_cache_hit"] = ref_cache_hit

    progress(0.24, desc="Preparing audio")
    t0 = stamp()
    cfg = base._dystream_cfg
    model = base._dystream_model
    audio_sr = int(OmegaConf.select(cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(OmegaConf.select(cfg.config, "model.pose_fps", default=25))
    hop = int(audio_sr / pose_fps)

    audio_np, _ = librosa.load(audio_path, sr=audio_sr)
    prefix_frames = model.inpainting_length
    audio_np = np.concatenate([np.zeros(prefix_frames * hop, dtype=np.float32), audio_np.astype(np.float32)])
    audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(base.DEVICE)
    audio_other = torch.zeros_like(audio)
    timings["audio_prepare"] = stamp() - t0

    progress(0.34, desc="Generating motion")
    t0 = stamp()
    motion_latent = motion_latent_cpu.to(base.DEVICE)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.unsqueeze(0)
    if motion_latent.dim() == 2:
        motion_latent = motion_latent.unsqueeze(0)

    total_frames = audio.shape[1] // hop
    motion_in = motion_latent[:, 0:1, :].repeat(1, total_frames, 1)

    model.cfg_audio = float(cfg_audio)
    model.cfg_audio_other = float(cfg_audio_other)
    model.cfg_anchor = float(cfg_anchor)
    model.cfg_all = float(cfg_all)

    if base._dystream_ema is not None:
        base._dystream_ema.to(base.DEVICE)
        ema_ctx = base._dystream_ema.average_parameters(model.parameters())
    else:
        from contextlib import nullcontext
        ema_ctx = nullcontext()

    motion_profile = {} if profile_motion else None
    with ema_ctx:
        motion_pred = model.inference(
            audio,
            audio_other=audio_other,
            init_motion=motion_in,
            cond_motion=motion_in,
            anchor_motion=motion_latent[:, 0:1, :],
            noise_scheduler=base._noise_scheduler,
            num_inference_steps=int(denoising_steps),
            profile=motion_profile,
            guidance_mode=guidance_mode,
            stream_stride=int(motion_chunk_frames),
        )
    motion_pred = motion_pred[:, prefix_frames:]
    timings["motion_inference"] = stamp() - t0
    if motion_profile is not None:
        timings["motion_gpt"] = motion_profile.get("gpt", 0.0)
        timings["motion_fm"] = motion_profile.get("fm", 0.0)
        timings["motion_audio_encoder"] = motion_profile.get("audio_encoder", 0.0)
        timings["motion_other"] = max(
            0.0,
            timings["motion_inference"]
            - timings["motion_gpt"]
            - timings["motion_fm"]
            - timings["motion_audio_encoder"],
        )

    progress(0.70, desc="Rendering")
    t0 = stamp()
    frames = latents_to_video_frames_batched(motion_pred, resized_pil, render_batch_size=render_batch_size)
    timings["render"] = stamp() - t0

    progress(0.88, desc="Writing video")
    t0 = stamp()
    output_dir = tempfile.mkdtemp(prefix="dystream_stream_")
    output_path = os.path.join(output_dir, "output.mp4")
    final_audio_path = audio_path if mux_audio else None
    base.save_video_with_audio(frames, final_audio_path, output_path, fps=pose_fps)
    timings["mux"] = stamp() - t0

    timings["total"] = stamp() - t_total
    progress(1.0, desc="Done")
    return output_path, resized_pil, masked_pil, fmt_timings(timings, motion_pred.shape[1], pose_fps)


def build_ui():
    with gr.Blocks(title="DyStream Stream Lite") as demo:
        gr.Markdown("# DyStream Stream Lite")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Reference image", type="pil")
                audio = gr.File(label="Audio or video with audio", file_types=[".wav", ".mp3", ".m4a", ".mov", ".mp4"], type="filepath")
                denoising_steps = gr.Slider(1, 8, value=1, step=1, label="Denoising steps")
                with gr.Row():
                    cfg_audio = gr.Slider(0, 3, value=0.5, step=0.1, label="CFG audio")
                    cfg_audio_other = gr.Slider(0, 3, value=0.5, step=0.1, label="CFG other")
                with gr.Row():
                    cfg_anchor = gr.Slider(0, 3, value=0.0, step=0.1, label="CFG anchor")
                    cfg_all = gr.Slider(0, 3, value=1.0, step=0.1, label="CFG all")
                render_batch_size = gr.Slider(1, 64, value=16, step=1, label="Render batch size")
                guidance_mode = gr.Radio(
                    choices=["full_5way", "uncond_all_2way", "all_only"],
                    value="full_5way",
                    label="Guidance mode",
                )
                motion_chunk_frames = gr.Slider(1, 8, value=1, step=1, label="Motion chunk frames")
                mux_audio = gr.Checkbox(value=True, label="Mux audio into output")
                profile_motion = gr.Checkbox(value=True, label="Profile motion breakdown")
                run_btn = gr.Button("Run", variant="primary")
            with gr.Column():
                video = gr.Video(label="Output")
                timings = gr.Markdown(label="Timings")
                with gr.Row():
                    resized = gr.Image(label="Preprocessed image")
                    masked = gr.Image(label="Masked image")

        run_btn.click(
            fn=run_stream_lite,
            inputs=[
                image,
                audio,
                denoising_steps,
                cfg_audio,
                cfg_audio_other,
                cfg_anchor,
                cfg_all,
                render_batch_size,
                guidance_mode,
                motion_chunk_frames,
                mux_audio,
                profile_motion,
            ],
            outputs=[video, resized, masked, timings],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    for var in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(var, "")
        if "localhost" not in cur:
            os.environ[var] = f"localhost,127.0.0.1,{cur}" if cur else "localhost,127.0.0.1"

    demo = build_ui()
    demo.queue()
    print(f"[StreamLite] Launching on http://{args.host}:{args.port}", flush=True)
    demo.launch(server_name=args.host, server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
