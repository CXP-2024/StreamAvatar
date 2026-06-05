"""
Replay cached DyStream motion latents and compare them with the source video.

This checks whether a real video encoded as motion_latent can be rendered back
through the visualization decoder with the first cropped frame as reference.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


def read_video_frames(video_path, pose_fps, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or pose_fps
    step = max(src_fps / float(pose_fps), 1.0)
    frames = []
    frame_idx = 0
    next_keep = 0.0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx + 1e-6 >= next_keep:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            next_keep += step
            if max_frames and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {video_path}")
    return frames


def crop_and_resize(frame, crop_info):
    if crop_info:
        frame = app.crop_from_bbox(
            frame,
            crop_info["center"],
            crop_info["bbox"],
            size=crop_info["size"],
        )
    return np.array(Image.fromarray(frame).convert("RGB").resize((512, 512), Image.BICUBIC))


def repeat_batch(value, batch_size):
    if torch.is_tensor(value):
        return value.repeat(batch_size, *([1] * (value.dim() - 1)))
    if isinstance(value, (list, tuple)):
        return type(value)(repeat_batch(v, batch_size) for v in value)
    if isinstance(value, dict):
        return {k: repeat_batch(v, batch_size) for k, v in value.items()}
    return value


@torch.inference_mode()
def render_latents(motion_latents, ref_image, batch_size):
    app.load_visualization_model()
    transform = app._vis_ctx["transform"]
    face_encoder = app._vis_ctx["face_encoder"]
    flow_estimator = app._vis_ctx["flow_estimator"]
    face_generator = app._vis_ctx["face_generator"]

    ref_tensor = transform(ref_image.convert("RGB")).unsqueeze(0).to(app.DEVICE)
    motion_latents = motion_latents.to(app.DEVICE).float()
    source = motion_latents[:1]

    face_feat = face_encoder(ref_tensor)
    chunks = []
    for start in range(0, motion_latents.shape[0], batch_size):
        target = motion_latents[start : start + batch_size]
        bsz = target.shape[0]
        flow = flow_estimator(source.repeat(bsz, 1), target)
        recon = face_generator(flow, repeat_batch(face_feat, bsz))
        chunks.append(recon.detach().cpu())

    video = torch.cat(chunks, dim=0).float().permute(0, 2, 3, 1).numpy()
    return np.clip((video + 1.0) * 127.5, 0, 255).astype(np.uint8)


def draw_label(frame, label):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def mux_audio(video_path, audio_path, output_path):
    if not audio_path or not os.path.exists(audio_path):
        return video_path
    try:
        import moviepy.editor as mpe

        clip = mpe.VideoFileClip(str(video_path))
        audio = mpe.AudioFileClip(str(audio_path))
        if audio.duration > clip.duration:
            audio = audio.subclip(0, clip.duration)
        clip = clip.set_audio(audio)
        clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        audio.close()
        return output_path
    except Exception as exc:
        print(f"[warn] failed to mux audio: {exc}")
        return video_path


def resolve_source_video(item, cache, lrs3_root):
    video_path = item.get("video_path") or cache.get("video_path")
    if video_path and os.path.exists(video_path):
        return video_path

    video_id = item.get("video_id") or cache.get("video_id")
    if video_id and "__" in video_id:
        candidate = Path(lrs3_root).joinpath(*video_id.split("__")).with_suffix(".mp4")
        if candidate.exists():
            return str(candidate)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/motion_cache_reconstruction")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--render-batch-size", type=int, default=16)
    parser.add_argument(
        "--lrs3-root",
        default="/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/trainval/trainval",
    )
    args = parser.parse_args()

    items = json.load(open(args.manifest, "r", encoding="utf-8"))
    if not items:
        raise RuntimeError(f"empty manifest: {args.manifest}")
    item = items[args.index % len(items)]
    cache = torch.load(item["cache_path"], map_location="cpu")

    video_path = resolve_source_video(item, cache, args.lrs3_root)
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(f"source video not found for item: {item.get('video_id')}")

    pose_fps = int(cache.get("pose_fps", item.get("pose_fps", 25)))
    motion_latent = cache["motion_latent"].float()
    n_frames = min(args.max_frames, motion_latent.shape[0])
    motion_latent = motion_latent[:n_frames]

    frames = read_video_frames(video_path, pose_fps, n_frames)
    frames = frames[:n_frames]
    crop_info = cache.get("crop_info")
    source_frames = np.stack([crop_and_resize(frame, crop_info) for frame in frames], axis=0)

    ref_image = Image.fromarray(source_frames[0])
    recon_frames = render_latents(motion_latent, ref_image, args.render_batch_size)
    n_frames = min(len(source_frames), len(recon_frames))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{item.get('video_id', args.index)}_{n_frames}f"
    comparison_no_audio = out_dir / f"{stem}_comparison_no_audio.mp4"
    comparison_audio = out_dir / f"{stem}_comparison.mp4"
    ref_path = out_dir / f"{stem}_reference.png"

    Image.fromarray(source_frames[0]).save(ref_path)
    with imageio.get_writer(comparison_no_audio, fps=pose_fps, codec="libx264") as writer:
        for src, rec in zip(source_frames[:n_frames], recon_frames[:n_frames]):
            pair = np.concatenate(
                [draw_label(src, "source crop"), draw_label(rec, "motion latent replay")],
                axis=1,
            )
            writer.append_data(pair)

    audio_path = item.get("audio_self_path")
    final_video = mux_audio(comparison_no_audio, audio_path, comparison_audio)
    print(json.dumps({
        "status": "ok",
        "video_id": item.get("video_id"),
        "source_video": video_path,
        "frames": n_frames,
        "pose_fps": pose_fps,
        "reference_image": str(ref_path),
        "comparison_video": str(final_video),
        "no_audio_video": str(comparison_no_audio),
    }, indent=2))


if __name__ == "__main__":
    main()
