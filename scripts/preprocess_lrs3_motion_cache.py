"""
Preprocess LRS3 videos into DyStream motion-latent caches.

Each processed clip produces:
  - cache/<split>/<video_id>.pt with audio and motion_latent tensors.
  - cache/<split>/<video_id>.npz with random_data for DyStream-style loaders.
  - cache/<split>/<video_id>.wav extracted mono audio.
  - manifest_<split>.json listing all successful clips.

The default crop mode uses a fixed crop from the first detected face, then
applies that crop to all frames. This avoids frame-by-frame crop jitter.
"""

import argparse
import json
import os
import sys
import wave
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def list_videos(input_root, max_videos=None):
    root = Path(input_root)
    videos = sorted(root.rglob("*.mp4"))
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    return videos


def safe_id(video_path, input_root):
    rel = Path(video_path).resolve().relative_to(Path(input_root).resolve())
    return "__".join(rel.with_suffix("").parts)


def write_wav(path, audio, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_i16.tobytes())
    return str(path)


def load_audio(video_path, sr):
    audio, _ = librosa.load(str(video_path), sr=sr, mono=True)
    return audio.astype(np.float32)


def read_video_frames(video_path, pose_fps=25, max_frames=None):
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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            next_keep += step
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {video_path}")
    return frames, float(src_fps)


def find_fixed_crop(frames, union_bbox_scale):
    import app

    app.load_face_detector()
    for frame in frames[: min(len(frames), 25)]:
        det_res = app._face_detector.get_face_xy_rotation_and_keypoints(
            frame,
            mouth_bbox_scale=1.0,
            eye_bbox_scale=1.0,
        )
        if not det_res or len(det_res[6]) == 0:
            continue
        face_bbox = det_res[5][0]
        x1, y1 = face_bbox[0]
        x2, y2 = face_bbox[1]
        center = [(y1 + y2) // 2, (x1 + x2) // 2]
        width = x2 - x1
        height = y2 - y1
        max_size = int(max(width, height) * union_bbox_scale)
        h, w = frame.shape[:2]
        crop_bbox = app.generate_crop_bounding_box(h, w, center, max_size)
        return {
            "center": center,
            "bbox": crop_bbox,
            "size": max_size,
            "scale": float(union_bbox_scale),
        }
    raise RuntimeError("no face detected in the first frames")


def apply_crop(frame, crop_info):
    import app

    if crop_info is None:
        return Image.fromarray(frame)
    cropped = app.crop_from_bbox(
        frame,
        crop_info["center"],
        crop_info["bbox"],
        size=crop_info["size"],
    )
    return Image.fromarray(cropped)


def encode_motion_latents(frames, crop_info, batch_size):
    import app

    app.load_visualization_model()
    transform = app._vis_ctx["transform"]
    motion_encoder = app._vis_ctx["motion_encoder"]
    latents = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            images = [apply_crop(frame, crop_info).convert("RGB") for frame in batch_frames]
            tensor = torch.stack([transform(image) for image in images], dim=0).to(app.DEVICE)
            out = motion_encoder(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            latents.append(out.detach().float().cpu())
    return torch.cat(latents, dim=0)


def process_one(video_path, input_root, output_dir, split, args):
    video_id = safe_id(video_path, input_root)
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pt_path = split_dir / f"{video_id}.pt"
    npz_path = split_dir / f"{video_id}.npz"
    wav_path = split_dir / f"{video_id}.wav"

    if pt_path.exists() and npz_path.exists() and wav_path.exists() and not args.overwrite:
        return {
            "status": "cached",
            "video_id": video_id,
            "cache_path": str(pt_path),
            "motion_self_path": str(npz_path),
            "audio_self_path": str(wav_path),
        }

    frames, src_fps = read_video_frames(video_path, pose_fps=args.pose_fps, max_frames=args.max_frames)
    if len(frames) < args.min_frames:
        raise RuntimeError(f"too few frames: {len(frames)} < {args.min_frames}")

    crop_info = None
    if args.crop_mode == "fixed_first":
        crop_info = find_fixed_crop(frames, args.union_bbox_scale)
    elif args.crop_mode != "none":
        raise ValueError(f"unsupported crop mode: {args.crop_mode}")

    motion_latent = encode_motion_latents(frames, crop_info, args.batch_size)
    audio = load_audio(video_path, args.audio_sr)
    write_wav(wav_path, audio, args.audio_sr)

    payload = {
        "video_id": video_id,
        "video_path": str(video_path),
        "audio": torch.from_numpy(audio),
        "audio_sr": int(args.audio_sr),
        "pose_fps": int(args.pose_fps),
        "src_fps": float(src_fps),
        "motion_latent": motion_latent,
        "num_frames": int(motion_latent.shape[0]),
        "crop_mode": args.crop_mode,
        "crop_info": crop_info,
    }
    torch.save(payload, pt_path)
    np.savez_compressed(npz_path, random_data=motion_latent.numpy(), motion_latent=motion_latent.numpy())

    return {
        "status": "ok",
        "mode": split,
        "dataset_type": "single",
        "video_id": video_id,
        "video_path": str(video_path),
        "cache_path": str(pt_path),
        "motion_self_path": str(npz_path),
        "motion_other_path": None,
        "audio_self_path": str(wav_path),
        "audio_other_path": None,
        "start_idx": 0,
        "end_idx": int(motion_latent.shape[0]),
        "frames": int(motion_latent.shape[0]),
        "pose_fps": int(args.pose_fps),
        "audio_sr": int(args.audio_sr),
    }


def verify_manifest(manifest_path):
    items = json.load(open(manifest_path, "r", encoding="utf-8"))
    if not items:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    item = items[0]
    cache = torch.load(item["cache_path"], map_location="cpu")
    motion_npz = np.load(item["motion_self_path"])
    audio, sr = librosa.load(item["audio_self_path"], sr=None, mono=True)
    print(json.dumps({
        "manifest": manifest_path,
        "items": len(items),
        "first_video_id": item["video_id"],
        "cache_motion_shape": list(cache["motion_latent"].shape),
        "cache_audio_shape": list(cache["audio"].shape),
        "npz_random_data_shape": list(motion_npz["random_data"].shape),
        "wav_samples": int(audio.shape[0]),
        "wav_sr": int(sr),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Preprocess LRS3 videos into DyStream motion-latent caches")
    parser.add_argument("--input-root", default="/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/trainval/trainval")
    parser.add_argument("--output-dir", default="data_cache/lrs3_dystream_motion")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--pose-fps", type=int, default=25)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-mode", choices=["fixed_first", "none"], default="fixed_first")
    parser.add_argument("--union-bbox-scale", type=float, default=1.6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", default=None, help="Verify an existing manifest and exit.")
    args = parser.parse_args()

    if args.verify_only:
        verify_manifest(args.verify_only)
        return

    import app

    app.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    videos = list_videos(args.input_root, args.max_videos)
    if not videos:
        raise RuntimeError(f"no mp4 files found under {args.input_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = []

    for video_path in tqdm(videos, desc=f"preprocess {args.split}", dynamic_ncols=True):
        try:
            item = process_one(video_path, args.input_root, args.output_dir, args.split, args)
            if item["status"] in {"ok", "cached"}:
                manifest.append(item)
        except Exception as exc:
            failures.append({"video_path": str(video_path), "error": repr(exc)})

    manifest_path = out_dir / f"manifest_{args.split}.json"
    fail_path = out_dir / f"failures_{args.split}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print(json.dumps({
        "status": "done",
        "input_root": args.input_root,
        "output_dir": args.output_dir,
        "split": args.split,
        "processed": len(manifest),
        "failed": len(failures),
        "manifest": str(manifest_path),
        "failures": str(fail_path),
    }, indent=2))


if __name__ == "__main__":
    main()
