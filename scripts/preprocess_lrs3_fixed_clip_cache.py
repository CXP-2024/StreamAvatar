"""
Preprocess one fixed short clip per LRS3 video into DyStream motion-latent cache.

Unlike scripts/preprocess_lrs3_motion_cache.py, this script does not encode the
full video. For each mp4, it deterministically selects one fixed segment
(default: 3 seconds), crops that segment, encodes exactly T frames, and saves a
cache item. Existing training code then sees T == duration * fps, so each epoch
uses the same fixed segment instead of re-sampling a new start point.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_lrs3_motion_cache import (  # noqa: E402
    find_fixed_crop,
    safe_id,
    write_wav,
)
from scripts.smoke_online_crop_train import encode_motion_latents_batched  # noqa: E402


def list_videos(input_root, max_videos=None):
    videos = sorted(Path(input_root).rglob("*.mp4"))
    if max_videos is not None:
        videos = videos[: int(max_videos)]
    if not videos:
        raise RuntimeError(f"no mp4 files found under {input_root}")
    return videos


def deterministic_start(video_id, max_start, seed):
    if max_start <= 0:
        return 0
    key = f"{seed}:{video_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha1(key).digest()[:8], "little")
    return value % (max_start + 1)


def choose_start(video_id, total_pose_frames, target_frames, mode, seed):
    max_start = max(0, total_pose_frames - target_frames)
    if mode == "first":
        return 0
    if mode == "middle":
        return max_start // 2
    if mode == "random":
        return deterministic_start(video_id, max_start, seed)
    raise ValueError(f"unsupported start mode: {mode}")


def read_video_segment(video_path, video_id, pose_fps, target_frames, start_mode, seed):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or pose_fps
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_pose_frames = max(1, int(total_src_frames / max(src_fps, 1e-6) * pose_fps))
    pose_start = choose_start(video_id, total_pose_frames, target_frames, start_mode, seed)
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
    if len(frames) < target_frames:
        frames = frames + [frames[-1]] * (target_frames - len(frames))
    return frames[:target_frames], pose_start, float(src_fps), int(total_pose_frames)


def read_audio_segment(video_path, pose_start, target_frames, pose_fps, audio_sr):
    audio, _ = librosa.load(str(video_path), sr=audio_sr, mono=True)
    hop = audio_sr // pose_fps
    sample_start = pose_start * hop
    sample_len = target_frames * hop
    segment = audio[sample_start : sample_start + sample_len].astype(np.float32, copy=False)
    if segment.shape[0] < sample_len:
        segment = np.pad(segment, (0, sample_len - segment.shape[0]))
    return segment.astype(np.float32, copy=False)


def make_item(video_path, input_root, split_dir, split, args):
    video_id = safe_id(video_path, input_root)
    pt_path = split_dir / f"{video_id}.pt"
    npz_path = split_dir / f"{video_id}.npz"
    wav_path = split_dir / f"{video_id}.wav"
    if pt_path.exists() and npz_path.exists() and wav_path.exists() and not args.overwrite:
        return {
            "status": "cached",
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
            "end_idx": int(round(args.duration_sec * args.pose_fps)),
            "frames": int(round(args.duration_sec * args.pose_fps)),
            "pose_fps": int(args.pose_fps),
            "audio_sr": int(args.audio_sr),
            "fixed_clip": True,
        }, None

    frames, pose_start, src_fps, total_pose_frames = read_video_segment(
        video_path,
        video_id,
        args.pose_fps,
        int(round(args.duration_sec * args.pose_fps)),
        args.start_mode,
        args.seed,
    )
    crop_info = find_fixed_crop(frames, args.union_bbox_scale)
    audio = read_audio_segment(video_path, pose_start, len(frames), args.pose_fps, args.audio_sr)
    return None, {
        "video_id": video_id,
        "video_path": video_path,
        "pt_path": pt_path,
        "npz_path": npz_path,
        "wav_path": wav_path,
        "frames": frames,
        "crop_info": crop_info,
        "audio": audio,
        "pose_start": int(pose_start),
        "src_fps": float(src_fps),
        "total_pose_frames": int(total_pose_frames),
    }


def save_pending(pending, motion_latents, input_root, split, args):
    items = []
    for item, motion in zip(pending, motion_latents):
        item["pt_path"].parent.mkdir(parents=True, exist_ok=True)
        audio = item["audio"]
        write_wav(item["wav_path"], audio, args.audio_sr)
        payload = {
            "video_id": item["video_id"],
            "video_path": str(item["video_path"]),
            "audio": torch.from_numpy(audio),
            "audio_sr": int(args.audio_sr),
            "pose_fps": int(args.pose_fps),
            "src_fps": float(item["src_fps"]),
            "motion_latent": motion.float(),
            "num_frames": int(motion.shape[0]),
            "crop_mode": "fixed_clip",
            "crop_info": item["crop_info"],
            "fixed_clip": True,
            "fixed_clip_duration_sec": float(args.duration_sec),
            "fixed_clip_start_mode": args.start_mode,
            "fixed_clip_start_frame": int(item["pose_start"]),
            "total_pose_frames": int(item["total_pose_frames"]),
            "cache_version": "dystream_fixed_3s_v1",
        }
        torch.save(payload, item["pt_path"])
        np.savez_compressed(
            item["npz_path"],
            random_data=motion.numpy(),
            motion_latent=motion.numpy(),
        )
        items.append({
            "status": "ok",
            "mode": split,
            "dataset_type": "single",
            "video_id": item["video_id"],
            "video_path": str(item["video_path"]),
            "cache_path": str(item["pt_path"]),
            "motion_self_path": str(item["npz_path"]),
            "motion_other_path": None,
            "audio_self_path": str(item["wav_path"]),
            "audio_other_path": None,
            "start_idx": 0,
            "end_idx": int(motion.shape[0]),
            "frames": int(motion.shape[0]),
            "pose_fps": int(args.pose_fps),
            "audio_sr": int(args.audio_sr),
            "fixed_clip": True,
            "fixed_clip_start_frame": int(item["pose_start"]),
        })
    return items


def flush_pending(pending, args):
    if not pending:
        return []
    motion_latents = encode_motion_latents_batched(
        [item["frames"] for item in pending],
        [item["crop_info"] for item in pending],
        args.motion_batch_size,
    )
    return save_pending(pending, motion_latents, args.input_root, args.split, args)


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
        "fixed_clip_start_frame": cache.get("fixed_clip_start_frame"),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Preprocess one fixed short clip per LRS3 video")
    parser.add_argument("--input-root", default="/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/pretrain/pretrain")
    parser.add_argument("--output-dir", default="data_cache/lrs3_dystream_motion_fixed3s")
    parser.add_argument("--split", default="pretrain_fixed3s")
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--start-mode", choices=["random", "first", "middle"], default="random")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--pose-fps", type=int, default=25)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--motion-batch-size", type=int, default=512)
    parser.add_argument("--union-bbox-scale", type=float, default=1.6)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", default=None)
    args = parser.parse_args()

    if args.verify_only:
        verify_manifest(args.verify_only)
        return

    import app

    app.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    split_dir = out_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    videos = list_videos(args.input_root, args.max_videos)

    manifest = []
    failures = []
    pending = []
    start_time = time.perf_counter()
    pbar = tqdm(videos, desc=f"fixed clip cache {args.split}", dynamic_ncols=True)
    # Gradio monkey-patches tqdm in this environment and expects this attr.
    pbar._progress = None
    for video_path in pbar:
        try:
            cached_item, pending_item = make_item(video_path, args.input_root, split_dir, args.split, args)
            if cached_item is not None:
                manifest.append(cached_item)
            else:
                pending.append(pending_item)
            if len(pending) >= args.sample_batch_size:
                manifest.extend(flush_pending(pending, args))
                pending = []
            pbar.set_postfix({"ok": len(manifest), "fail": len(failures)})
        except Exception as exc:
            failures.append({"video_path": str(video_path), "error": repr(exc)})
            pending = []
    if pending:
        try:
            manifest.extend(flush_pending(pending, args))
        except Exception as exc:
            failures.append({"video_path": "pending_batch", "error": repr(exc)})

    manifest_path = out_dir / f"manifest_{args.split}.json"
    train_manifest_path = out_dir / f"manifest_{args.split}_train.json"
    val_manifest_path = out_dir / f"manifest_{args.split}_val{args.val_count}.json"
    fail_path = out_dir / f"failures_{args.split}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if args.val_count > 0 and len(manifest) > args.val_count:
        train_items = manifest[:-args.val_count]
        val_items = manifest[-args.val_count:]
        with open(train_manifest_path, "w", encoding="utf-8") as f:
            json.dump(train_items, f, indent=2)
        with open(val_manifest_path, "w", encoding="utf-8") as f:
            json.dump(val_items, f, indent=2)
    else:
        train_items = manifest
        val_items = []
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    elapsed = time.perf_counter() - start_time
    print(json.dumps({
        "status": "done",
        "input_root": args.input_root,
        "output_dir": args.output_dir,
        "split": args.split,
        "duration_sec": args.duration_sec,
        "start_mode": args.start_mode,
        "processed": len(manifest),
        "train": len(train_items),
        "val": len(val_items),
        "failed": len(failures),
        "elapsed_sec": elapsed,
        "clips_per_hour": len(manifest) / max(elapsed / 3600.0, 1e-9),
        "manifest": str(manifest_path),
        "train_manifest": str(train_manifest_path) if val_items else None,
        "val_manifest": str(val_manifest_path) if val_items else None,
        "failures": str(fail_path),
    }, indent=2))


if __name__ == "__main__":
    main()
