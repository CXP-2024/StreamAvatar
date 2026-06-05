#!/usr/bin/env python3
"""
Evaluate lip-sync with the original SyncNet model.

Outputs:
  - lse_d: minimum audio/video embedding distance, lower is better
  - lse_c: SyncNet confidence, higher is better
  - offset_frames: estimated AV offset from SyncNet

The script resizes each input video to 224x224 at 25fps before evaluation,
matching the public syncnet_python demo expectations.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def ffmpeg_prepare(input_video, output_video, duration=None):
    vf = "fps=25,scale=224:224:flags=bicubic"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_video),
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf",
        vf,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:v",
        "mpeg4",
        "-q:v",
        "2",
        "-c:a",
        "pcm_s16le",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def evaluate_one(syncnet_dir, model_path, video_path, batch_size, vshift, duration=None, device=None):
    syncnet_dir = Path(syncnet_dir).resolve()
    model_path = Path(model_path).resolve()
    video_path = Path(video_path).resolve()
    sys.path.insert(0, str(syncnet_dir))

    from SyncNetInstance import SyncNetInstance

    with tempfile.TemporaryDirectory(prefix="syncnet_eval_") as tmp:
        tmp = Path(tmp)
        prepared = tmp / "prepared.avi"
        ffmpeg_prepare(video_path, prepared, duration=duration)
        opt = SimpleNamespace(
            initial_model=str(model_path),
            batch_size=int(batch_size),
            vshift=int(vshift),
            videofile=str(prepared),
            tmp_dir=str(tmp / "work"),
            reference="sample",
        )
        model = SyncNetInstance(device=device)
        model.loadParameters(str(model_path))
        offset, conf, dists = model.evaluate(opt, videofile=str(prepared))
        lse_d = float(dists.mean(axis=0).min())
        return {
            "video": str(video_path),
            "prepared_resolution": "224x224",
            "prepared_fps": 25,
            "duration_limit_sec": duration,
            "offset_frames": int(offset),
            "offset_sec": float(offset / 25.0),
            "lse_d": lse_d,
            "lse_c": float(conf),
            "frames_scored": int(dists.shape[0]),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSE-D/LSE-C with SyncNet.")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--syncnet-dir", default="third_party/syncnet_python")
    parser.add_argument("--model", default="third_party/syncnet_python/data/syncnet_v2.model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--duration", type=float, default=None, help="Optional first N seconds for quick testing.")
    parser.add_argument("--device", default=None, help="cpu or cuda. Defaults to SyncNet auto-detect.")
    args = parser.parse_args()

    results = [
        evaluate_one(
            args.syncnet_dir,
            args.model,
            video,
            batch_size=args.batch_size,
            vshift=args.vshift,
            duration=args.duration,
            device=args.device,
        )
        for video in args.videos
    ]
    output = args.output
    if output is None:
        output = str(Path(args.videos[0]).resolve().parent / "syncnet_lse_metrics.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
