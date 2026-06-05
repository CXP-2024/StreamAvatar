#!/usr/bin/env python3
"""
Lightweight lip-sync/mouth-motion metrics from generated videos.

This is not a SyncNet/LSE replacement. It estimates whether visible mouth motion
tracks the audio envelope by extracting MediaPipe face landmarks and comparing a
normalized mouth-opening curve with frame-level audio RMS.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import mediapipe as mp
import numpy as np


MOUTH_UPPER_INNER = 13
MOUTH_LOWER_INNER = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1.0e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def extract_audio(video_path, sr):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        tmp.name,
    ]
    subprocess.run(cmd, check=True)
    audio, _ = librosa.load(tmp.name, sr=sr, mono=True)
    os.unlink(tmp.name)
    return audio


def audio_rms_per_frame(audio, sr, fps, n_frames):
    frame_len = max(1, int(round(sr / fps)))
    vals = []
    for i in range(n_frames):
        start = i * frame_len
        end = min(len(audio), start + frame_len)
        if start >= len(audio):
            vals.append(0.0)
            continue
        chunk = audio[start:end]
        vals.append(float(np.sqrt(np.mean(chunk * chunk) + 1.0e-12)))
    return np.asarray(vals, dtype=np.float32)


def mouth_curve(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        total = min(total, int(max_frames))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    openings = []
    valid = []
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            openings.append(np.nan)
            valid.append(False)
            frame_idx += 1
            continue
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def xy(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        upper = xy(MOUTH_UPPER_INNER)
        lower = xy(MOUTH_LOWER_INNER)
        left = xy(MOUTH_LEFT)
        right = xy(MOUTH_RIGHT)
        width = np.linalg.norm(right - left)
        opening = np.linalg.norm(lower - upper) / max(width, 1.0e-6)
        openings.append(float(opening))
        valid.append(True)
        frame_idx += 1

    cap.release()
    face_mesh.close()
    return np.asarray(openings, dtype=np.float32), np.asarray(valid, dtype=bool), float(fps)


def fill_nan(x):
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).all():
        return x
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good]).astype(np.float32)


def best_lag_corr(mouth, audio, max_lag):
    best = {"corr": float("nan"), "lag_frames": 0}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            m = mouth[:lag]
            a = audio[-lag:]
        elif lag > 0:
            m = mouth[lag:]
            a = audio[:-lag]
        else:
            m = mouth
            a = audio
        c = pearson(m, a)
        if np.isfinite(c) and (not np.isfinite(best["corr"]) or c > best["corr"]):
            best = {"corr": float(c), "lag_frames": int(lag)}
    return best


def evaluate_video(video_path, max_lag_frames=10, audio_sr=16000, max_frames=None):
    video_path = Path(video_path)
    opening, valid, fps = mouth_curve(video_path, max_frames=max_frames)
    opening = fill_nan(opening)
    audio = extract_audio(video_path, audio_sr)
    rms = audio_rms_per_frame(audio, audio_sr, fps, len(opening))

    # Use log RMS because speech loudness is roughly logarithmic and less
    # dominated by a few high-energy phonemes.
    log_rms = np.log(rms + 1.0e-5)
    mouth_vel = np.abs(np.diff(opening, prepend=opening[:1]))
    audio_vel = np.maximum(np.diff(log_rms, prepend=log_rms[:1]), 0.0)

    open_lag = best_lag_corr(opening, log_rms, max_lag_frames)
    vel_lag = best_lag_corr(mouth_vel, audio_vel, max_lag_frames)

    q20 = float(np.quantile(log_rms, 0.2))
    q60 = float(np.quantile(log_rms, 0.6))
    silent = log_rms <= q20
    speech = log_rms >= q60

    metrics = {
        "video": str(video_path),
        "frames": int(len(opening)),
        "fps": float(fps),
        "duration_sec": float(len(opening) / max(fps, 1.0e-6)),
        "face_detect_rate": float(valid.mean()) if len(valid) else 0.0,
        "mouth_open_mean": float(np.mean(opening)),
        "mouth_open_std": float(np.std(opening)),
        "mouth_velocity_mean": float(np.mean(mouth_vel)),
        "audio_log_rms_std": float(np.std(log_rms)),
        "mouth_audio_corr": float(open_lag["corr"]),
        "mouth_audio_lag_frames": int(open_lag["lag_frames"]),
        "mouth_audio_lag_sec": float(open_lag["lag_frames"] / max(fps, 1.0e-6)),
        "mouth_velocity_audio_onset_corr": float(vel_lag["corr"]),
        "mouth_velocity_audio_onset_lag_frames": int(vel_lag["lag_frames"]),
        "mouth_velocity_audio_onset_lag_sec": float(vel_lag["lag_frames"] / max(fps, 1.0e-6)),
        "silent_mouth_velocity_mean": float(np.mean(mouth_vel[silent])) if silent.any() else float("nan"),
        "speech_mouth_velocity_mean": float(np.mean(mouth_vel[speech])) if speech.any() else float("nan"),
        "speech_silent_velocity_ratio": float(
            (np.mean(mouth_vel[speech]) + 1.0e-8) / (np.mean(mouth_vel[silent]) + 1.0e-8)
        ) if silent.any() and speech.any() else float("nan"),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated video mouth/audio sync with landmarks.")
    parser.add_argument("videos", nargs="+", help="Video files with audio.")
    parser.add_argument("--output", default=None, help="JSON output path. Defaults to <first_video_dir>/lip_sync_metrics.json")
    parser.add_argument("--max-lag-frames", type=int, default=10)
    parser.add_argument("--audio-sr", type=int, default=16000)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    results = [
        evaluate_video(v, max_lag_frames=args.max_lag_frames, audio_sr=args.audio_sr, max_frames=args.max_frames)
        for v in args.videos
    ]
    output = args.output
    if output is None:
        output = str(Path(args.videos[0]).resolve().parent / "lip_sync_metrics.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
