"""
Plot block-wise distillation training metrics.

Usage:
  .venv/bin/python plot_blockwise_metrics.py \
    --metrics outputs/blockwise_stream_distill/metrics.jsonl
"""

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"))

import matplotlib.pyplot as plt


def load_metrics(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("event") == "train":
                rows.append(item)
    if not rows:
        raise ValueError(f"No train events found in {path}")
    return rows


def smooth(values, window):
    if window <= 1:
        return values
    out = []
    acc = 0.0
    queue = []
    for value in values:
        queue.append(value)
        acc += value
        if len(queue) > window:
            acc -= queue.pop(0)
        out.append(acc / len(queue))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot block-wise distillation metrics")
    parser.add_argument("--metrics", default="outputs/blockwise_stream_distill/metrics.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth", type=int, default=50)
    args = parser.parse_args()

    rows = load_metrics(args.metrics)
    steps = [row["step"] for row in rows]
    output = args.output or os.path.join(os.path.dirname(args.metrics), "loss_curves.png")

    curves = [
        ("loss", "total"),
        ("loss_motion", "motion"),
        ("loss_velocity", "velocity"),
        ("loss_acceleration", "acceleration"),
        ("loss_boundary", "boundary"),
    ]

    plt.figure(figsize=(12, 7))
    for key, label in curves:
        values = [row[key] for row in rows if key in row]
        if len(values) != len(steps):
            continue
        plt.plot(steps, smooth(values, args.smooth), label=label)

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Block-wise Streaming Distillation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
