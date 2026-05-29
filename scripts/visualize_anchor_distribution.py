import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def resolve_path(path, manifest_path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return manifest_path.parent / path


def load_real_anchors(manifest_path, sample_count, seed):
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    items = [
        item for item in items
        if item.get("status") in {"ok", "cached"} and item.get("cache_path")
    ]
    rng = random.Random(seed)
    if sample_count < len(items):
        items = rng.sample(items, sample_count)

    anchors = []
    used = []
    for item in items:
        cache_path = resolve_path(item["cache_path"], manifest_path)
        cache = torch.load(cache_path, map_location="cpu")
        motion = cache["motion_latent"].float()
        if motion.ndim != 2 or motion.shape[-1] != 512:
            continue
        anchors.append(motion[0].numpy())
        used.append(item.get("video_id", cache_path.stem))
    if not anchors:
        raise RuntimeError(f"no usable anchors loaded from {manifest_path}")
    return np.stack(anchors, axis=0).astype(np.float32), used


def plot_embedding(points, labels, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = np.asarray(labels)
    for name, color in [("real_cache_anchor", "tab:orange"), ("random_gaussian_anchor", "tab:blue")]:
        mask = labels == name
        ax.scatter(points[mask, 0], points[mask, 1], s=8, alpha=0.45, label=name, c=color)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_norms(real, random_anchor, path):
    real_norm = np.linalg.norm(real, axis=1)
    random_norm = np.linalg.norm(random_anchor, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_norm, bins=60, alpha=0.55, density=True, label="real_cache_anchor")
    ax.hist(random_norm, bins=60, alpha=0.55, density=True, label="random_gaussian_anchor")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("density")
    ax.set_title("Anchor Norm Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def vector_stats(x):
    norms = np.linalg.norm(x, axis=1)
    return {
        "count": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "p05_norm": float(np.quantile(norms, 0.05)),
        "p50_norm": float(np.quantile(norms, 0.50)),
        "p95_norm": float(np.quantile(norms, 0.95)),
        "mean_abs_value": float(np.abs(x).mean()),
        "global_std": float(x.std()),
        "mean_dim_std": float(x.std(axis=0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_cache/lrs3_dystream_motion_volc/manifest_pretrain_60k_train.json")
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--anchor-std", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--output-dir", default="outputs/anchor_distribution_pretrain60k")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real, used = load_real_anchors(manifest_path, args.sample_count, args.seed)
    rng = np.random.default_rng(args.seed)
    random_anchor = (rng.standard_normal(real.shape).astype(np.float32) * args.anchor_std)

    labels = np.array(["real_cache_anchor"] * len(real) + ["random_gaussian_anchor"] * len(random_anchor))
    combined = np.concatenate([real, random_anchor], axis=0)

    pca50 = PCA(n_components=min(50, combined.shape[1]), random_state=args.seed)
    reduced = pca50.fit_transform(combined)

    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(combined)
    plot_embedding(pca2, labels, "PCA of Real Cache Anchors vs Random Gaussian Anchors", output_dir / "anchor_pca.png")

    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, max(5, (combined.shape[0] - 1) / 3)),
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        max_iter=1000,
        verbose=1,
    )
    tsne_points = tsne.fit_transform(reduced)
    plot_embedding(tsne_points, labels, "t-SNE of Real Cache Anchors vs Random Gaussian Anchors", output_dir / "anchor_tsne.png")
    plot_norms(real, random_anchor, output_dir / "anchor_norm_hist.png")

    summary = {
        "manifest": str(manifest_path),
        "sample_count": int(real.shape[0]),
        "anchor_std_for_random": float(args.anchor_std),
        "real_cache_anchor": vector_stats(real),
        "random_gaussian_anchor": vector_stats(random_anchor),
        "pca_explained_variance_ratio_first10": [float(x) for x in pca50.explained_variance_ratio_[:10]],
        "outputs": {
            "pca": str(output_dir / "anchor_pca.png"),
            "tsne": str(output_dir / "anchor_tsne.png"),
            "norm_hist": str(output_dir / "anchor_norm_hist.png"),
            "summary": str(output_dir / "anchor_distribution_summary.json"),
        },
        "first_used_video_ids": used[:10],
    }
    with open(output_dir / "anchor_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
