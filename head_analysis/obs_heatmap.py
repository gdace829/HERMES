"""Offline plots for Observation 1: heterogeneous temporal head preferences.

This script only reads an existing ``head_pseudo.npz`` or ``head_scores.npz``
file and writes figures/statistics under ``results/observations``.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_score(path):
    data = np.load(path)
    keys = set(data.files)

    if "shift" in keys:
        score = np.nan_to_num(data["shift"].astype(np.float64))
        panels = [
            ("Local-query early ratio", np.nan_to_num(data["local_early"].astype(np.float64))),
            ("Global-query early ratio", np.nan_to_num(data["global_early"].astype(np.float64))),
            ("Temporal shift\n(global early - local early)", score),
        ]
        return score, panels, "pseudo_shift"

    if "recent_A" in keys and "recent_B" in keys:
        recent_a = np.nan_to_num(data["recent_A"].astype(np.float64))
        recent_b = np.nan_to_num(data["recent_B"].astype(np.float64))
        score = recent_a - recent_b
        panels = [
            ("Recent ratio on memory tasks", recent_a),
            ("Recent ratio on recent tasks", recent_b),
            ("Task-conditioned temporal preference", score),
        ]
        return score, panels, "task_recent_ratio"

    raise ValueError(f"Unsupported score keys: {sorted(keys)}")


def top_heads(score, largest=True, k=12):
    flat = [
        (layer, head, float(score[layer, head]))
        for layer in range(score.shape[0])
        for head in range(score.shape[1])
    ]
    flat.sort(key=lambda item: item[2], reverse=largest)
    return [{"layer": l, "head": h, "score": s} for l, h, s in flat[:k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        default="results/head_analysis/pseudo-qwen2.5_vl_7b-kv6000-hermes/head_pseudo.npz",
    )
    parser.add_argument("--save_dir", default="results/observations/obs1_heatmap")
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    score, panels, score_type = load_score(args.scores)
    num_layers, num_heads = score.shape

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, matrix) in zip(axes, panels):
        symmetric = "shift" in title.lower() or "preference" in title.lower()
        if symmetric:
            vmax = max(float(np.abs(matrix).max()), 1e-6)
            vmin = -vmax
        else:
            vmin = float(matrix.min())
            vmax = float(matrix.max())
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1e-6
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Observation 1: Head-Level Temporal Preference", fontsize=13)
    fig.tight_layout()
    heatmap_path = os.path.join(args.save_dir, "temporal_head_heatmap.png")
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    layer_std = score.std(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(num_layers), score.mean(axis=1), marker="o", linewidth=1)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Layer mean temporal score")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean score")
    axes[0].grid(alpha=0.25)
    axes[1].bar(np.arange(num_layers), layer_std)
    axes[1].set_title("Within-layer head diversity")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Std. across heads")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    diversity_path = os.path.join(args.save_dir, "within_layer_diversity.png")
    fig.savefig(diversity_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    threshold = 0.1 * max(float(np.abs(score).max()), 1e-6)
    summary = {
        "scores": args.scores,
        "score_type": score_type,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "specialization_threshold_abs": float(threshold),
        "num_positive_specialized": int((score > threshold).sum()),
        "num_negative_specialized": int((score < -threshold).sum()),
        "mean_within_layer_std": float(layer_std.mean()),
        "max_within_layer_std": float(layer_std.max()),
        "top_positive_heads": top_heads(score, largest=True, k=args.top_k),
        "top_negative_heads": top_heads(score, largest=False, k=args.top_k),
        "outputs": {
            "heatmap": heatmap_path,
            "within_layer_diversity": diversity_path,
        },
    }

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
