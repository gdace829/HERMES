"""Offline split-stability plot for Observation 2.

Input is the ``raw_obs.csv`` saved by ``analyze_heads_pseudo.py``. If the raw
file has no video/question identifiers, the script uses a random row split and
labels the result as a preliminary stability check.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def corr(x, y, spearman=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(df, num_layers=None, num_heads=None):
    if "shift" not in df.columns:
        df = df.copy()
        df["shift"] = df["global_early_ratio"] - df["local_early_ratio"]

    if num_layers is None:
        num_layers = int(df["layer"].max()) + 1
    if num_heads is None:
        num_heads = int(df["head"].max()) + 1

    sums = np.zeros((num_layers, num_heads), dtype=np.float64)
    counts = np.zeros((num_layers, num_heads), dtype=np.float64)
    for row in df.itertuples(index=False):
        layer = int(getattr(row, "layer"))
        head = int(getattr(row, "head"))
        value = float(getattr(row, "shift"))
        sums[layer, head] += value
        counts[layer, head] += 1
    avg = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return avg, counts


def split_df(df, seed):
    rng = np.random.default_rng(seed)
    if "video_id" in df.columns:
        ids = np.asarray(sorted(df["video_id"].astype(str).unique()))
        rng.shuffle(ids)
        left = set(ids[: len(ids) // 2])
        return df[df["video_id"].astype(str).isin(left)], df[~df["video_id"].astype(str).isin(left)], "video"
    if "question_id" in df.columns:
        ids = np.asarray(sorted(df["question_id"].astype(str).unique()))
        rng.shuffle(ids)
        left = set(ids[: len(ids) // 2])
        return df[df["question_id"].astype(str).isin(left)], df[~df["question_id"].astype(str).isin(left)], "question"
    mask = rng.random(len(df)) < 0.5
    return df[mask], df[~mask], "row_preliminary"


def top_overlap(a, b, k, largest=True):
    xa = a.reshape(-1)
    xb = b.reshape(-1)
    valid = np.where(np.isfinite(xa) & np.isfinite(xb))[0]
    if valid.size == 0:
        return float("nan")
    k = min(k, valid.size)
    order_a = valid[np.argsort(xa[valid])]
    order_b = valid[np.argsort(xb[valid])]
    if largest:
        order_a = order_a[::-1]
        order_b = order_b[::-1]
    return float(len(set(order_a[:k]) & set(order_b[:k])) / max(k, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--save_dir", default="results/observations/obs2_stability")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.raw_csv)
    left, right, split_mode = split_df(df, args.seed)
    score_a, count_a = aggregate(left)
    score_b, count_b = aggregate(right, score_a.shape[0], score_a.shape[1])

    x = score_a.reshape(-1)
    y = score_b.reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    pearson = corr(x[valid], y[valid], spearman=False)
    spearman = corr(x[valid], y[valid], spearman=True)
    top_pos = top_overlap(score_a, score_b, args.top_k, largest=True)
    top_neg = top_overlap(score_a, score_b, args.top_k, largest=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(x[valid], y[valid], s=14, alpha=0.65)
    if valid.any():
        lo = float(min(x[valid].min(), y[valid].min()))
        hi = float(max(x[valid].max(), y[valid].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Split A temporal shift")
    ax.set_ylabel("Split B temporal shift")
    ax.set_title(f"Observation 2: Split Stability ({split_mode})")
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"Pearson={pearson:.3f}\nSpearman={spearman:.3f}\nTop-{args.top_k}+ overlap={top_pos:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.tight_layout()
    plot_path = os.path.join(args.save_dir, "split_correlation.png")
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    npz_path = os.path.join(args.save_dir, "split_scores.npz")
    np.savez(npz_path, score_a=score_a, score_b=score_b, count_a=count_a, count_b=count_b)

    summary = {
        "raw_csv": args.raw_csv,
        "split_mode": split_mode,
        "rows_total": int(len(df)),
        "rows_split_a": int(len(left)),
        "rows_split_b": int(len(right)),
        "num_valid_heads": int(valid.sum()),
        "pearson": pearson,
        "spearman": spearman,
        "top_k": int(args.top_k),
        "top_positive_overlap": top_pos,
        "top_negative_overlap": top_neg,
        "outputs": {"plot": plot_path, "scores": npz_path},
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
