"""Aggregate previous/current pseudo-query attention from query heads to KV heads.

The eager observer writes one row per layer-query-head. For GQA models such as
Qwen2.5-VL-7B, physical KV-cache storage is organized by KV heads, not query
heads. This script groups query heads by their shared KV head and recomputes
previous-memory vs latest-chunk metrics from attention mass.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.generate_prev_current_profile_artifacts import (
    EPS,
    _corr,
    _quantile,
    save_heatmap,
    save_hist,
    save_scatter,
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _finite_mean(values):
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(vals.mean()) if vals.size else None


def _finite_median(values):
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(vals)) if vals.size else None


def _matrix_from_kv(kv_df, value_col):
    n_layers = int(kv_df["layer"].max()) + 1
    n_kv_heads = int(kv_df["kv_head"].max()) + 1
    arr = np.full((n_layers, n_kv_heads), np.nan, dtype=np.float64)
    for row in kv_df.itertuples(index=False):
        arr[int(row.layer), int(row.kv_head)] = float(getattr(row, value_col))
    return arr


def _chunk_half_split(df):
    if not {"video_idx", "chunk_idx"}.issubset(df.columns):
        return None
    flags = []
    medians = df.groupby("video_idx")["chunk_idx"].median().to_dict()
    for row in df.itertuples(index=False):
        flags.append("early" if int(row.chunk_idx) <= medians[int(row.video_idx)] else "late")
    out = df.copy()
    out["chunk_half"] = flags
    return out


def _split_corr(df, split_col, split_values, score_col):
    parts = []
    for values in split_values:
        part = df[df[split_col].isin(values)]
        agg = part.groupby(["layer", "kv_head"])[score_col].mean().reset_index()
        parts.append(agg)
    merged = parts[0].merge(parts[1], on=["layer", "kv_head"], suffixes=("_a", "_b"))
    return merged, _corr(merged[f"{score_col}_a"], merged[f"{score_col}_b"])


def _aggregate_raw_to_kv(df, num_query_heads, num_kv_heads):
    group_size = int(num_query_heads) // int(num_kv_heads)
    if group_size <= 0 or int(num_query_heads) % int(num_kv_heads) != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )

    df = df.copy()
    df["kv_head"] = (df["head"].astype(int) // group_size).astype(int)
    df["query_heads_in_group"] = group_size

    preferred_keys = [
        "video_idx",
        "chunk_idx",
        "question_idx",
        "frame_start",
        "frame_end",
        "task",
        "layer",
        "kv_head",
    ]
    group_keys = [key for key in preferred_keys if key in df.columns]

    # Sum masses over query heads sharing one KV head. Since each KV group has
    # the same number of query heads, sum and mean give identical shares/ratios.
    agg = (
        df.groupby(group_keys, dropna=False)
        .agg(
            query_head_start=("head", "min"),
            query_head_end=("head", "max"),
            query_heads_observed=("head", "size"),
            pre_len=("pre_len", "first"),
            post_len=("post_len", "first"),
            cached_kv_len=("cached_kv_len", "first"),
            visual_start=("visual_start", "first"),
            prev_visual_tokens=("prev_visual_tokens", "first"),
            current_chunk_tokens=("current_chunk_tokens", "first"),
            local_prev_mass=("local_prev_mass", "sum"),
            local_current_mass=("local_current_mass", "sum"),
            global_prev_mass=("global_prev_mass", "sum"),
            global_current_mass=("global_current_mass", "sum"),
        )
        .reset_index()
    )
    agg["query_group_size"] = group_size

    local_total = agg["local_prev_mass"] + agg["local_current_mass"]
    global_total = agg["global_prev_mass"] + agg["global_current_mass"]
    agg["local_prev_share"] = agg["local_prev_mass"] / (local_total + EPS)
    agg["local_current_share"] = agg["local_current_mass"] / (local_total + EPS)
    agg["global_prev_share"] = agg["global_prev_mass"] / (global_total + EPS)
    agg["global_current_share"] = agg["global_current_mass"] / (global_total + EPS)
    agg["current_token_fraction"] = agg["current_chunk_tokens"] / (
        agg["prev_visual_tokens"] + agg["current_chunk_tokens"] + EPS
    )
    agg["local_current_share_minus_token_fraction"] = (
        agg["local_current_share"] - agg["current_token_fraction"]
    )
    agg["global_current_share_minus_token_fraction"] = (
        agg["global_current_share"] - agg["current_token_fraction"]
    )
    agg["local_current_to_prev_mass_ratio"] = agg["local_current_mass"] / (
        agg["local_prev_mass"] + EPS
    )
    agg["global_current_to_prev_mass_ratio"] = agg["global_current_mass"] / (
        agg["global_prev_mass"] + EPS
    )
    agg["local_current_to_prev_per_token_ratio"] = (
        agg["local_current_mass"] / (agg["current_chunk_tokens"] + EPS)
    ) / (agg["local_prev_mass"] / (agg["prev_visual_tokens"] + EPS) + EPS)
    agg["global_current_to_prev_per_token_ratio"] = (
        agg["global_current_mass"] / (agg["current_chunk_tokens"] + EPS)
    ) / (agg["global_prev_mass"] / (agg["prev_visual_tokens"] + EPS) + EPS)
    agg["global_minus_local_current_share"] = (
        agg["global_current_share"] - agg["local_current_share"]
    )
    agg["s_obs"] = 0.5 * (agg["local_current_share"] + agg["global_current_share"])
    agg["local_log_ratio"] = np.log(
        np.maximum(agg["local_current_to_prev_per_token_ratio"], EPS)
    )
    agg["global_log_ratio"] = np.log(
        np.maximum(agg["global_current_to_prev_per_token_ratio"], EPS)
    )
    agg["b_obs"] = 0.5 * (agg["local_log_ratio"] + agg["global_log_ratio"])
    agg["r_obs"] = np.exp(agg["b_obs"])
    return agg


def _aggregate_kv_profile(raw_kv_df):
    grouped = (
        raw_kv_df.groupby(["layer", "kv_head"])
        .agg(
            local_current_share=("local_current_share", "mean"),
            global_current_share=("global_current_share", "mean"),
            s_current_share=("s_obs", "mean"),
            b_log_per_token_ratio=("b_obs", "mean"),
            r_per_token_ratio=("r_obs", "mean"),
            local_per_token_ratio=("local_current_to_prev_per_token_ratio", "mean"),
            global_per_token_ratio=("global_current_to_prev_per_token_ratio", "mean"),
            current_token_fraction=("current_token_fraction", "mean"),
            num_observations=("s_obs", "size"),
            query_heads_observed=("query_heads_observed", "mean"),
        )
        .reset_index()
    )

    lo = grouped["b_log_per_token_ratio"].quantile(0.2)
    hi = grouped["b_log_per_token_ratio"].quantile(0.8)
    grouped["b_quantile_class"] = "mixed"
    grouped.loc[grouped["b_log_per_token_ratio"] <= lo, "b_quantile_class"] = (
        "memory_oriented_bottom20"
    )
    grouped.loc[grouped["b_log_per_token_ratio"] >= hi, "b_quantile_class"] = (
        "current_oriented_top20"
    )
    grouped["s_threshold_class"] = "mixed"
    grouped.loc[grouped["s_current_share"] < 0.25, "s_threshold_class"] = "memory_mass_lt25"
    grouped.loc[grouped["s_current_share"] > 0.75, "s_threshold_class"] = "current_mass_gt75"
    return grouped


def generate(raw_csv, out_dir, num_query_heads=28, num_kv_heads=4):
    _ensure_dir(out_dir)
    df = pd.read_csv(raw_csv)
    raw_kv = _aggregate_raw_to_kv(df, num_query_heads, num_kv_heads)
    kv_profile = _aggregate_kv_profile(raw_kv)

    raw_kv_csv = os.path.join(out_dir, "raw_prev_current_attention_kv.csv")
    kv_profile_csv = os.path.join(out_dir, "kv_head_profile_scores.csv")
    per_layer_csv = os.path.join(out_dir, "per_layer_kv_summary.csv")
    raw_kv.to_csv(raw_kv_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    kv_profile.to_csv(kv_profile_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    (
        kv_profile.groupby("layer")[
            [
                "local_current_share",
                "global_current_share",
                "s_current_share",
                "b_log_per_token_ratio",
                "r_per_token_ratio",
            ]
        ]
        .mean()
        .reset_index()
        .to_csv(per_layer_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    )

    s_arr = _matrix_from_kv(kv_profile, "s_current_share")
    b_arr = _matrix_from_kv(kv_profile, "b_log_per_token_ratio")
    r_arr = _matrix_from_kv(kv_profile, "r_per_token_ratio")

    outputs = {
        "raw_kv_csv": raw_kv_csv,
        "kv_profile_csv": kv_profile_csv,
        "per_layer_csv": per_layer_csv,
        "s_heatmap": os.path.join(out_dir, "kv_s_current_share_heatmap.png"),
        "b_heatmap": os.path.join(out_dir, "kv_b_log_per_token_ratio_heatmap.png"),
        "r_heatmap": os.path.join(out_dir, "kv_r_per_token_ratio_heatmap.png"),
        "local_global_scatter": os.path.join(out_dir, "kv_local_global_current_share_scatter.png"),
        "s_histogram": os.path.join(out_dir, "kv_s_current_share_histogram.png"),
        "b_histogram": os.path.join(out_dir, "kv_b_log_per_token_ratio_histogram.png"),
        "summary_json": os.path.join(out_dir, "kv_head_profile_summary.json"),
        "stability_json": os.path.join(out_dir, "kv_head_profile_stability.json"),
    }

    save_heatmap(
        s_arr,
        outputs["s_heatmap"],
        "KV s: current attention share",
        "sequential",
        0.0,
        1.0,
    )
    b_abs = float(np.nanquantile(np.abs(b_arr), 0.98))
    b_abs = max(b_abs, 0.1)
    save_heatmap(
        b_arr,
        outputs["b_heatmap"],
        "KV b: log current/previous per-token density",
        "diverging",
        -b_abs,
        b_abs,
    )
    r_clip = float(np.nanquantile(r_arr, 0.98))
    save_heatmap(
        r_arr,
        outputs["r_heatmap"],
        "KV r: current/previous per-token density",
        "sequential",
        0.0,
        r_clip,
    )
    save_scatter(
        kv_profile["local_current_share"].tolist(),
        kv_profile["global_current_share"].tolist(),
        outputs["local_global_scatter"],
        "Local vs global current-share by layer-KV-head",
        "local current_share",
        "global current_share",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    save_hist(
        kv_profile["s_current_share"].tolist(),
        outputs["s_histogram"],
        "Distribution of KV-head current-share scores",
        "s = mean(local, global) current_share",
        bins=24,
        xrange=(0.0, 1.0),
        vlines=[(0.25, "0.25", (190, 49, 49)), (0.75, "0.75", (190, 49, 49))],
    )
    save_hist(
        kv_profile["b_log_per_token_ratio"].tolist(),
        outputs["b_histogram"],
        "Distribution of KV-head log per-token density ratios",
        "b = log current/previous per-token ratio",
        bins=28,
        xrange=(
            float(np.nanquantile(kv_profile["b_log_per_token_ratio"], 0.01)),
            float(np.nanquantile(kv_profile["b_log_per_token_ratio"], 0.99)),
        ),
        vlines=[(0.0, "0", (190, 49, 49))],
    )

    stability = {}
    if "video_idx" in raw_kv.columns:
        videos = sorted(int(v) for v in raw_kv["video_idx"].unique())
        half = max(1, len(videos) // 2)
        if videos[half:]:
            for score_col in ("s_obs", "b_obs"):
                _, corr = _split_corr(raw_kv, "video_idx", [videos[:half], videos[half:]], score_col)
                stability[f"video_split_{score_col}_corr"] = corr
                stability[f"video_split_{score_col}_left_videos"] = videos[:half]
                stability[f"video_split_{score_col}_right_videos"] = videos[half:]

    chunk_df = _chunk_half_split(raw_kv)
    if chunk_df is not None:
        for score_col in ("s_obs", "b_obs"):
            _, corr = _split_corr(chunk_df, "chunk_half", [["early"], ["late"]], score_col)
            stability[f"chunk_split_{score_col}_corr"] = corr

    with open(outputs["stability_json"], "w") as f:
        json.dump(stability, f, indent=2)

    local = kv_profile["local_current_share"].to_numpy(dtype=np.float64)
    global_ = kv_profile["global_current_share"].to_numpy(dtype=np.float64)
    s = kv_profile["s_current_share"].to_numpy(dtype=np.float64)
    b = kv_profile["b_log_per_token_ratio"].to_numpy(dtype=np.float64)
    r = kv_profile["r_per_token_ratio"].to_numpy(dtype=np.float64)
    diff = global_ - local

    class_counts = {
        "s_lt_0p25": int((s < 0.25).sum()),
        "s_gt_0p75": int((s > 0.75).sum()),
        "global_lt_0p25": int((global_ < 0.25).sum()),
        "global_gt_0p75": int((global_ > 0.75).sum()),
        "b_bottom20_memory": int(
            (kv_profile["b_quantile_class"] == "memory_oriented_bottom20").sum()
        ),
        "b_top20_current": int(
            (kv_profile["b_quantile_class"] == "current_oriented_top20").sum()
        ),
        "b_mixed": int((kv_profile["b_quantile_class"] == "mixed").sum()),
        "r_gt_1": int((r > 1.0).sum()),
        "r_lt_1": int((r < 1.0).sum()),
        "r_lt_0p5": int((r < 0.5).sum()),
    }

    summary = {
        "source_raw_csv": raw_csv,
        "aggregation": "query_heads_grouped_by_shared_kv_head",
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(num_kv_heads),
        "group_size": int(num_query_heads) // int(num_kv_heads),
        "num_raw_query_rows": int(len(df)),
        "num_raw_kv_rows": int(len(raw_kv)),
        "num_layer_kv_heads": int(len(kv_profile)),
        "num_layers": int(kv_profile["layer"].max()) + 1,
        "kv_heads_per_layer": int(kv_profile["kv_head"].max()) + 1,
        "local_global_current_share_corr": _corr(local, global_),
        "mean_global_minus_local_current_share": float(np.nanmean(diff)),
        "mean_abs_global_minus_local_current_share": float(np.nanmean(np.abs(diff))),
        "median_abs_global_minus_local_current_share": float(np.nanmedian(np.abs(diff))),
        "s_current_share_mean": float(np.nanmean(s)),
        "s_current_share_std": float(np.nanstd(s, ddof=1)),
        "s_current_share_iqr": _quantile(s, 0.75) - _quantile(s, 0.25),
        "s_current_share_quantiles": {
            str(q): _quantile(s, q) for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        },
        "b_log_per_token_ratio_mean": float(np.nanmean(b)),
        "b_log_per_token_ratio_std": float(np.nanstd(b, ddof=1)),
        "b_log_per_token_ratio_quantiles": {
            str(q): _quantile(b, q) for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        },
        "r_per_token_ratio_median": float(np.nanmedian(r)),
        "r_per_token_ratio_quantiles": {
            str(q): _quantile(r, q) for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        },
        "mean_current_token_fraction": _finite_mean(raw_kv["current_token_fraction"].tolist()),
        "median_global_current_to_prev_per_token_ratio": _finite_median(
            raw_kv["global_current_to_prev_per_token_ratio"].tolist()
        ),
        "class_counts": class_counts,
        "stability": stability,
        "outputs": outputs,
    }
    with open(outputs["summary_json"], "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        default=(
            "results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full/"
            "raw_prev_current_attention.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full_kv",
    )
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    args = parser.parse_args()
    summary = generate(
        args.raw_csv,
        args.out_dir,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
    )
    print(
        json.dumps(
            {
                "summary_json": summary["outputs"]["summary_json"],
                "kv_profile_csv": summary["outputs"]["kv_profile_csv"],
                "raw_kv_csv": summary["outputs"]["raw_kv_csv"],
                "num_layer_kv_heads": summary["num_layer_kv_heads"],
                "s_current_share_mean": summary["s_current_share_mean"],
                "r_per_token_ratio_median": summary["r_per_token_ratio_median"],
                "class_counts": summary["class_counts"],
                "stability": summary["stability"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
