"""Plot token-position readout attention from pseudo-query tokens.

This is a lightweight diagnostic for checking whether the previous/current
readout bias is only a coarse region effect. It does not save full attention
matrices. Instead, it bins cached visual tokens by their normalized position
inside previous memory and latest chunk:

    previous memory: [-1, 0)
    latest chunk:    [0, 1]

For each pseudo-query forward, attention is first averaged over query tokens,
then binned per layer/head, and finally averaged across heads, layers, and
observations.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.obs_prev_current_chunk_attention_eager import build_observed_model
from video_qa.base import BaseVQA


EPS = 1e-12


def _region_bin_edges(start, end, num_bins):
    length = int(end) - int(start)
    if length <= 0:
        return []
    edges = np.linspace(0, length, int(num_bins) + 1)
    spans = []
    for i in range(int(num_bins)):
        lo = int(round(edges[i]))
        hi = int(round(edges[i + 1]))
        if hi <= lo:
            continue
        spans.append((int(start) + lo, int(start) + hi, i))
    return spans


def _accumulate_attention_bins(
    attn,
    visual_start,
    pre_len,
    post_len,
    prev_sum,
    prev_count,
    curr_sum,
    curr_count,
    region_mass,
    num_bins,
    head_indices=None,
):
    """Accumulate per-token attention density into normalized region bins."""
    if attn.dim() < 4:
        return False

    # [num_heads, kv_len], averaged over pseudo-query tokens.
    head_to_kv = torch.nan_to_num(attn[0].mean(dim=1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    if head_indices is not None:
        head_indices = [int(h) for h in head_indices if 0 <= int(h) < int(head_to_kv.shape[0])]
        if not head_indices:
            return False
        idx = torch.as_tensor(head_indices, device=head_to_kv.device, dtype=torch.long)
        head_to_kv = head_to_kv.index_select(0, idx)
    kv_len = int(head_to_kv.shape[-1])

    prev_start = min(int(visual_start), kv_len)
    prev_end = min(max(int(pre_len), int(visual_start)), kv_len)
    curr_start = min(max(int(pre_len), int(visual_start)), kv_len)
    curr_end = min(max(int(post_len), curr_start), kv_len)
    if prev_end <= prev_start or curr_end <= curr_start:
        return False

    prev = head_to_kv[:, prev_start:prev_end]
    curr = head_to_kv[:, curr_start:curr_end]
    num_heads = int(head_to_kv.shape[0])

    region_mass["prev"] += float(prev.sum().item())
    region_mass["curr"] += float(curr.sum().item())

    for lo, hi, bin_idx in _region_bin_edges(prev_start, prev_end, num_bins):
        values = torch.nan_to_num(head_to_kv[:, lo:hi].mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        prev_sum[bin_idx] += float(values.sum().item())
        prev_count[bin_idx] += num_heads

    for lo, hi, bin_idx in _region_bin_edges(curr_start, curr_end, num_bins):
        values = torch.nan_to_num(head_to_kv[:, lo:hi].mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        curr_sum[bin_idx] += float(values.sum().item())
        curr_count[bin_idx] += num_heads
    return True


def _accumulate_whole_cache_bins(
    attn,
    visual_start,
    pre_len,
    post_len,
    total_sum,
    total_count,
    boundary_values,
    num_bins,
    head_indices=None,
):
    """Accumulate per-token density over the whole visual cache.

    Unlike ``_accumulate_attention_bins``, this does not normalize previous and
    current regions to equal width. The x-axis is the whole visual cache, so the
    latest chunk occupies more bins when it contains more tokens.
    """
    if attn.dim() < 4:
        return False

    head_to_kv = torch.nan_to_num(attn[0].mean(dim=1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    if head_indices is not None:
        head_indices = [int(h) for h in head_indices if 0 <= int(h) < int(head_to_kv.shape[0])]
        if not head_indices:
            return False
        idx = torch.as_tensor(head_indices, device=head_to_kv.device, dtype=torch.long)
        head_to_kv = head_to_kv.index_select(0, idx)
    kv_len = int(head_to_kv.shape[-1])
    start = min(int(visual_start), kv_len)
    pre = min(max(int(pre_len), start), kv_len)
    end = min(max(int(post_len), pre), kv_len)
    if pre <= start or end <= pre:
        return False

    total_len = end - start
    boundary_values.append((pre - start) / max(total_len, 1))
    num_heads = int(head_to_kv.shape[0])
    edges = np.linspace(0, total_len, int(num_bins) + 1)
    for bin_idx in range(int(num_bins)):
        lo = int(round(edges[bin_idx]))
        hi = int(round(edges[bin_idx + 1]))
        if hi <= lo:
            continue
        values = torch.nan_to_num(
            head_to_kv[:, start + lo:start + hi].mean(dim=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        total_sum[bin_idx] += float(values.sum().item())
        total_count[bin_idx] += num_heads
    return True


def _load_head_groups(path, num_query_heads=28, num_kv_heads=4):
    """Return class -> layer -> query-head ids.

    Supports both query-head class JSON and KV-head class JSON. KV heads are
    expanded to the query heads that share each KV head.
    """
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)

    num_query_heads = int(data.get("num_query_heads", data.get("num_heads", num_query_heads)))
    num_kv_heads = int(data.get("num_kv_heads", num_kv_heads))
    group_size = int(data.get("group_size", num_query_heads // max(num_kv_heads, 1)))

    if data.get("granularity") == "kv" or "memory_kv_heads" in data or "memory_oriented_kv" in data:
        raw_groups = {
            "memory": data.get("memory_kv_heads", data.get("memory_oriented_kv", [])),
            "current": data.get("current_kv_heads", data.get("current_sensitive_kv", [])),
            "mixed": data.get("mixed_kv_heads", data.get("mixed_kv", [])),
        }
        groups = {}
        for name, heads in raw_groups.items():
            by_layer = defaultdict(list)
            for layer, kv_head in heads:
                start = int(kv_head) * group_size
                end = min(start + group_size, num_query_heads)
                by_layer[int(layer)].extend(range(start, end))
            groups[name] = {layer: sorted(set(values)) for layer, values in by_layer.items()}
        return groups

    raw_groups = {
        "memory": data.get("memory_oriented", data.get("memory_bottom", [])),
        "current": data.get("current_sensitive", data.get("current_top", [])),
        "mixed": data.get("mixed", []),
    }
    groups = {}
    for name, heads in raw_groups.items():
        by_layer = defaultdict(list)
        for layer, head in heads:
            by_layer[int(layer)].append(int(head))
        groups[name] = {layer: sorted(set(values)) for layer, values in by_layer.items()}
    return groups


def _query_heads_for_kv(kv_head, num_query_heads=28, num_kv_heads=4):
    group_size = int(num_query_heads) // max(int(num_kv_heads), 1)
    start = int(kv_head) * group_size
    end = min(start + group_size, int(num_query_heads))
    return list(range(start, end))


def _load_selected_kv_heads(profile_csv, top_k=4, metric="b_log_per_token_ratio", selected_group="both"):
    """Return selected layer-KV heads from the static profile CSV."""
    if not profile_csv or not os.path.exists(profile_csv):
        return []
    try:
        import pandas as pd
    except ImportError:
        return []
    df = pd.read_csv(profile_csv)
    if "layer" not in df or "kv_head" not in df or metric not in df:
        return []
    top = df.sort_values(metric, ascending=False).head(int(top_k)).copy()
    bottom = df.sort_values(metric, ascending=True).head(int(top_k)).copy()
    selected = []
    parts = []
    if selected_group in ("both", "top"):
        parts.append(("top", top))
    if selected_group in ("both", "bottom"):
        parts.append(("bottom", bottom))
    for label, part in parts:
        for _, row in part.iterrows():
            selected.append(
                {
                    "group": label,
                    "layer": int(row["layer"]),
                    "kv_head": int(row["kv_head"]),
                    "score": float(row[metric]),
                    "metric": metric,
                    "s": float(row.get("s_current_share", float("nan"))),
                }
            )
    return selected


def _plot_kv_head_comparison(out_dir, num_bins, local_stats, global_stats, prefix, title, labels=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(0.5 / num_bins, 1.0 - 0.5 / num_bins, num_bins)
    rows = []
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    fig_log, ax_log = plt.subplots(figsize=(9.0, 4.5))
    cmap = plt.get_cmap("tab10")

    boundary_pool = []
    for idx, name in enumerate(sorted(local_stats.keys())):
        if name not in global_stats:
            continue
        local = local_stats[name]["total_sum"] / np.maximum(local_stats[name]["total_count"], 1.0)
        global_ = global_stats[name]["total_sum"] / np.maximum(global_stats[name]["total_count"], 1.0)
        y = 0.5 * (local + global_)
        label = labels.get(name, name) if labels else name
        for bin_idx, value in enumerate(y):
            rows.append([name, label, bin_idx, x[bin_idx], value, local[bin_idx], global_[bin_idx]])
        boundary_pool.extend(local_stats[name].get("boundary_values", []))
        color = cmap(idx % 10)
        ax.plot(x, y, label=label, color=color, linewidth=2.0)
        ax_log.plot(x, np.maximum(np.nan_to_num(y, nan=0.0), EPS), label=label, color=color, linewidth=2.0)

    mean_boundary = float(np.nanmean(boundary_pool)) if boundary_pool else None
    for axis in (ax, ax_log):
        if mean_boundary is not None:
            axis.axvline(mean_boundary, color="black", linestyle=":", linewidth=1.2)
        axis.set_xlabel("Normalized visual-token position over the whole cache")
        axis.set_ylabel("Mean attention density per token")
        axis.legend(frameon=False, ncol=2)

    ax.set_title(title)
    fig.tight_layout()
    linear_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(linear_path, dpi=220)
    plt.close(fig)

    ax_log.set_yscale("log")
    ax_log.set_title(f"{title} (log scale)")
    ax_log.set_ylabel("Mean attention density per token (log scale)")
    fig_log.tight_layout()
    log_path = os.path.join(out_dir, f"{prefix}_log.png")
    fig_log.savefig(log_path, dpi=220)
    plt.close(fig_log)

    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label", "bin", "x", "mean_density", "local_density", "global_density"])
        writer.writerows(rows)

    return {
        f"{prefix}_csv": csv_path,
        f"{prefix}_linear": linear_path,
        f"{prefix}_log": log_path,
        f"{prefix}_mean_previous_token_fraction": mean_boundary,
    }


def _plot_head_group_comparison(out_dir, num_bins, local_group_stats, global_group_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "memory": "#4C78A8",
        "current": "#F58518",
        "mixed": "#54A24B",
    }
    labels = {
        "memory": "Memory-preferential",
        "current": "Current-sensitive",
        "mixed": "Mixed",
    }
    x = np.linspace(0.5 / num_bins, 1.0 - 0.5 / num_bins, num_bins)
    rows = []
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    fig_log, ax_log = plt.subplots(figsize=(9.0, 4.5))

    boundary_pool = []
    for name in ("memory", "current", "mixed"):
        if name not in local_group_stats or name not in global_group_stats:
            continue
        local = local_group_stats[name]["total_sum"] / np.maximum(local_group_stats[name]["total_count"], 1.0)
        global_ = global_group_stats[name]["total_sum"] / np.maximum(global_group_stats[name]["total_count"], 1.0)
        y = 0.5 * (local + global_)
        for bin_idx, value in enumerate(y):
            rows.append([name, bin_idx, x[bin_idx], value, local[bin_idx], global_[bin_idx]])
        boundary_pool.extend(local_group_stats[name].get("boundary_values", []))
        ax.plot(x, y, label=labels.get(name, name), color=colors.get(name), linewidth=2.0)
        ax_log.plot(x, np.maximum(np.nan_to_num(y, nan=0.0), EPS), label=labels.get(name, name), color=colors.get(name), linewidth=2.0)

    mean_boundary = float(np.nanmean(boundary_pool)) if boundary_pool else None
    for axis in (ax, ax_log):
        if mean_boundary is not None:
            axis.axvline(mean_boundary, color="black", linestyle=":", linewidth=1.2)
        axis.set_xlabel("Normalized visual-token position over the whole cache")
        axis.set_ylabel("Mean attention density per token")
        axis.legend(frameon=False)

    ax.set_title("Readout attention by head class")
    fig.tight_layout()
    linear_path = os.path.join(out_dir, "whole_cache_attention_by_head_class.png")
    fig.savefig(linear_path, dpi=220)
    plt.close(fig)

    ax_log.set_yscale("log")
    ax_log.set_title("Readout attention by head class (log scale)")
    ax_log.set_ylabel("Mean attention density per token (log scale)")
    fig_log.tight_layout()
    log_path = os.path.join(out_dir, "whole_cache_attention_by_head_class_log.png")
    fig_log.savefig(log_path, dpi=220)
    plt.close(fig_log)

    csv_path = os.path.join(out_dir, "whole_cache_attention_by_head_class.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "bin", "x", "mean_density", "local_density", "global_density"])
        writer.writerows(rows)

    return {
        "class_csv": csv_path,
        "class_linear": linear_path,
        "class_log": log_path,
        "class_mean_previous_token_fraction": mean_boundary,
    }


def _plot_outputs(
    out_dir,
    num_bins,
    local_stats,
    global_stats,
    meta,
    local_group_stats=None,
    global_group_stats=None,
    local_kv_id_stats=None,
    global_kv_id_stats=None,
    local_selected_kv_stats=None,
    global_selected_kv_stats=None,
    kv_id_labels=None,
    selected_kv_labels=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    x_prev = np.linspace(-1.0 + 0.5 / num_bins, -0.5 / num_bins, num_bins)
    x_curr = np.linspace(0.0 + 0.5 / num_bins, 1.0 - 0.5 / num_bins, num_bins)
    x = np.concatenate([x_prev, x_curr])

    def density(stats, region):
        sums = stats[f"{region}_sum"]
        counts = np.maximum(stats[f"{region}_count"], 1.0)
        return sums / counts

    local_y = np.concatenate([density(local_stats, "prev"), density(local_stats, "curr")])
    global_y = np.concatenate([density(global_stats, "prev"), density(global_stats, "curr")])
    local_total_y = local_stats["total_sum"] / np.maximum(local_stats["total_count"], 1.0)
    global_total_y = global_stats["total_sum"] / np.maximum(global_stats["total_count"], 1.0)
    boundary_values = np.asarray(local_stats.get("boundary_values", []), dtype=np.float64)
    mean_boundary = float(np.nanmean(boundary_values)) if boundary_values.size else None

    csv_path = os.path.join(out_dir, "region_aligned_token_attention_bins.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "bin", "x", "local_density", "global_density"])
        for i in range(num_bins):
            writer.writerow(["previous", i, x_prev[i], local_y[i], global_y[i]])
        for i in range(num_bins):
            writer.writerow(["current", i, x_curr[i], local_y[num_bins + i], global_y[num_bins + i]])

    total_csv_path = os.path.join(out_dir, "whole_cache_token_attention_bins.csv")
    total_x = np.linspace(0.5 / num_bins, 1.0 - 0.5 / num_bins, num_bins)
    with open(total_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "x", "local_density", "global_density"])
        for i in range(num_bins):
            writer.writerow([i, total_x[i], local_total_y[i], global_total_y[i]])

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(x, local_y, label="Local pseudo-query", linewidth=2.0, color="#4C78A8")
    ax.plot(x, global_y, label="Global pseudo-query", linewidth=2.0, color="#F58518")
    ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Normalized visual-token position (previous memory -> latest chunk)")
    ax.set_ylabel("Mean attention density per token")
    ax.set_title("Query-averaged token-level readout attention")
    ax.text(-0.5, ax.get_ylim()[1] * 0.96, "previous memory", ha="center", va="top", fontsize=10)
    ax.text(0.5, ax.get_ylim()[1] * 0.96, "latest chunk", ha="center", va="top", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_aligned_token_attention_density.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(total_x, local_total_y, label="Local pseudo-query", linewidth=2.0, color="#4C78A8")
    ax.plot(total_x, global_total_y, label="Global pseudo-query", linewidth=2.0, color="#F58518")
    if mean_boundary is not None:
        ax.axvline(mean_boundary, color="black", linestyle=":", linewidth=1.2)
        ymax = ax.get_ylim()[1]
        ax.text(mean_boundary / 2, ymax * 0.96, "previous memory", ha="center", va="top", fontsize=10)
        ax.text((1 + mean_boundary) / 2, ymax * 0.96, "latest chunk", ha="center", va="top", fontsize=10)
    ax.set_xlabel("Normalized visual-token position over the whole cache")
    ax.set_ylabel("Mean attention density per token")
    ax.set_title("Query-averaged readout attention (token-length scaled)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "whole_cache_token_attention_density.png"), dpi=220)
    plt.close(fig)

    if np.isfinite(local_total_y).any() or np.isfinite(global_total_y).any():
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        ax.plot(total_x, np.maximum(np.nan_to_num(local_total_y, nan=0.0), EPS), label="Local pseudo-query", linewidth=2.0, color="#4C78A8")
        ax.plot(total_x, np.maximum(np.nan_to_num(global_total_y, nan=0.0), EPS), label="Global pseudo-query", linewidth=2.0, color="#F58518")
        if mean_boundary is not None:
            ax.axvline(mean_boundary, color="black", linestyle=":", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Normalized visual-token position over the whole cache")
        ax.set_ylabel("Mean attention density per token (log scale)")
        ax.set_title("Query-averaged readout attention (token-length scaled)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "whole_cache_token_attention_density_log.png"), dpi=220)
        plt.close(fig)

    if np.isfinite(local_y).any() or np.isfinite(global_y).any():
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        ax.plot(x, np.maximum(np.nan_to_num(local_y, nan=0.0), EPS), label="Local pseudo-query", linewidth=2.0, color="#4C78A8")
        ax.plot(x, np.maximum(np.nan_to_num(global_y, nan=0.0), EPS), label="Global pseudo-query", linewidth=2.0, color="#F58518")
        ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Normalized visual-token position (previous memory -> latest chunk)")
        ax.set_ylabel("Mean attention density per token (log scale)")
        ax.set_title("Query-averaged token-level readout attention")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "region_aligned_token_attention_density_log.png"), dpi=220)
        plt.close(fig)

    def shares(stats):
        prev = stats["mass"]["prev"]
        curr = stats["mass"]["curr"]
        total = prev + curr + EPS
        return prev / total, curr / total

    local_prev_share, local_curr_share = shares(local_stats)
    global_prev_share, global_curr_share = shares(global_stats)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ["Previous memory", "Latest chunk"]
    values = [[local_prev_share, local_curr_share], [global_prev_share, global_curr_share]]
    xpos = np.arange(2)
    width = 0.35
    ax.bar(xpos - width / 2, values[0], width=width, label="Local", color="#4C78A8")
    ax.bar(xpos + width / 2, values[1], width=width, label="Global", color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Attention mass share")
    ax.set_title("Region mass after query-token averaging")
    ax.legend(frameon=False)
    for j, vals in enumerate(values):
        for i, value in enumerate(vals):
            ax.text(i + (-width / 2 if j == 0 else width / 2), value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_attention_mass_share.png"), dpi=220)
    plt.close(fig)

    class_outputs = {}
    if local_group_stats and global_group_stats:
        class_outputs = _plot_head_group_comparison(
            out_dir,
            num_bins,
            local_group_stats,
            global_group_stats,
        )

    kv_outputs = {}
    if local_kv_id_stats and global_kv_id_stats:
        kv_outputs.update(
            _plot_kv_head_comparison(
                out_dir,
                num_bins,
                local_kv_id_stats,
                global_kv_id_stats,
                "whole_cache_attention_by_kv_id",
                "Readout attention by KV-head id",
                labels=kv_id_labels or {},
            )
        )
    if local_selected_kv_stats and global_selected_kv_stats:
        kv_outputs.update(
            _plot_kv_head_comparison(
                out_dir,
                num_bins,
                local_selected_kv_stats,
                global_selected_kv_stats,
                "whole_cache_attention_by_selected_kv_heads",
                "Readout attention by selected layer-KV heads",
                labels=selected_kv_labels or {},
            )
        )

    summary = {
        **meta,
        "csv": csv_path,
        "whole_cache_csv": total_csv_path,
        "mean_previous_token_fraction": mean_boundary,
        "figures": {
            "density": os.path.join(out_dir, "region_aligned_token_attention_density.png"),
            "density_log": os.path.join(out_dir, "region_aligned_token_attention_density_log.png"),
            "whole_cache_density": os.path.join(out_dir, "whole_cache_token_attention_density.png"),
            "whole_cache_density_log": os.path.join(out_dir, "whole_cache_token_attention_density_log.png"),
            "mass_share": os.path.join(out_dir, "region_attention_mass_share.png"),
            **class_outputs,
            **kv_outputs,
        },
        "local_region_share": {
            "previous": local_prev_share,
            "current": local_curr_share,
        },
        "global_region_share": {
            "previous": global_prev_share,
            "current": global_curr_share,
        },
    }
    with open(os.path.join(out_dir, "token_attention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)

    with open(args.anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/token_readout_attention_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    def empty_stats():
        return {
            "prev_sum": np.zeros(args.num_bins, dtype=np.float64),
            "prev_count": np.zeros(args.num_bins, dtype=np.float64),
            "curr_sum": np.zeros(args.num_bins, dtype=np.float64),
            "curr_count": np.zeros(args.num_bins, dtype=np.float64),
            "total_sum": np.zeros(args.num_bins, dtype=np.float64),
            "total_count": np.zeros(args.num_bins, dtype=np.float64),
            "boundary_values": [],
            "mass": {"prev": 0.0, "curr": 0.0},
        }

    local_stats = empty_stats()
    global_stats = empty_stats()
    head_groups = _load_head_groups(args.head_classes, args.num_query_heads, args.num_kv_heads)
    local_group_stats = {name: empty_stats() for name in head_groups}
    global_group_stats = {name: empty_stats() for name in head_groups}
    local_kv_id_stats = {}
    global_kv_id_stats = {}
    kv_id_labels = {}
    local_selected_kv_stats = {}
    global_selected_kv_stats = {}
    selected_kv_labels = {}
    selected_kv_heads = []
    if args.plot_kv_heads:
        for kv_head in range(int(args.num_kv_heads)):
            name = f"kv{kv_head}"
            local_kv_id_stats[name] = empty_stats()
            global_kv_id_stats[name] = empty_stats()
            q_heads = _query_heads_for_kv(kv_head, args.num_query_heads, args.num_kv_heads)
            kv_id_labels[name] = f"KV{kv_head} (Q{q_heads[0]}-Q{q_heads[-1]})"

        selected_kv_heads = _load_selected_kv_heads(
            args.kv_profile_csv,
            args.selected_kv_topk,
            args.kv_profile_metric,
            args.selected_kv_group,
        )
        for item in selected_kv_heads:
            name = f"{item['group']}_L{item['layer']}_KV{item['kv_head']}"
            local_selected_kv_stats[name] = empty_stats()
            global_selected_kv_stats[name] = empty_stats()
            label_metric = str(item["metric"]).replace("_", " ")
            selected_kv_labels[name] = f"{item['group']} L{item['layer']}-KV{item['kv_head']} ({label_metric}={item['score']:.3g})"
    obs_count = 0

    for video_idx, video_sample in enumerate(tqdm(anno, desc="Videos")):
        video_path = video_sample["video_path"]
        if video_path.endswith(".npy"):
            video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            video_fps = video_sample.get("fps", None)
            video = analyzer.load_video_frames(video_path, video_fps, clip=video_sample.get("clip", None))
            video_tensor = torch.from_numpy(video)
        else:
            video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
            video_tensor = torch.from_numpy(video)

        model.clear_cache()
        model.encode_init_prompt()
        current_frame_idx = 0
        chunk_idx = 0
        conversations = video_sample.get("conversations", [])
        if args.max_questions is not None:
            conversations = conversations[: args.max_questions]

        for question_idx, sample in enumerate(conversations):
            end_frame_idx = math.ceil(sample.get("end_time", len(video_tensor)) * args.sample_fps)
            while current_frame_idx < end_frame_idx:
                if args.max_observations and obs_count >= args.max_observations:
                    return _plot_outputs(
                        args.save_dir,
                        args.num_bins,
                        local_stats,
                        global_stats,
                        {
                            "num_observations": obs_count,
                            "num_bins_per_region": args.num_bins,
                            "num_videos_requested": args.num_videos,
                            "max_questions": args.max_questions,
                            "head_classes": args.head_classes,
                            "plot_kv_heads": args.plot_kv_heads,
                            "kv_profile_csv": args.kv_profile_csv,
                        },
                        local_group_stats,
                        global_group_stats,
                        local_kv_id_stats,
                        global_kv_id_stats,
                        local_selected_kv_stats,
                        global_selected_kv_stats,
                        kv_id_labels,
                        selected_kv_labels,
                    )

                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                if next_encode_end <= current_frame_idx:
                    break

                video_chunk = video_tensor[current_frame_idx:next_encode_end]
                pre_lens = model._get_cache_seq_len_per_layer()
                model.encode_video_chunk(video_chunk)
                post_lens = model._get_cache_seq_len_per_layer()

                local_q, global_q = model.predict_next_question()
                local_ids = torch.as_tensor(
                    [processor.tokenizer(local_q).input_ids],
                    device=model.device,
                    dtype=torch.int,
                )
                global_ids = torch.as_tensor(
                    [processor.tokenizer(global_q).input_ids],
                    device=model.device,
                    dtype=torch.int,
                )
                attn_local = model.compute_eager_attentions(local_ids)
                attn_global = model.compute_eager_attentions(global_ids)

                visual_start = int(model.visual_start_idx)
                valid_layers = 0
                obs_boundaries = []
                for layer_idx, (al, ag) in enumerate(zip(attn_local, attn_global)):
                    local_valid = _accumulate_attention_bins(
                        al,
                        visual_start,
                        pre_lens[layer_idx],
                        post_lens[layer_idx],
                        local_stats["prev_sum"],
                        local_stats["prev_count"],
                        local_stats["curr_sum"],
                        local_stats["curr_count"],
                        local_stats["mass"],
                        args.num_bins,
                    )
                    global_valid = _accumulate_attention_bins(
                        ag,
                        visual_start,
                        pre_lens[layer_idx],
                        post_lens[layer_idx],
                        global_stats["prev_sum"],
                        global_stats["prev_count"],
                        global_stats["curr_sum"],
                        global_stats["curr_count"],
                        global_stats["mass"],
                        args.num_bins,
                    )
                    if local_valid and global_valid:
                        valid_layers += 1
                        _accumulate_whole_cache_bins(
                            al,
                            visual_start,
                            pre_lens[layer_idx],
                            post_lens[layer_idx],
                            local_stats["total_sum"],
                            local_stats["total_count"],
                            obs_boundaries,
                            args.num_bins,
                        )
                        _accumulate_whole_cache_bins(
                            ag,
                            visual_start,
                            pre_lens[layer_idx],
                            post_lens[layer_idx],
                            global_stats["total_sum"],
                            global_stats["total_count"],
                            [],
                            args.num_bins,
                        )

                        for group_name, by_layer in head_groups.items():
                            heads = by_layer.get(layer_idx, [])
                            if not heads:
                                continue
                            _accumulate_attention_bins(
                                al,
                                visual_start,
                                pre_lens[layer_idx],
                                post_lens[layer_idx],
                                local_group_stats[group_name]["prev_sum"],
                                local_group_stats[group_name]["prev_count"],
                                local_group_stats[group_name]["curr_sum"],
                                local_group_stats[group_name]["curr_count"],
                                local_group_stats[group_name]["mass"],
                                args.num_bins,
                                head_indices=heads,
                            )
                            _accumulate_attention_bins(
                                ag,
                                visual_start,
                                pre_lens[layer_idx],
                                post_lens[layer_idx],
                                global_group_stats[group_name]["prev_sum"],
                                global_group_stats[group_name]["prev_count"],
                                global_group_stats[group_name]["curr_sum"],
                                global_group_stats[group_name]["curr_count"],
                                global_group_stats[group_name]["mass"],
                                args.num_bins,
                                head_indices=heads,
                            )
                            _accumulate_whole_cache_bins(
                                al,
                                visual_start,
                                pre_lens[layer_idx],
                                post_lens[layer_idx],
                                local_group_stats[group_name]["total_sum"],
                                local_group_stats[group_name]["total_count"],
                                local_group_stats[group_name]["boundary_values"],
                                args.num_bins,
                                head_indices=heads,
                            )
                            _accumulate_whole_cache_bins(
                                ag,
                                visual_start,
                                pre_lens[layer_idx],
                                post_lens[layer_idx],
                                global_group_stats[group_name]["total_sum"],
                                global_group_stats[group_name]["total_count"],
                                [],
                                args.num_bins,
                                head_indices=heads,
                            )

                        if args.plot_kv_heads:
                            for kv_head in range(int(args.num_kv_heads)):
                                name = f"kv{kv_head}"
                                heads = _query_heads_for_kv(kv_head, args.num_query_heads, args.num_kv_heads)
                                _accumulate_whole_cache_bins(
                                    al,
                                    visual_start,
                                    pre_lens[layer_idx],
                                    post_lens[layer_idx],
                                    local_kv_id_stats[name]["total_sum"],
                                    local_kv_id_stats[name]["total_count"],
                                    local_kv_id_stats[name]["boundary_values"],
                                    args.num_bins,
                                    head_indices=heads,
                                )
                                _accumulate_whole_cache_bins(
                                    ag,
                                    visual_start,
                                    pre_lens[layer_idx],
                                    post_lens[layer_idx],
                                    global_kv_id_stats[name]["total_sum"],
                                    global_kv_id_stats[name]["total_count"],
                                    [],
                                    args.num_bins,
                                    head_indices=heads,
                                )

                            for item in selected_kv_heads:
                                if int(item["layer"]) != int(layer_idx):
                                    continue
                                name = f"{item['group']}_L{item['layer']}_KV{item['kv_head']}"
                                heads = _query_heads_for_kv(item["kv_head"], args.num_query_heads, args.num_kv_heads)
                                _accumulate_whole_cache_bins(
                                    al,
                                    visual_start,
                                    pre_lens[layer_idx],
                                    post_lens[layer_idx],
                                    local_selected_kv_stats[name]["total_sum"],
                                    local_selected_kv_stats[name]["total_count"],
                                    local_selected_kv_stats[name]["boundary_values"],
                                    args.num_bins,
                                    head_indices=heads,
                                )
                                _accumulate_whole_cache_bins(
                                    ag,
                                    visual_start,
                                    pre_lens[layer_idx],
                                    post_lens[layer_idx],
                                    global_selected_kv_stats[name]["total_sum"],
                                    global_selected_kv_stats[name]["total_count"],
                                    [],
                                    args.num_bins,
                                    head_indices=heads,
                                )

                if model.compress_mode == "streamingvlm":
                    model._sliding_window_compress()
                else:
                    model.pseudo_forward(local_q, global_q)

                if valid_layers > 0:
                    obs_count += 1
                    local_stats["boundary_values"].extend(obs_boundaries)
                current_frame_idx = next_encode_end
                chunk_idx += 1
                torch.cuda.empty_cache()

    return _plot_outputs(
        args.save_dir,
        args.num_bins,
        local_stats,
        global_stats,
        {
            "num_observations": obs_count,
            "num_bins_per_region": args.num_bins,
            "num_videos_requested": args.num_videos,
            "max_questions": args.max_questions,
            "head_classes": args.head_classes,
            "plot_kv_heads": args.plot_kv_heads,
            "kv_profile_csv": args.kv_profile_csv,
        },
        local_group_stats,
        global_group_stats,
        local_kv_id_stats,
        global_kv_id_stats,
        local_selected_kv_stats,
        global_selected_kv_stats,
        kv_id_labels,
        selected_kv_labels,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--num_videos", type=int, default=1)
    parser.add_argument("--max_questions", type=int, default=2)
    parser.add_argument("--max_observations", type=int, default=8)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--head_classes", default=None)
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--plot_kv_heads", action="store_true")
    parser.add_argument(
        "--kv_profile_csv",
        default="results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_head_profile_scores.csv",
    )
    parser.add_argument("--kv_profile_metric", default="b_log_per_token_ratio")
    parser.add_argument("--selected_kv_topk", type=int, default=4)
    parser.add_argument("--selected_kv_group", choices=["both", "top", "bottom"], default="both")
    parser.add_argument(
        "--save_dir",
        default="results/observations/token_readout_attention_eager_n1_o8",
    )
    summary = run(parser.parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
