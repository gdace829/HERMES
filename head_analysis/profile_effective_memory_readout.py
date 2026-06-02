"""Profile true-query internal-memory readout heads.

This script is answer-facing rather than compression-selector-facing. It uses
the real StreamingBench question prompt to measure which layer-KV-head groups
read internal memory strongly, sharply, and away from the chunk boundary.

The resulting score is an offline head prior:

    readout_shape_score = internal_readout_share
                        * internal_topk_concentration
                        * transition_suppression

It does not use pseudo-query attention as the primary signal. Pseudo-query
compression is still executed after each observation to keep the streaming
cache evolution consistent with HERMES.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.obs_prev_current_chunk_attention_eager import build_observed_model
from video_qa.base import BaseVQA


EPS = 1e-12


def query_heads_for_kv(kv_head, num_query_heads=28, num_kv_heads=4):
    group_size = int(num_query_heads) // max(int(num_kv_heads), 1)
    start = int(kv_head) * group_size
    end = min(start + group_size, int(num_query_heads))
    return list(range(start, end))


def finite_mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def finite_median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else None


def load_video_tensor(analyzer, video_sample):
    video_path = video_sample["video_path"]
    if video_path.endswith(".npy"):
        video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
    elif os.path.isdir(video_path):
        video_fps = video_sample.get("fps", None)
        video = analyzer.load_video_frames(
            video_path,
            video_fps,
            clip=video_sample.get("clip", None),
        )
    else:
        video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
    return torch.from_numpy(video)


def build_future_prompt(analyzer, model, sample):
    question = sample.get("question", "")
    choices = sample.get("choices", None) or sample.get("candidates", None) or sample.get("options", None)
    if question and choices:
        return analyzer.format_mcqa_prompt(question, choices)["prompt"]
    if question:
        return model.get_prompt(question)
    return model.get_prompt("Answer the question based on the video.")


def tokenize(processor, text, device):
    input_ids = processor.tokenizer(text).input_ids
    return torch.as_tensor([input_ids], device=device, dtype=torch.int)


def pool_query_attention(attn, mode="mean", last_n=4):
    """Pool attention over selected query tokens, returning [num_query_heads, kv_len]."""
    if attn is None or attn.dim() < 4:
        return None
    values = attn[0].float()
    mode = str(mode or "mean").lower()
    if mode == "mean":
        pooled = values.mean(dim=1)
    elif mode == "last":
        pooled = values[:, -1, :]
    elif mode == "last_n":
        n = max(1, min(int(last_n), int(values.shape[1])))
        pooled = values[:, -n:, :].mean(dim=1)
    else:
        raise ValueError(f"Unsupported query_pool mode: {mode}")
    return torch.nan_to_num(
        pooled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def mean_query_attention(attn):
    """Backward-compatible mean pooling over all query tokens."""
    return pool_query_attention(attn, mode="mean", last_n=4)


def normalized_entropy(values):
    if values.numel() <= 1:
        return 0.0
    total = values.sum()
    if float(total.item()) <= EPS:
        return 0.0
    prob = values / (total + EPS)
    entropy = -(prob * torch.log(prob + EPS)).sum()
    return float((entropy / math.log(values.numel())).item())


def topk_concentration(values, top_k):
    if values.numel() <= 0:
        return 0.0
    total = float(values.sum().item())
    if total <= EPS:
        return 0.0
    k = min(int(top_k), int(values.numel()))
    return float(torch.topk(values, k=k, sorted=False).values.sum().item() / (total + EPS))


def topk_mean_attention(values, top_k):
    if values.numel() <= 0:
        return 0.0
    k = min(int(top_k), int(values.numel()))
    return float(torch.topk(values, k=k, sorted=False).values.mean().item())


def peak_to_median_stats(values):
    if values.numel() <= 0:
        return 0.0, 0.0, 0.0
    max_value = float(values.max().item())
    median_value = float(values.median().item())
    ratio = max_value / (median_value + EPS)
    return max_value, ratio, math.log1p(max(0.0, ratio))


def compute_readout_observation(
    future_attn,
    layer_idx,
    kv_head,
    q_heads,
    visual_start,
    pre_len,
    post_len,
    cached_kv_len,
    boundary_window,
    current_boundary_window,
    top_k,
):
    if future_attn is None:
        return None

    kv_len = min(int(cached_kv_len), int(future_attn.shape[-1]))
    mem_start = min(int(visual_start), kv_len)
    mem_end = min(max(int(pre_len), mem_start), kv_len)
    curr_start = mem_end
    curr_end = min(max(int(post_len), curr_start), kv_len)
    memory_boundary = max(0, int(boundary_window))
    current_boundary = max(0, int(current_boundary_window))
    mem_core_start = min(mem_end, mem_start + memory_boundary)
    mem_core_end = max(mem_core_start, mem_end - memory_boundary)
    mem_start_bnd_start = mem_start
    mem_start_bnd_end = mem_core_start
    prev_bnd_start = mem_core_end
    prev_bnd_end = mem_end
    curr_bnd_start = curr_start
    curr_bnd_end = min(curr_end, curr_start + current_boundary)
    curr_core_start = curr_bnd_end
    curr_core_end = curr_end

    num_internal = mem_core_end - mem_core_start
    num_memory_start_boundary = mem_start_bnd_end - mem_start_bnd_start
    num_boundary = prev_bnd_end - prev_bnd_start
    num_current_start_boundary = curr_bnd_end - curr_bnd_start
    num_current_core = curr_core_end - curr_core_start
    num_current = curr_end - curr_start
    if num_internal <= 0 or not q_heads:
        return None

    q_idx = torch.as_tensor(q_heads, device=future_attn.device, dtype=torch.long)
    group_attn = future_attn.index_select(0, q_idx).mean(dim=0)[:kv_len]
    group_attn = torch.clamp(group_attn, min=0.0)

    internal = group_attn[mem_core_start:mem_core_end]
    memory_start_boundary = group_attn[mem_start_bnd_start:mem_start_bnd_end]
    boundary = group_attn[prev_bnd_start:prev_bnd_end]
    current_start_boundary = group_attn[curr_bnd_start:curr_bnd_end]
    current_core = group_attn[curr_core_start:curr_core_end]
    current = group_attn[curr_start:curr_end]
    memory = group_attn[mem_start:mem_end]
    visual = group_attn[mem_start:curr_end]

    internal_mass = float(internal.sum().item())
    memory_start_boundary_mass = (
        float(memory_start_boundary.sum().item()) if memory_start_boundary.numel() else 0.0
    )
    boundary_mass = float(boundary.sum().item()) if boundary.numel() else 0.0
    current_start_boundary_mass = (
        float(current_start_boundary.sum().item()) if current_start_boundary.numel() else 0.0
    )
    current_core_mass = float(current_core.sum().item()) if current_core.numel() else 0.0
    current_mass = float(current.sum().item()) if current.numel() else 0.0
    memory_mass = float(memory.sum().item())
    visual_mass = float(visual.sum().item())
    transition_boundary_mass = memory_start_boundary_mass + boundary_mass + current_start_boundary_mass

    internal_readout_share = internal_mass / (visual_mass + EPS)
    memory_readout_share = memory_mass / (visual_mass + EPS)
    current_readout_share = current_mass / (visual_mass + EPS)
    current_core_readout_share = current_core_mass / (visual_mass + EPS)
    clean_internal_readout_share = internal_mass / (internal_mass + current_core_mass + EPS)
    previous_boundary_ratio = boundary_mass / (memory_mass + EPS)
    current_start_boundary_ratio = current_start_boundary_mass / (current_mass + EPS)
    transition_boundary_ratio = transition_boundary_mass / (visual_mass + EPS)
    boundary_ratio = transition_boundary_ratio
    boundary_suppression = max(0.0, 1.0 - transition_boundary_ratio)

    concentration = topk_concentration(internal, top_k)
    top1_concentration = topk_concentration(internal, 1)
    top4_concentration = topk_concentration(internal, 4)
    top8_concentration = topk_concentration(internal, 8)
    top100_mean_attention = topk_mean_attention(internal, 100)
    entropy = normalized_entropy(internal)
    peakiness = max(0.0, 1.0 - entropy)
    internal_peak, peak_to_median, log_peak_to_median = peak_to_median_stats(internal)

    readout_shape_score = internal_readout_share * concentration * boundary_suppression
    peak_readout_shape_score = readout_shape_score * peakiness
    spiky_readout_score = (
        internal_readout_share
        * top4_concentration
        * peakiness
        * log_peak_to_median
        * boundary_suppression
    )

    return {
        "layer": int(layer_idx),
        "kv_head": int(kv_head),
        "visual_start": int(visual_start),
        "pre_len": int(pre_len),
        "post_len": int(post_len),
        "cached_kv_len": int(cached_kv_len),
        "memory_core_start": int(mem_core_start),
        "memory_core_end": int(mem_core_end),
        "num_internal_tokens": int(num_internal),
        "num_memory_start_boundary_tokens": int(num_memory_start_boundary),
        "num_boundary_tokens": int(num_boundary),
        "num_current_start_boundary_tokens": int(num_current_start_boundary),
        "num_current_core_tokens": int(num_current_core),
        "num_current_tokens": int(num_current),
        "top_k_eff": int(min(int(top_k), int(internal.numel()))),
        "internal_mass": internal_mass,
        "memory_start_boundary_mass": memory_start_boundary_mass,
        "boundary_mass": boundary_mass,
        "current_start_boundary_mass": current_start_boundary_mass,
        "current_core_mass": current_core_mass,
        "transition_boundary_mass": transition_boundary_mass,
        "current_mass": current_mass,
        "memory_mass": memory_mass,
        "visual_mass": visual_mass,
        "internal_readout_share": float(internal_readout_share),
        "memory_readout_share": float(memory_readout_share),
        "current_readout_share": float(current_readout_share),
        "current_core_readout_share": float(current_core_readout_share),
        "clean_internal_readout_share": float(clean_internal_readout_share),
        "previous_boundary_ratio": float(previous_boundary_ratio),
        "current_start_boundary_ratio": float(current_start_boundary_ratio),
        "transition_boundary_ratio": float(transition_boundary_ratio),
        "boundary_ratio": float(boundary_ratio),
        "boundary_suppression": float(boundary_suppression),
        "internal_topk_concentration": float(concentration),
        "internal_top1_concentration": float(top1_concentration),
        "internal_top4_concentration": float(top4_concentration),
        "internal_top8_concentration": float(top8_concentration),
        "internal_top100_mean_attention": float(top100_mean_attention),
        "internal_peakiness": float(peakiness),
        "internal_peak": float(internal_peak),
        "internal_peak_to_median": float(peak_to_median),
        "internal_log_peak_to_median": float(log_peak_to_median),
        "readout_shape_score": float(readout_shape_score),
        "peak_readout_shape_score": float(peak_readout_shape_score),
        "spiky_readout_score": float(spiky_readout_score),
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def quantile_classes(score_rows, metric, quantile):
    valid = [
        row
        for row in score_rows
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]
    valid = sorted(valid, key=lambda row: float(row[metric]))
    total = len(valid)
    if total == 0:
        return [], [], []
    k = max(1, int(math.ceil(total * float(quantile))))
    k = min(k, total // 2)
    weak = valid[:k]
    effective = valid[-k:]
    selected = {(row["layer"], row["kv_head"]) for row in weak + effective}
    mixed = [row for row in valid if (row["layer"], row["kv_head"]) not in selected]
    return effective, weak, mixed


def aggregate_rows(rows, args):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["kv_head"]))].append(row)

    score_rows = []
    for (layer, kv_head), part in sorted(grouped.items()):
        score_rows.append(
            {
                "layer": layer,
                "kv_head": kv_head,
                "readout_shape_score": finite_mean([r["readout_shape_score"] for r in part]),
                "readout_shape_score_median": finite_median([r["readout_shape_score"] for r in part]),
                "peak_readout_shape_score": finite_mean([r["peak_readout_shape_score"] for r in part]),
                "spiky_readout_score": finite_mean([r["spiky_readout_score"] for r in part]),
                "internal_readout_share": finite_mean([r["internal_readout_share"] for r in part]),
                "memory_readout_share": finite_mean([r["memory_readout_share"] for r in part]),
                "current_readout_share": finite_mean([r["current_readout_share"] for r in part]),
                "current_core_readout_share": finite_mean([r["current_core_readout_share"] for r in part]),
                "clean_internal_readout_share": finite_mean([r["clean_internal_readout_share"] for r in part]),
                "internal_topk_concentration": finite_mean([r["internal_topk_concentration"] for r in part]),
                "internal_top1_concentration": finite_mean([r["internal_top1_concentration"] for r in part]),
                "internal_top4_concentration": finite_mean([r["internal_top4_concentration"] for r in part]),
                "internal_top8_concentration": finite_mean([r["internal_top8_concentration"] for r in part]),
                "internal_top100_mean_attention": finite_mean([r["internal_top100_mean_attention"] for r in part]),
                "internal_peakiness": finite_mean([r["internal_peakiness"] for r in part]),
                "internal_peak_to_median": finite_mean([r["internal_peak_to_median"] for r in part]),
                "internal_log_peak_to_median": finite_mean([r["internal_log_peak_to_median"] for r in part]),
                "previous_boundary_ratio": finite_mean([r["previous_boundary_ratio"] for r in part]),
                "current_start_boundary_ratio": finite_mean([r["current_start_boundary_ratio"] for r in part]),
                "transition_boundary_ratio": finite_mean([r["transition_boundary_ratio"] for r in part]),
                "boundary_ratio": finite_mean([r["boundary_ratio"] for r in part]),
                "boundary_suppression": finite_mean([r["boundary_suppression"] for r in part]),
                "num_observations": len(part),
            }
        )

    effective, weak, mixed = quantile_classes(score_rows, args.class_metric, args.quantile)
    classes = {
        "granularity": "kv",
        "score": "true_query_internal_memory_readout_shape",
        "score_column": args.class_metric,
        "quantile": float(args.quantile),
        "boundary_window": int(args.boundary_window),
        "current_boundary_window": int(args.current_boundary_window),
        "top_k": int(args.top_k),
        "num_query_heads": int(args.num_query_heads),
        "num_kv_heads": int(args.num_kv_heads),
        "effective_readout_heads": [[r["layer"], r["kv_head"]] for r in effective],
        "weak_readout_heads": [[r["layer"], r["kv_head"]] for r in weak],
        "mixed_heads": [[r["layer"], r["kv_head"]] for r in mixed],
        "counts": {
            "effective_readout_heads": len(effective),
            "weak_readout_heads": len(weak),
            "mixed_heads": len(mixed),
            "total": len(score_rows),
        },
        "scores": [
            {
                "layer": r["layer"],
                "kv_head": r["kv_head"],
                "readout_shape_score": r["readout_shape_score"],
                "peak_readout_shape_score": r["peak_readout_shape_score"],
                "spiky_readout_score": r["spiky_readout_score"],
                "internal_readout_share": r["internal_readout_share"],
                "clean_internal_readout_share": r["clean_internal_readout_share"],
                "internal_topk_concentration": r["internal_topk_concentration"],
                "internal_top4_concentration": r["internal_top4_concentration"],
                "internal_top100_mean_attention": r["internal_top100_mean_attention"],
                "internal_log_peak_to_median": r["internal_log_peak_to_median"],
                "transition_boundary_ratio": r["transition_boundary_ratio"],
                "boundary_ratio": r["boundary_ratio"],
                "num_observations": r["num_observations"],
            }
            for r in score_rows
        ],
    }
    return score_rows, classes


def matrix_from_scores(score_rows, value_col, num_layers, num_kv_heads):
    mat = [[float("nan") for _ in range(int(num_kv_heads))] for _ in range(int(num_layers))]
    for row in score_rows:
        layer = int(row["layer"])
        kv_head = int(row["kv_head"])
        if 0 <= layer < num_layers and 0 <= kv_head < num_kv_heads:
            value = row.get(value_col)
            mat[layer][kv_head] = float(value) if value is not None else float("nan")
    return mat


def plot_outputs(score_rows, save_dir, num_layers, num_kv_heads):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping plots because matplotlib/numpy is unavailable: {exc}")
        return {}

    os.makedirs(save_dir, exist_ok=True)
    outputs = {}
    specs = [
        ("readout_shape_score", "effective_readout_score_heatmap.png", "True-query internal-memory readout score", "viridis"),
        ("spiky_readout_score", "spiky_readout_score_heatmap.png", "Spiky true-query internal-memory readout score", "viridis"),
        ("internal_readout_share", "internal_readout_share_heatmap.png", "Internal-memory readout share", "Blues"),
        ("clean_internal_readout_share", "clean_internal_readout_share_heatmap.png", "Internal share excluding transition boundaries", "Blues"),
        ("internal_topk_concentration", "internal_topk_concentration_heatmap.png", "Internal top-K concentration", "Purples"),
        ("internal_top100_mean_attention", "internal_top100_mean_attention_heatmap.png", "Memory-core top-100 mean attention", "YlOrRd"),
        ("internal_log_peak_to_median", "internal_log_peak_to_median_heatmap.png", "Internal log peak-to-median", "Oranges"),
        ("transition_boundary_ratio", "transition_boundary_ratio_heatmap.png", "Transition-boundary mass ratio", "magma"),
    ]
    for value_col, filename, title, cmap in specs:
        arr = np.asarray(matrix_from_scores(score_rows, value_col, num_layers, num_kv_heads), dtype=np.float64)
        fig, ax = plt.subplots(figsize=(5.2, 9.0))
        im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_xlabel("KV head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        ax.set_xticks(np.arange(num_kv_heads))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs[value_col] = path

    valid = [
        r
        for r in score_rows
        if r["internal_readout_share"] is not None
        and r["internal_topk_concentration"] is not None
        and r["readout_shape_score"] is not None
    ]
    if valid:
        x = np.asarray([r["internal_readout_share"] for r in valid], dtype=np.float64)
        y = np.asarray([r["internal_topk_concentration"] for r in valid], dtype=np.float64)
        c = np.asarray([r["readout_shape_score"] for r in valid], dtype=np.float64)
        labels = [(r["layer"], r["kv_head"]) for r in valid]
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=28, alpha=0.82)
        ax.set_xlabel("Internal-memory readout share")
        ax.set_ylabel("Internal top-K concentration")
        ax.set_title("Readout mass vs concentration")
        for idx in np.argsort(c)[-4:]:
            ax.text(x[idx], y[idx], f"L{labels[idx][0]}-K{labels[idx][1]}", fontsize=8)
        fig.colorbar(sc, ax=ax, label="readout score")
        fig.tight_layout()
        path = os.path.join(save_dir, "readout_mass_vs_concentration_scatter.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs["readout_mass_vs_concentration_scatter"] = path

    return outputs


def save_results(rows, args, obs_count):
    os.makedirs(args.save_dir, exist_ok=True)
    raw_fields = [
        "video_idx",
        "question_idx",
        "chunk_idx",
        "task",
        "frame_start",
        "frame_end",
        "layer",
        "kv_head",
        "visual_start",
        "pre_len",
        "post_len",
        "cached_kv_len",
        "memory_core_start",
        "memory_core_end",
        "num_internal_tokens",
        "num_memory_start_boundary_tokens",
        "num_boundary_tokens",
        "num_current_start_boundary_tokens",
        "num_current_core_tokens",
        "num_current_tokens",
        "top_k_eff",
        "internal_mass",
        "memory_start_boundary_mass",
        "boundary_mass",
        "current_start_boundary_mass",
        "current_core_mass",
        "transition_boundary_mass",
        "current_mass",
        "memory_mass",
        "visual_mass",
        "internal_readout_share",
        "memory_readout_share",
        "current_readout_share",
        "current_core_readout_share",
        "clean_internal_readout_share",
        "previous_boundary_ratio",
        "current_start_boundary_ratio",
        "transition_boundary_ratio",
        "boundary_ratio",
        "boundary_suppression",
        "internal_topk_concentration",
        "internal_top1_concentration",
        "internal_top4_concentration",
        "internal_top8_concentration",
        "internal_top100_mean_attention",
        "internal_peakiness",
        "internal_peak",
        "internal_peak_to_median",
        "internal_log_peak_to_median",
        "readout_shape_score",
        "peak_readout_shape_score",
        "spiky_readout_score",
    ]
    raw_csv = os.path.join(args.save_dir, "raw_effective_readout.csv")
    write_csv(raw_csv, rows, raw_fields)

    score_rows, classes = aggregate_rows(rows, args)
    score_fields = [
        "layer",
        "kv_head",
        "readout_shape_score",
        "readout_shape_score_median",
        "peak_readout_shape_score",
        "spiky_readout_score",
        "internal_readout_share",
        "memory_readout_share",
        "current_readout_share",
        "current_core_readout_share",
        "clean_internal_readout_share",
        "internal_topk_concentration",
        "internal_top1_concentration",
        "internal_top4_concentration",
        "internal_top8_concentration",
        "internal_top100_mean_attention",
        "internal_peakiness",
        "internal_peak_to_median",
        "internal_log_peak_to_median",
        "previous_boundary_ratio",
        "current_start_boundary_ratio",
        "transition_boundary_ratio",
        "boundary_ratio",
        "boundary_suppression",
        "num_observations",
    ]
    score_csv = os.path.join(args.save_dir, "effective_readout_scores.csv")
    write_csv(score_csv, score_rows, score_fields)

    classes_path = os.path.join(args.save_dir, "head_classes_effective_readout.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)

    plots = plot_outputs(score_rows, args.save_dir, args.num_layers, args.num_kv_heads)
    summary = {
        "attention_source": "eager_output_attentions_for_true_question_prompt",
        "usage": "offline_answer_facing_readout_profile_not_online_eviction_signal",
        "num_chunk_observations": int(obs_count),
        "num_layer_kv_observations": int(len(rows)),
        "num_layers": int(args.num_layers),
        "num_kv_heads": int(args.num_kv_heads),
        "num_query_heads": int(args.num_query_heads),
        "top_k": int(args.top_k),
        "boundary_window": int(args.boundary_window),
        "current_boundary_window": int(args.current_boundary_window),
        "query_pool": args.query_pool,
        "last_n": int(args.last_n),
        "class_metric": args.class_metric,
        "raw_csv": raw_csv,
        "score_csv": score_csv,
        "head_classes_json": classes_path,
        "figures": plots,
        "mean_readout_shape_score": finite_mean([r["readout_shape_score"] for r in rows]),
        "mean_spiky_readout_score": finite_mean([r["spiky_readout_score"] for r in rows]),
        "median_readout_shape_score": finite_median([r["readout_shape_score"] for r in rows]),
        "mean_internal_readout_share": finite_mean([r["internal_readout_share"] for r in rows]),
        "mean_clean_internal_readout_share": finite_mean([r["clean_internal_readout_share"] for r in rows]),
        "mean_internal_topk_concentration": finite_mean([r["internal_topk_concentration"] for r in rows]),
        "mean_internal_top4_concentration": finite_mean([r["internal_top4_concentration"] for r in rows]),
        "mean_internal_top100_mean_attention": finite_mean([r["internal_top100_mean_attention"] for r in rows]),
        "mean_internal_log_peak_to_median": finite_mean([r["internal_log_peak_to_median"] for r in rows]),
        "mean_previous_boundary_ratio": finite_mean([r["previous_boundary_ratio"] for r in rows]),
        "mean_current_start_boundary_ratio": finite_mean([r["current_start_boundary_ratio"] for r in rows]),
        "mean_transition_boundary_ratio": finite_mean([r["transition_boundary_ratio"] for r in rows]),
        "mean_boundary_ratio": finite_mean([r["boundary_ratio"] for r in rows]),
        "class_counts": classes["counts"],
    }
    summary_path = os.path.join(args.save_dir, "effective_readout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(args):
    model_name = "Qwen2.5-VL-7B-Instruct" if args.model == "qwen2.5_vl_7b" else args.model
    model_path = f"models/{model_name}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)
    args.num_layers = int(getattr(model, "num_layers", args.num_layers))

    with open(args.anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/effective_memory_readout_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    rows = []
    obs_count = 0
    random.seed(args.seed)

    for video_idx, video_sample in enumerate(tqdm(anno, desc="Videos")):
        video_tensor = load_video_tensor(analyzer, video_sample)
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
                    summary = save_results(rows, args, obs_count)
                    print(json.dumps(summary, indent=2))
                    return summary

                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                if next_encode_end <= current_frame_idx:
                    break

                video_chunk = video_tensor[current_frame_idx:next_encode_end]
                pre_lens = model._get_cache_seq_len_per_layer()
                model.encode_video_chunk(video_chunk)
                post_lens = model._get_cache_seq_len_per_layer()

                future_prompt = build_future_prompt(analyzer, model, sample)
                attn_future = model.compute_eager_attentions(tokenize(processor, future_prompt, model.device))

                visual_start = int(model.visual_start_idx)
                pos_cache = getattr(model, "_position_ids_cache", [])
                valid_rows_this_obs = 0
                for layer_idx, af in enumerate(attn_future):
                    future = pool_query_attention(af, mode=args.query_pool, last_n=args.last_n)
                    if future is None:
                        continue
                    if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                        cached_kv_len = int(pos_cache[layer_idx].shape[1])
                    else:
                        cached_kv_len = int(future.shape[-1])

                    for kv_head in range(int(args.num_kv_heads)):
                        q_heads = query_heads_for_kv(kv_head, args.num_query_heads, args.num_kv_heads)
                        row = compute_readout_observation(
                            future_attn=future,
                            layer_idx=layer_idx,
                            kv_head=kv_head,
                            q_heads=q_heads,
                            visual_start=visual_start,
                            pre_len=pre_lens[layer_idx],
                            post_len=post_lens[layer_idx],
                            cached_kv_len=cached_kv_len,
                            boundary_window=args.boundary_window,
                            current_boundary_window=args.current_boundary_window,
                            top_k=args.top_k,
                        )
                        if row is None:
                            continue
                        row.update(
                            {
                                "video_idx": video_idx,
                                "question_idx": question_idx,
                                "chunk_idx": chunk_idx,
                                "task": sample.get("task", "Unknown"),
                                "frame_start": current_frame_idx,
                                "frame_end": next_encode_end,
                            }
                        )
                        rows.append(row)
                        valid_rows_this_obs += 1

                if model.compress_mode == "streamingvlm":
                    model._sliding_window_compress()
                else:
                    local_q, global_q = model.predict_next_question()
                    model.pseudo_forward(local_q, global_q)

                if valid_rows_this_obs > 0:
                    obs_count += 1
                current_frame_idx = next_encode_end
                chunk_idx += 1
                torch.cuda.empty_cache()

    summary = save_results(rows, args, obs_count)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=4)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--max_observations", type=int, default=80)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--boundary_window", type=int, default=64)
    parser.add_argument("--current_boundary_window", type=int, default=None)
    parser.add_argument("--query_pool", choices=["mean", "last", "last_n"], default="mean")
    parser.add_argument("--last_n", type=int, default=4)
    parser.add_argument("--class_metric", default="readout_shape_score")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        default="results/observations/effective_memory_readout_n4_o80",
    )
    args = parser.parse_args()
    if args.current_boundary_window is None:
        args.current_boundary_window = args.boundary_window
    run(args)


if __name__ == "__main__":
    main()
