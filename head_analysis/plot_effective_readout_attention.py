"""Plot true-question readout attention for selected layer-KV-head groups."""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.obs_prev_current_chunk_attention_eager import build_observed_model
from head_analysis.profile_effective_memory_readout import (
    build_future_prompt,
    load_video_tensor,
    pool_query_attention,
    query_heads_for_kv,
    tokenize,
)
from video_qa.base import BaseVQA


EPS = 1e-12


def load_selected_kv_heads(profile_csv, metric, top_k, group="top"):
    import pandas as pd

    df = pd.read_csv(profile_csv)
    required = {"layer", "kv_head", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{profile_csv} missing columns: {sorted(missing)}")
    ascending = group == "bottom"
    part = df.sort_values(metric, ascending=ascending).head(int(top_k)).copy()
    return [
        {
            "layer": int(row.layer),
            "kv_head": int(row.kv_head),
            "score": float(getattr(row, metric)),
            "metric": metric,
            "group": group,
        }
        for row in part.itertuples(index=False)
    ]


def empty_stats(num_bins):
    return {
        "total_sum": np.zeros(int(num_bins), dtype=np.float64),
        "total_count": np.zeros(int(num_bins), dtype=np.float64),
        "boundary_values": [],
        "internal_mass": 0.0,
        "memory_start_boundary_mass": 0.0,
        "boundary_mass": 0.0,
        "current_start_boundary_mass": 0.0,
        "transition_boundary_mass": 0.0,
        "current_core_mass": 0.0,
        "current_mass": 0.0,
    }


def accumulate_whole_cache_bins(
    head_to_kv,
    visual_start,
    pre_len,
    post_len,
    stats,
    num_bins,
    boundary_window,
    current_boundary_window,
):
    if head_to_kv is None or head_to_kv.numel() == 0:
        return False
    kv_len = int(head_to_kv.shape[-1])
    start = min(int(visual_start), kv_len)
    pre = min(max(int(pre_len), start), kv_len)
    end = min(max(int(post_len), pre), kv_len)
    if pre <= start or end <= pre:
        return False

    memory_boundary = max(0, int(boundary_window))
    current_boundary = max(0, int(current_boundary_window))
    memory_core_start = min(pre, start + memory_boundary)
    memory_core_end = max(memory_core_start, pre - memory_boundary)
    current_boundary_end = min(end, pre + current_boundary)
    stats["internal_mass"] += float(head_to_kv[:, memory_core_start:memory_core_end].sum().item())
    memory_start_boundary_mass = float(head_to_kv[:, start:memory_core_start].sum().item())
    prev_boundary_mass = float(head_to_kv[:, memory_core_end:pre].sum().item())
    current_start_boundary_mass = float(head_to_kv[:, pre:current_boundary_end].sum().item())
    stats["memory_start_boundary_mass"] += memory_start_boundary_mass
    stats["boundary_mass"] += prev_boundary_mass
    stats["current_start_boundary_mass"] += current_start_boundary_mass
    stats["transition_boundary_mass"] += (
        memory_start_boundary_mass + prev_boundary_mass + current_start_boundary_mass
    )
    stats["current_core_mass"] += float(head_to_kv[:, current_boundary_end:end].sum().item())
    stats["current_mass"] += float(head_to_kv[:, pre:end].sum().item())

    total_len = end - start
    stats["boundary_values"].append((pre - start) / max(total_len, 1))
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
        stats["total_sum"][bin_idx] += float(values.sum().item())
        stats["total_count"][bin_idx] += num_heads
    return True


def plot_outputs(save_dir, selected, stats_by_name, num_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    x = np.linspace(0.5 / int(num_bins), 1.0 - 0.5 / int(num_bins), int(num_bins))
    rows = []
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    fig_log, ax_log = plt.subplots(figsize=(9.0, 4.6))
    cmap = plt.get_cmap("tab10")
    boundary_pool = []

    for idx, item in enumerate(selected):
        name = f"L{item['layer']}-KV{item['kv_head']}"
        stats = stats_by_name[name]
        y = stats["total_sum"] / np.maximum(stats["total_count"], 1.0)
        label = f"{name} ({item['metric']}={item['score']:.3g})"
        color = cmap(idx % 10)
        ax.plot(x, y, label=label, color=color, linewidth=2.0)
        ax_log.plot(x, np.maximum(np.nan_to_num(y, nan=0.0), EPS), label=label, color=color, linewidth=2.0)
        boundary_pool.extend(stats.get("boundary_values", []))
        for bin_idx, value in enumerate(y):
            rows.append([name, label, bin_idx, x[bin_idx], float(value)])

    mean_boundary = float(np.nanmean(boundary_pool)) if boundary_pool else None
    for axis in (ax, ax_log):
        if mean_boundary is not None:
            axis.axvline(mean_boundary, color="black", linestyle=":", linewidth=1.2)
        axis.set_xlabel("Normalized visual-token position over the whole cache")
        axis.set_ylabel("Mean true-question attention density per token")
        axis.legend(frameon=False, ncol=2, fontsize=8)

    ax.set_title("True-question readout attention by selected KV heads")
    fig.tight_layout()
    linear_path = os.path.join(save_dir, "true_question_readout_selected_kv_heads.png")
    fig.savefig(linear_path, dpi=220)
    plt.close(fig)

    ax_log.set_yscale("log")
    ax_log.set_title("True-question readout attention by selected KV heads (log scale)")
    ax_log.set_ylabel("Mean true-question attention density per token (log scale)")
    fig_log.tight_layout()
    log_path = os.path.join(save_dir, "true_question_readout_selected_kv_heads_log.png")
    fig_log.savefig(log_path, dpi=220)
    plt.close(fig_log)

    csv_path = os.path.join(save_dir, "true_question_readout_selected_kv_heads.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label", "bin", "x", "mean_density"])
        writer.writerows(rows)

    mass_rows = []
    for item in selected:
        name = f"L{item['layer']}-KV{item['kv_head']}"
        stats = stats_by_name[name]
        total = (
            stats["internal_mass"]
            + stats["memory_start_boundary_mass"]
            + stats["boundary_mass"]
            + stats["current_start_boundary_mass"]
            + stats["current_core_mass"]
            + EPS
        )
        memory = stats["internal_mass"] + stats["boundary_mass"]
        effective = stats["internal_mass"] + stats["current_core_mass"]
        mass_rows.append(
            {
                "name": name,
                "layer": item["layer"],
                "kv_head": item["kv_head"],
                "score": item["score"],
                "internal_share": stats["internal_mass"] / total,
                "memory_start_boundary_share": stats["memory_start_boundary_mass"] / total,
                "previous_boundary_share": stats["boundary_mass"] / total,
                "current_start_boundary_share": stats["current_start_boundary_mass"] / total,
                "transition_boundary_share": stats["transition_boundary_mass"] / total,
                "current_core_share": stats["current_core_mass"] / total,
                "current_share": stats["current_mass"] / total,
                "clean_internal_share": stats["internal_mass"] / (effective + EPS),
                "boundary_ratio_in_memory": stats["boundary_mass"] / (memory + EPS),
            }
        )
    mass_csv = os.path.join(save_dir, "selected_kv_region_mass.csv")
    with open(mass_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "layer",
                "kv_head",
                "score",
                "internal_share",
                "memory_start_boundary_share",
                "previous_boundary_share",
                "current_start_boundary_share",
                "transition_boundary_share",
                "current_core_share",
                "current_share",
                "clean_internal_share",
                "boundary_ratio_in_memory",
            ],
        )
        writer.writeheader()
        writer.writerows(mass_rows)

    return {
        "linear": linear_path,
        "log": log_path,
        "csv": csv_path,
        "mass_csv": mass_csv,
        "mean_previous_token_fraction": mean_boundary,
        "selected": selected,
    }


def run(args):
    model_name = "Qwen2.5-VL-7B-Instruct" if args.model == "qwen2.5_vl_7b" else args.model
    model_path = f"models/{model_name}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)

    with open(args.anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    selected = load_selected_kv_heads(
        args.profile_csv,
        args.metric,
        args.selected_topk,
        args.selected_group,
    )
    print("Selected KV heads:")
    for item in selected:
        print(f"  L{item['layer']}-KV{item['kv_head']} {item['metric']}={item['score']:.6g}")

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/plot_effective_readout_attention_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    stats_by_name = {f"L{x['layer']}-KV{x['kv_head']}": empty_stats(args.num_bins) for x in selected}
    obs_count = 0
    for video_idx, video_sample in enumerate(tqdm(anno, desc="Videos")):
        video_tensor = load_video_tensor(analyzer, video_sample)
        model.clear_cache()
        model.encode_init_prompt()
        current_frame_idx = 0
        chunk_idx = 0
        conversations = video_sample.get("conversations", [])
        if args.max_questions is not None:
            conversations = conversations[: args.max_questions]

        for _, sample in enumerate(conversations):
            end_frame_idx = math.ceil(sample.get("end_time", len(video_tensor)) * args.sample_fps)
            while current_frame_idx < end_frame_idx:
                if args.max_observations and obs_count >= args.max_observations:
                    summary = plot_outputs(args.save_dir, selected, stats_by_name, args.num_bins)
                    summary.update({"num_observations": obs_count})
                    with open(os.path.join(args.save_dir, "true_question_readout_summary.json"), "w") as f:
                        json.dump(summary, f, indent=2)
                    print(json.dumps(summary, indent=2))
                    return summary

                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                if next_encode_end <= current_frame_idx:
                    break

                pre_lens = model._get_cache_seq_len_per_layer()
                model.encode_video_chunk(video_tensor[current_frame_idx:next_encode_end])
                post_lens = model._get_cache_seq_len_per_layer()

                future_prompt = build_future_prompt(analyzer, model, sample)
                attn_future = model.compute_eager_attentions(tokenize(processor, future_prompt, model.device))
                visual_start = int(model.visual_start_idx)
                pos_cache = getattr(model, "_position_ids_cache", [])
                valid = False
                for item in selected:
                    layer_idx = int(item["layer"])
                    if layer_idx >= len(attn_future):
                        continue
                    future = pool_query_attention(
                        attn_future[layer_idx],
                        mode=args.query_pool,
                        last_n=args.last_n,
                    )
                    if future is None:
                        continue
                    q_heads = query_heads_for_kv(item["kv_head"], args.num_query_heads, args.num_kv_heads)
                    q_idx = torch.as_tensor(q_heads, device=future.device, dtype=torch.long)
                    group = future.index_select(0, q_idx)
                    if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                        cached_kv_len = int(pos_cache[layer_idx].shape[1])
                    else:
                        cached_kv_len = int(future.shape[-1])
                    group = group[:, :cached_kv_len]
                    name = f"L{item['layer']}-KV{item['kv_head']}"
                    valid = accumulate_whole_cache_bins(
                        group,
                        visual_start,
                        pre_lens[layer_idx],
                        post_lens[layer_idx],
                        stats_by_name[name],
                        args.num_bins,
                        args.boundary_window,
                        args.current_boundary_window,
                    ) or valid

                if model.compress_mode == "streamingvlm":
                    model._sliding_window_compress()
                else:
                    local_q, global_q = model.predict_next_question()
                    model.pseudo_forward(local_q, global_q)

                if valid:
                    obs_count += 1
                current_frame_idx = next_encode_end
                chunk_idx += 1
                torch.cuda.empty_cache()

    summary = plot_outputs(args.save_dir, selected, stats_by_name, args.num_bins)
    summary.update({"num_observations": obs_count})
    with open(os.path.join(args.save_dir, "true_question_readout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
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
    parser.add_argument("--num_videos", type=int, default=2)
    parser.add_argument("--max_questions", type=int, default=4)
    parser.add_argument("--max_observations", type=int, default=8)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--num_bins", type=int, default=200)
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--boundary_window", type=int, default=64)
    parser.add_argument("--current_boundary_window", type=int, default=None)
    parser.add_argument("--query_pool", choices=["mean", "last", "last_n"], default="mean")
    parser.add_argument("--last_n", type=int, default=4)
    parser.add_argument(
        "--profile_csv",
        default="results/observations/effective_memory_readout_n4_o80/effective_readout_scores.csv",
    )
    parser.add_argument("--metric", default="readout_shape_score")
    parser.add_argument("--selected_topk", type=int, default=4)
    parser.add_argument("--selected_group", choices=["top", "bottom"], default="top")
    parser.add_argument(
        "--save_dir",
        default="results/observations/effective_readout_attention_top4_n2_o8",
    )
    args = parser.parse_args()
    if args.current_boundary_window is None:
        args.current_boundary_window = args.boundary_window
    run(args)


if __name__ == "__main__":
    main()
