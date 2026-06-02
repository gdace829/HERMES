"""Plot memory-core top-k raw attention for selected layer-KV-head groups."""

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


def load_selected(profile_csv, metric, top_k):
    import pandas as pd

    df = pd.read_csv(profile_csv)
    required = {"layer", "kv_head", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{profile_csv} missing columns: {sorted(missing)}")
    top = df.sort_values(metric, ascending=False).head(int(top_k)).copy()
    bottom = df.sort_values(metric, ascending=True).head(int(top_k)).copy()

    def rows(part, group):
        return [
            {
                "layer": int(row.layer),
                "kv_head": int(row.kv_head),
                "score": float(getattr(row, metric)),
                "metric": metric,
                "group": group,
                "name": f"L{int(row.layer)}-KV{int(row.kv_head)}",
            }
            for row in part.itertuples(index=False)
        ]

    return rows(top, "top"), rows(bottom, "bottom")


def topk_means(values, ks):
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.clamp(values, min=0.0)
    if values.numel() <= 0:
        return {int(k): float("nan") for k in ks}
    sorted_values = torch.sort(values, descending=True).values
    out = {}
    for k in ks:
        kk = min(int(k), int(sorted_values.numel()))
        out[int(k)] = float(sorted_values[:kk].mean().item()) if kk > 0 else float("nan")
    return out


def empty_stats(ks):
    return {int(k): [] for k in ks}


def summarize(values):
    vals = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=0))


def plot(save_dir, selected, stats_by_name, ks, obs_count):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    x = np.asarray([int(k) for k in ks], dtype=np.float64)
    rows = []

    group_values = {"top": {int(k): [] for k in ks}, "bottom": {int(k): [] for k in ks}}
    per_head = []
    for item in selected:
        name = item["name"]
        group = item["group"]
        y = []
        for k in ks:
            mean, std = summarize(stats_by_name[name][int(k)])
            y.append(mean)
            rows.append(
                {
                    "group": group,
                    "name": name,
                    "layer": item["layer"],
                    "kv_head": item["kv_head"],
                    "selection_score": item["score"],
                    "top_k": int(k),
                    "mean_attention": mean,
                    "std_attention": std,
                    "num_values": len(stats_by_name[name][int(k)]),
                }
            )
            group_values[group][int(k)].extend(stats_by_name[name][int(k)])
        per_head.append((item, np.asarray(y, dtype=np.float64)))

    csv_path = os.path.join(save_dir, "memory_core_topk_attention.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "group",
            "name",
            "layer",
            "kv_head",
            "selection_score",
            "top_k",
            "mean_attention",
            "std_attention",
            "num_values",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    group_rows = []
    for group in ("top", "bottom"):
        for k in ks:
            mean, std = summarize(group_values[group][int(k)])
            group_rows.append(
                {
                    "group": group,
                    "top_k": int(k),
                    "mean_attention": mean,
                    "std_attention": std,
                    "num_values": len(group_values[group][int(k)]),
                }
            )
    group_csv = os.path.join(save_dir, "memory_core_topk_attention_group_mean.csv")
    with open(group_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "top_k", "mean_attention", "std_attention", "num_values"],
        )
        writer.writeheader()
        writer.writerows(group_rows)

    group_df = {g: [] for g in ("top", "bottom")}
    group_std = {g: [] for g in ("top", "bottom")}
    for group in ("top", "bottom"):
        for k in ks:
            mean, std = summarize(group_values[group][int(k)])
            group_df[group].append(mean)
            group_std[group].append(std)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 260,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = {"top": "#1f77b4", "bottom": "#d62728"}
    labels = {"top": "Top memory-active KV groups", "bottom": "Bottom KV groups"}
    for group in ("top", "bottom"):
        y = np.asarray(group_df[group], dtype=np.float64)
        std = np.asarray(group_std[group], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=2.2, color=colors[group], label=labels[group])
        ax.fill_between(x, np.maximum(y - std, EPS), y + std, color=colors[group], alpha=0.14, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Top-k tokens within memory core")
    ax.set_ylabel("Mean raw attention of top-k tokens")
    ax.set_title("Memory-core top-k attention")
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = os.path.join(save_dir, "memory_core_topk_attention_top_vs_bottom.png")
    pdf_path = os.path.join(save_dir, "memory_core_topk_attention_top_vs_bottom.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for item, y in per_head:
        color = colors[item["group"]]
        alpha = 0.9 if item["group"] == "top" else 0.55
        linestyle = "-" if item["group"] == "top" else "--"
        ax.plot(x, y, marker="o", linewidth=1.6, color=color, alpha=alpha, linestyle=linestyle, label=item["name"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Top-k tokens within memory core")
    ax.set_ylabel("Mean raw attention of top-k tokens")
    ax.set_title("Selected KV groups")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    per_head_png = os.path.join(save_dir, "memory_core_topk_attention_selected_heads.png")
    per_head_pdf = os.path.join(save_dir, "memory_core_topk_attention_selected_heads.pdf")
    fig.savefig(per_head_png)
    fig.savefig(per_head_pdf)
    plt.close(fig)

    summary = {
        "num_observations": int(obs_count),
        "ks": [int(k) for k in ks],
        "selected": selected,
        "csv": csv_path,
        "group_csv": group_csv,
        "top_vs_bottom_png": png_path,
        "top_vs_bottom_pdf": pdf_path,
        "selected_heads_png": per_head_png,
        "selected_heads_pdf": per_head_pdf,
    }
    summary_path = os.path.join(save_dir, "memory_core_topk_attention_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def run(args):
    model_name = "Qwen2.5-VL-7B-Instruct" if args.model == "qwen2.5_vl_7b" else args.model
    model_path = f"models/{model_name}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)

    with open(args.anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    ks = [int(x) for x in args.topk_values.split(",") if str(x).strip()]
    top, bottom = load_selected(args.profile_csv, args.metric, args.selected_topk)
    selected = top + bottom
    print("Selected KV heads:")
    for item in selected:
        print(f"  {item['group']:>6} {item['name']} {item['metric']}={item['score']:.6g}")

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/plot_memory_core_topk_attention_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    stats_by_name = {item["name"]: empty_stats(ks) for item in selected}
    obs_count = 0
    for video_sample in tqdm(anno, desc="Videos"):
        video_tensor = load_video_tensor(analyzer, video_sample)
        model.clear_cache()
        model.encode_init_prompt()
        current_frame_idx = 0
        conversations = video_sample.get("conversations", [])
        if args.max_questions is not None:
            conversations = conversations[: args.max_questions]

        for sample in conversations:
            end_frame_idx = math.ceil(sample.get("end_time", len(video_tensor)) * args.sample_fps)
            while current_frame_idx < end_frame_idx:
                if args.max_observations and obs_count >= args.max_observations:
                    return plot(args.save_dir, selected, stats_by_name, ks, obs_count)

                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                if next_encode_end <= current_frame_idx:
                    break

                pre_lens = model._get_cache_seq_len_per_layer()
                model.encode_video_chunk(video_tensor[current_frame_idx:next_encode_end])

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
                    if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                        cached_kv_len = int(pos_cache[layer_idx].shape[1])
                    else:
                        cached_kv_len = int(future.shape[-1])
                    kv_len = min(cached_kv_len, int(future.shape[-1]))
                    mem_start = min(visual_start, kv_len)
                    mem_end = min(max(int(pre_lens[layer_idx]), mem_start), kv_len)
                    core_start = min(mem_end, mem_start + int(args.boundary_window))
                    core_end = max(core_start, mem_end - int(args.boundary_window))
                    if core_end <= core_start:
                        continue
                    q_heads = query_heads_for_kv(item["kv_head"], args.num_query_heads, args.num_kv_heads)
                    q_idx = torch.as_tensor(q_heads, device=future.device, dtype=torch.long)
                    group = future.index_select(0, q_idx).mean(dim=0)[:kv_len]
                    values = group[core_start:core_end]
                    means = topk_means(values, ks)
                    for k, v in means.items():
                        stats_by_name[item["name"]][int(k)].append(v)
                    valid = True

                if model.compress_mode == "streamingvlm":
                    model._sliding_window_compress()
                else:
                    local_q, global_q = model.predict_next_question()
                    model.pseudo_forward(local_q, global_q)

                if valid:
                    obs_count += 1
                current_frame_idx = next_encode_end
                torch.cuda.empty_cache()

    return plot(args.save_dir, selected, stats_by_name, ks, obs_count)


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
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--boundary_window", type=int, default=64)
    parser.add_argument("--query_pool", choices=["mean", "last", "last_n"], default="last_n")
    parser.add_argument("--last_n", type=int, default=4)
    parser.add_argument("--topk_values", default="1,4,8,16,32,64,100")
    parser.add_argument(
        "--profile_csv",
        default="results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv",
    )
    parser.add_argument("--metric", default="internal_top100_mean_attention")
    parser.add_argument("--selected_topk", type=int, default=4)
    parser.add_argument(
        "--save_dir",
        default="results/observations/memory_core_topk_attention_topbottom_n2_o8",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
