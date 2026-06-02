"""Boundary-aware internal-memory selector profiling.

This script measures whether compression-time pseudo-query attention can select
internal memory tokens that later receive true-query readout attention.

It is an offline profiling tool:
  1. Encode each streaming video chunk normally.
  2. Before HERMES compression, collect local/global pseudo-query attention.
  3. Collect the current sample's true question attention at the same cache state.
  4. For each layer-KV-head group, select top-K internal-memory tokens by the
     pseudo-query selector and measure their future-query attention coverage.
  5. Continue with the normal online compression path.

The true question is never used to update the cache or to make online eviction
decisions. It is only used as an offline cache-access proxy.
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


def mean_query_attention(attn):
    """Average attention over query tokens, returning [num_query_heads, kv_len]."""
    if attn is None or attn.dim() < 4:
        return None
    return torch.nan_to_num(
        attn[0].mean(dim=1).float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def attention_coverage(target, indices):
    if target.numel() == 0 or indices.numel() == 0:
        return 0.0
    denom = float(target.sum().item())
    if denom <= EPS:
        return 0.0
    return float(target.index_select(0, indices).sum().item() / (denom + EPS))


def random_coverage(target, k, random_trials, rng):
    n = int(target.numel())
    k = min(int(k), n)
    if n <= 0 or k <= 0:
        return 0.0
    vals = []
    for _ in range(int(random_trials)):
        sampled = rng.sample(range(n), k)
        indices = torch.as_tensor(sampled, device=target.device, dtype=torch.long)
        vals.append(attention_coverage(target, indices))
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_internal_memory_coverage(
    selector_local,
    selector_global,
    future_attn,
    layer_idx,
    kv_head,
    q_heads,
    visual_start,
    pre_len,
    post_len,
    cached_kv_len,
    boundary_window,
    top_k,
    random_trials,
    rng,
    future_head_pool,
):
    """Compute one layer-KV-head observation.

    Internal memory excludes the boundary window immediately before pre_len:
      [visual_start, pre_len - boundary_window)
    """
    if selector_local is None or selector_global is None or future_attn is None:
        return None

    kv_len = min(
        int(cached_kv_len),
        int(selector_local.shape[-1]),
        int(selector_global.shape[-1]),
        int(future_attn.shape[-1]),
    )
    mem_start = min(int(visual_start), kv_len)
    mem_end = min(max(int(pre_len), mem_start), kv_len)
    curr_end = min(max(int(post_len), mem_end), kv_len)

    int_end = max(mem_start, mem_end - int(boundary_window))
    bnd_start = int_end
    bnd_end = mem_end

    num_memory = mem_end - mem_start
    num_boundary = bnd_end - bnd_start
    num_internal = int_end - mem_start
    if num_internal <= 0 or not q_heads:
        return None

    q_idx = torch.as_tensor(q_heads, device=selector_local.device, dtype=torch.long)
    # Eager attention with past_key_values can include the current text query
    # tokens in the key axis. Local/global/future prompts have different text
    # lengths, so always truncate to the common cached-KV prefix before mixing.
    selector_local_group = selector_local.index_select(0, q_idx)[:, :kv_len]
    selector_global_group = selector_global.index_select(0, q_idx)[:, :kv_len]
    selector = 0.5 * selector_local_group + 0.5 * selector_global_group
    selector_group = selector.mean(dim=0)[:kv_len]

    selector_int = selector_group[mem_start:int_end]
    selector_mem = selector_group[mem_start:mem_end]
    selector_boundary = selector_group[bnd_start:bnd_end]
    if selector_int.numel() <= 0:
        return None

    if future_head_pool == "same_kv":
        future_group = future_attn.index_select(0, q_idx).mean(dim=0)[:kv_len]
    elif future_head_pool == "all":
        future_group = future_attn.mean(dim=0)[:kv_len]
    else:
        raise ValueError(f"Unknown future_head_pool={future_head_pool}")

    future_int = torch.clamp(future_group[mem_start:int_end], min=0.0)
    k_eff = min(int(top_k), int(selector_int.numel()))
    if k_eff <= 0:
        return None

    selected = torch.topk(selector_int, k=k_eff, sorted=False).indices
    oracle = torch.topk(future_int, k=k_eff, sorted=False).indices

    cov_int = attention_coverage(future_int, selected)
    cov_rand_int = random_coverage(future_int, k_eff, random_trials, rng)
    cov_oracle_int = attention_coverage(future_int, oracle)
    u_obs_int = (cov_int - cov_rand_int) / (cov_oracle_int - cov_rand_int + EPS)

    selector_memory_mass = float(selector_mem.sum().item())
    selector_boundary_mass = float(selector_boundary.sum().item()) if selector_boundary.numel() else 0.0
    selector_internal_mass = float(selector_int.sum().item())

    return {
        "layer": int(layer_idx),
        "kv_head": int(kv_head),
        "visual_start": int(visual_start),
        "pre_len": int(pre_len),
        "post_len": int(post_len),
        "cached_kv_len": int(cached_kv_len),
        "num_memory_tokens": int(num_memory),
        "num_boundary_tokens": int(num_boundary),
        "num_internal_tokens": int(num_internal),
        "num_latest_tokens": int(max(0, curr_end - mem_end)),
        "top_k_eff": int(k_eff),
        "cov_int": float(cov_int),
        "cov_rand_int": float(cov_rand_int),
        "cov_oracle_int": float(cov_oracle_int),
        "u_obs_int": float(u_obs_int),
        "boundary_ratio": float(selector_boundary_mass / (selector_memory_mass + EPS)),
        "future_mass_internal": float(future_int.sum().item()),
        "selector_mass_internal": float(selector_internal_mass),
        "selector_mass_boundary": float(selector_boundary_mass),
        "selector_mass_memory": float(selector_memory_mass),
    }


def quantile_classes(score_rows, quantile):
    valid = [
        row
        for row in score_rows
        if row.get("u_int_mean") is not None and math.isfinite(float(row["u_int_mean"]))
    ]
    valid = sorted(valid, key=lambda row: float(row["u_int_mean"]))
    total = len(valid)
    if total == 0:
        return [], [], []
    k = max(1, int(math.ceil(total * float(quantile))))
    k = min(k, total // 2)
    weak = valid[:k]
    memory_selector = valid[-k:]
    selected = {(row["layer"], row["kv_head"]) for row in weak + memory_selector}
    mixed = [row for row in valid if (row["layer"], row["kv_head"]) not in selected]
    return memory_selector, weak, mixed


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
                "u_int_mean": finite_mean([r["u_obs_int"] for r in part]),
                "u_int_median": finite_median([r["u_obs_int"] for r in part]),
                "cov_int_mean": finite_mean([r["cov_int"] for r in part]),
                "cov_rand_int_mean": finite_mean([r["cov_rand_int"] for r in part]),
                "cov_oracle_int_mean": finite_mean([r["cov_oracle_int"] for r in part]),
                "boundary_ratio_mean": finite_mean([r["boundary_ratio"] for r in part]),
                "future_mass_internal_mean": finite_mean([r["future_mass_internal"] for r in part]),
                "selector_mass_internal_mean": finite_mean([r["selector_mass_internal"] for r in part]),
                "num_observations": len(part),
            }
        )

    memory_selector, weak_selector, mixed = quantile_classes(score_rows, args.quantile)
    classes = {
        "granularity": "kv",
        "score": "boundary_aware_internal_memory_utility",
        "score_column": "u_int_mean",
        "quantile": float(args.quantile),
        "boundary_window": int(args.boundary_window),
        "top_k": int(args.top_k),
        "future_head_pool": args.future_head_pool,
        "num_query_heads": int(args.num_query_heads),
        "num_kv_heads": int(args.num_kv_heads),
        "memory_selector_heads": [[r["layer"], r["kv_head"]] for r in memory_selector],
        "weak_selector_heads": [[r["layer"], r["kv_head"]] for r in weak_selector],
        "mixed_heads": [[r["layer"], r["kv_head"]] for r in mixed],
        "counts": {
            "memory_selector_heads": len(memory_selector),
            "weak_selector_heads": len(weak_selector),
            "mixed_heads": len(mixed),
            "total": len(score_rows),
        },
        "scores": [
            {
                "layer": r["layer"],
                "kv_head": r["kv_head"],
                "u_int_mean": r["u_int_mean"],
                "u_int_median": r["u_int_median"],
                "boundary_ratio_mean": r["boundary_ratio_mean"],
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
        (
            "u_int_mean",
            "internal_memory_utility_heatmap.png",
            "Boundary-aware internal-memory utility",
            "viridis",
        ),
        (
            "boundary_ratio_mean",
            "boundary_ratio_heatmap.png",
            "Selector mass near memory boundary",
            "magma",
        ),
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

    valid_rows = [r for r in score_rows if r["u_int_mean"] is not None and r["boundary_ratio_mean"] is not None]
    if valid_rows:
        u = np.asarray([r["u_int_mean"] for r in valid_rows], dtype=np.float64)
        b = np.asarray([r["boundary_ratio_mean"] for r in valid_rows], dtype=np.float64)
        labels = [(r["layer"], r["kv_head"]) for r in valid_rows]
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.scatter(b, u, s=24, alpha=0.78, color="#4C78A8")
        ax.set_xlabel("Mean boundary mass ratio")
        ax.set_ylabel("Mean internal-memory utility")
        ax.set_title("Utility vs boundary concentration")
        for idx in np.argsort(u)[-4:]:
            ax.text(b[idx], u[idx], f"L{labels[idx][0]}-K{labels[idx][1]}", fontsize=8)
        fig.tight_layout()
        path = os.path.join(save_dir, "utility_vs_boundary_ratio_scatter.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs["utility_vs_boundary_ratio_scatter"] = path

    sorted_rows = sorted(
        [r for r in score_rows if r["u_int_mean"] is not None],
        key=lambda item: item["u_int_mean"],
    )
    if sorted_rows:
        k = max(1, min(4, len(sorted_rows) // 3 if len(sorted_rows) >= 3 else 1))
        bottom = sorted_rows[:k]
        top = sorted_rows[-k:]
        middle_start = max(0, len(sorted_rows) // 2 - k // 2)
        middle = sorted_rows[middle_start:middle_start + k]
        bars = [
            ("weak", finite_mean([r["cov_int_mean"] for r in bottom]), finite_mean([r["cov_rand_int_mean"] for r in bottom])),
            ("middle", finite_mean([r["cov_int_mean"] for r in middle]), finite_mean([r["cov_rand_int_mean"] for r in middle])),
            ("top", finite_mean([r["cov_int_mean"] for r in top]), finite_mean([r["cov_rand_int_mean"] for r in top])),
        ]
        x = np.arange(len(bars))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar(x - width / 2, [v[1] or 0.0 for v in bars], width=width, label="selector", color="#4C78A8")
        ax.bar(x + width / 2, [v[2] or 0.0 for v in bars], width=width, label="random", color="#F58518")
        ax.set_xticks(x, [v[0] for v in bars])
        ax.set_ylabel("Internal future-attention coverage")
        ax.set_title("Selected-token coverage vs random")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = os.path.join(save_dir, "top_random_bottom_coverage_bar.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs["top_random_bottom_coverage_bar"] = path

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
        "num_memory_tokens",
        "num_boundary_tokens",
        "num_internal_tokens",
        "num_latest_tokens",
        "top_k_eff",
        "cov_int",
        "cov_rand_int",
        "cov_oracle_int",
        "u_obs_int",
        "boundary_ratio",
        "future_mass_internal",
        "selector_mass_internal",
        "selector_mass_boundary",
        "selector_mass_memory",
    ]
    raw_csv = os.path.join(args.save_dir, "raw_internal_memory_coverage.csv")
    write_csv(raw_csv, rows, raw_fields)

    score_rows, classes = aggregate_rows(rows, args)
    score_fields = [
        "layer",
        "kv_head",
        "u_int_mean",
        "u_int_median",
        "cov_int_mean",
        "cov_rand_int_mean",
        "cov_oracle_int_mean",
        "boundary_ratio_mean",
        "future_mass_internal_mean",
        "selector_mass_internal_mean",
        "num_observations",
    ]
    score_csv = os.path.join(args.save_dir, "internal_memory_selector_scores.csv")
    write_csv(score_csv, score_rows, score_fields)

    classes_path = os.path.join(args.save_dir, "head_classes_internal_memory_selector.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)

    plots = plot_outputs(score_rows, args.save_dir, args.num_layers, args.num_kv_heads)
    summary = {
        "attention_source": "eager_output_attentions_for_pseudo_and_future_query",
        "future_attention_usage": "offline_profile_only_not_used_for_online_eviction",
        "num_chunk_observations": int(obs_count),
        "num_layer_kv_observations": int(len(rows)),
        "num_layers": int(args.num_layers),
        "num_kv_heads": int(args.num_kv_heads),
        "num_query_heads": int(args.num_query_heads),
        "top_k": int(args.top_k),
        "boundary_window": int(args.boundary_window),
        "future_head_pool": args.future_head_pool,
        "raw_csv": raw_csv,
        "score_csv": score_csv,
        "head_classes_json": classes_path,
        "figures": plots,
        "mean_u_int": finite_mean([r["u_obs_int"] for r in rows]),
        "median_u_int": finite_median([r["u_obs_int"] for r in rows]),
        "mean_cov_int": finite_mean([r["cov_int"] for r in rows]),
        "mean_cov_rand_int": finite_mean([r["cov_rand_int"] for r in rows]),
        "mean_cov_oracle_int": finite_mean([r["cov_oracle_int"] for r in rows]),
        "mean_boundary_ratio": finite_mean([r["boundary_ratio"] for r in rows]),
        "class_counts": classes["counts"],
    }
    summary_path = os.path.join(args.save_dir, "internal_memory_selector_summary.json")
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
        save_dir="/tmp/internal_memory_selector_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    rows = []
    obs_count = 0
    rng = random.Random(args.seed)

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

                local_q, global_q = model.predict_next_question()
                attn_local = model.compute_eager_attentions(tokenize(processor, local_q, model.device))
                attn_global = model.compute_eager_attentions(tokenize(processor, global_q, model.device))
                future_prompt = build_future_prompt(analyzer, model, sample)
                attn_future = model.compute_eager_attentions(tokenize(processor, future_prompt, model.device))

                visual_start = int(model.visual_start_idx)
                pos_cache = getattr(model, "_position_ids_cache", [])
                valid_rows_this_obs = 0
                for layer_idx, (al, ag, af) in enumerate(zip(attn_local, attn_global, attn_future)):
                    selector_local = mean_query_attention(al)
                    selector_global = mean_query_attention(ag)
                    future = mean_query_attention(af)
                    if selector_local is None or selector_global is None or future is None:
                        continue

                    if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                        cached_kv_len = int(pos_cache[layer_idx].shape[1])
                    else:
                        cached_kv_len = int(min(selector_local.shape[-1], selector_global.shape[-1], future.shape[-1]))

                    for kv_head in range(int(args.num_kv_heads)):
                        q_heads = query_heads_for_kv(kv_head, args.num_query_heads, args.num_kv_heads)
                        row = compute_internal_memory_coverage(
                            selector_local=selector_local,
                            selector_global=selector_global,
                            future_attn=future,
                            layer_idx=layer_idx,
                            kv_head=kv_head,
                            q_heads=q_heads,
                            visual_start=visual_start,
                            pre_len=pre_lens[layer_idx],
                            post_len=post_lens[layer_idx],
                            cached_kv_len=cached_kv_len,
                            boundary_window=args.boundary_window,
                            top_k=args.top_k,
                            random_trials=args.random_trials,
                            rng=rng,
                            future_head_pool=args.future_head_pool,
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
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--boundary_window", type=int, default=64)
    parser.add_argument("--random_trials", type=int, default=32)
    parser.add_argument("--future_head_pool", choices=["same_kv", "all"], default="same_kv")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        default="results/observations/internal_memory_selector_n4_o80",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
