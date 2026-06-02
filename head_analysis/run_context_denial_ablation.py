r"""Run task-balanced StreamingBench context-access ablations.

Example:

python head_analysis/run_context_denial_ablation.py \
  --setting deny_memory_to_memory_heads \
  --max_questions_per_task 20 \
  --tasks Counting,Causal\ Reasoning,Attribute\ Recognition,Object\ Recognition,Prediction,Summarization \
  --device cuda:0 \
  --save_dir results/observations/obs_context_denial/deny_memory_to_memory_heads
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.context_denial import (
    apply_context_denial_ablation,
    build_head_classes_from_profile_csv,
    build_kv_head_classes_from_profile_csv,
    load_head_classes,
    save_head_classes,
)


def parse_csv_list(value):
    if value is None or str(value).strip() == "":
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def task_of(sample, video_sample=None):
    video_sample = video_sample or {}
    return sample.get("task", sample.get("question_type", video_sample.get("task", "Unknown")))


def filter_task_balanced(anno, tasks=None, max_questions_per_task=None, max_videos=None):
    task_set = set(tasks) if tasks else None
    counts = {}
    filtered = []

    for video_sample in anno:
        if max_videos is not None and len(filtered) >= max_videos:
            break

        new_conversations = []
        for sample in video_sample.get("conversations", []):
            task = task_of(sample, video_sample)
            if task_set is not None and task not in task_set:
                continue
            if max_questions_per_task is not None and counts.get(task, 0) >= max_questions_per_task:
                continue
            new_conversations.append(sample)
            counts[task] = counts.get(task, 0) + 1

        if new_conversations:
            copied = dict(video_sample)
            copied["conversations"] = new_conversations
            filtered.append(copied)

        if task_set is not None and max_questions_per_task is not None:
            if all(counts.get(task, 0) >= max_questions_per_task for task in task_set):
                break

    return filtered, counts


def ensure_head_classes(args):
    if args.head_classes and os.path.exists(args.head_classes):
        return load_head_classes(args.head_classes), args.head_classes

    if not args.profile_csv:
        raise ValueError("Provide --head_classes or --profile_csv")

    if args.head_granularity == "kv":
        head_classes = build_kv_head_classes_from_profile_csv(
            args.profile_csv,
            metric=args.metric,
            quantile=args.quantile,
            num_layers=args.num_layers,
            num_query_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            aggregation=args.kv_aggregation,
            score_mode=args.kv_score_mode,
        )
    else:
        head_classes = build_head_classes_from_profile_csv(
            args.profile_csv,
            metric=args.metric,
            quantile=args.quantile,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    output = args.head_classes or os.path.join(args.save_dir, "head_classes.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_head_classes(head_classes, output)
    return head_classes, output


def summarize_csv(csv_path, summary_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows or "qa_acc" not in rows[0]:
        summary = {"csv": csv_path, "n": int(len(rows)), "overall": None, "tasks": {}}
    else:
        acc_values = [float(row["qa_acc"]) / 100.0 for row in rows if row.get("qa_acc") not in ("", None)]
        overall = sum(acc_values) / len(acc_values) if acc_values else None
        task_values = {}
        for row in rows:
            if "task" not in row or row.get("qa_acc") in ("", None):
                continue
            task_values.setdefault(row["task"], []).append(float(row["qa_acc"]) / 100.0)
        tasks = {
            task: {"n": len(vals), "acc": sum(vals) / len(vals)}
            for task, vals in sorted(task_values.items())
            if vals
        }
        summary = {"csv": csv_path, "n": int(len(rows)), "overall": overall, "tasks": tasks}

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes", choices=["hermes", "streamingvlm"])
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--setting", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_mask_q_len", type=int, default=512)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--max_questions_per_task", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--head_granularity", choices=["query", "kv"], default="query")
    parser.add_argument("--kv_aggregation", choices=["mean", "median", "min", "max"], default="mean")
    parser.add_argument(
        "--kv_score_mode",
        choices=["aggregate", "pooled"],
        default="aggregate",
        help=(
            "aggregate: aggregate per-query-head profile scores inside each KV group; "
            "pooled: pool raw per-observation attention masses inside each KV group before scoring"
        ),
    )
    parser.add_argument("--head_classes", default=None)
    parser.add_argument(
        "--profile_csv",
        default=(
            "results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full/"
            "head_profile_scores.csv"
        ),
    )
    parser.add_argument("--metric", default="b_log_per_token_ratio")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    from video_qa.base import MODELS
    from video_qa.hermes_vqa import HermesVQA

    with open(args.anno_path) as f:
        anno = json.load(f)

    tasks = parse_csv_list(args.tasks)
    filtered_anno, task_counts = filter_task_balanced(
        anno,
        tasks=tasks,
        max_questions_per_task=args.max_questions_per_task,
        max_videos=args.max_videos,
    )
    filtered_anno_path = os.path.join(args.save_dir, "filtered_anno.json")
    with open(filtered_anno_path, "w") as f:
        json.dump(filtered_anno, f, indent=2)

    head_classes, head_classes_path = ensure_head_classes(args)

    model_path = MODELS[args.model]["model_path"]
    load_func = MODELS[args.model]["load_func"]
    print(f"[context_denial_runner] loading {args.model} from {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=args.streaming,
        device=args.device,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
    )

    videoqa_model = apply_context_denial_ablation(
        videoqa_model,
        head_classes=head_classes,
        setting=args.setting,
        seed=args.seed,
        num_query_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        max_mask_q_len=args.max_mask_q_len,
        head_granularity=args.head_granularity,
    )

    config = {
        "args": vars(args),
        "filtered_anno": filtered_anno_path,
        "task_counts": task_counts,
        "head_classes": head_classes_path,
        "context_denial_config": getattr(videoqa_model, "_context_denial_config", None),
    }
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    analyzer = HermesVQA(
        anno=filtered_anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        num_chunks=1,
        chunk_idx=0,
        save_dir=args.save_dir,
    )
    analyzer.analyze(debug=False)

    result_csv = os.path.join(args.save_dir, "1_0.csv")
    summary = summarize_csv(result_csv, os.path.join(args.save_dir, "summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
