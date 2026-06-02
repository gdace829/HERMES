"""Run a task-balanced RoleKV subset on StreamingBench.

This runner is intentionally small and deterministic.  It filters the
annotation file once by task quotas, then evaluates one mode on that exact
subset.  Launch the same arguments for baseline/rolekv/random/inverted to get
paired results.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def summarize_annotation(anno):
    task_counts = Counter()
    video_task_counts = []
    for video_sample in anno:
        local = Counter()
        for sample in video_sample.get("conversations", []):
            task = task_of(sample, video_sample)
            task_counts[task] += 1
            local[task] += 1
        video_task_counts.append(
            {
                "video_id": video_sample.get("video_id"),
                "num_questions": sum(local.values()),
                "tasks": dict(sorted(local.items())),
            }
        )
    return {
        "num_videos": len(anno),
        "num_questions": sum(task_counts.values()),
        "task_counts": dict(sorted(task_counts.items())),
        "videos": video_task_counts,
    }


def summarize_csv(csv_path, summary_path):
    rows = []
    with open(csv_path, newline="") as f:
        rows.extend(csv.DictReader(f))

    acc_values = [
        float(row["qa_acc"]) / 100.0
        for row in rows
        if row.get("qa_acc") not in (None, "")
    ]
    task_values = {}
    for row in rows:
        if row.get("task") and row.get("qa_acc") not in (None, ""):
            task_values.setdefault(row["task"], []).append(float(row["qa_acc"]) / 100.0)

    summary = {
        "csv": csv_path,
        "n": len(rows),
        "overall": sum(acc_values) / len(acc_values) if acc_values else None,
        "tasks": {
            task: {"n": len(vals), "acc": sum(vals) / len(vals)}
            for task, vals in sorted(task_values.items())
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--filtered_anno", default=None)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes", choices=["hermes", "streamingvlm"])
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rolekv_mode",
        default="rolekv",
        choices=["baseline", "rolekv", "random", "inverted", "norole"],
        help=(
            "baseline leaves the original compression unchanged; norole installs "
            "RoleKV's voting scaffold without role bonuses."
        ),
    )
    parser.add_argument("--lambda_memory", type=float, default=0.2)
    parser.add_argument("--lambda_current", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--max_questions_per_task", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument(
        "--head_classes",
        default="results/observations/head_classes_prev_current/head_classes.json",
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.filtered_anno:
        with open(args.filtered_anno) as f:
            filtered_anno = json.load(f)
        filter_counts = summarize_annotation(filtered_anno)["task_counts"]
    else:
        with open(args.anno_path) as f:
            anno = json.load(f)
        tasks = parse_csv_list(args.tasks)
        filtered_anno, filter_counts = filter_task_balanced(
            anno,
            tasks=tasks,
            max_questions_per_task=args.max_questions_per_task,
            max_videos=args.max_videos,
        )

    filtered_anno_path = os.path.join(args.save_dir, "filtered_anno.json")
    with open(filtered_anno_path, "w") as f:
        json.dump(filtered_anno, f, indent=2)

    anno_summary = summarize_annotation(filtered_anno)
    config = {
        "args": vars(args),
        "filtered_anno": filtered_anno_path,
        "filter_counts": filter_counts,
        "annotation_summary": anno_summary,
    }
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("[RoleKV subset] selected annotation:")
    print(json.dumps(anno_summary, indent=2))
    if args.dry_run:
        return

    from head_analysis.rolekv_policy import apply_rolekv_policy
    from video_qa.base import MODELS
    from video_qa.hermes_vqa import HermesVQA

    model_path = MODELS[args.model]["model_path"]
    load_func = MODELS[args.model]["load_func"]
    print(f"[RoleKV subset] loading {args.model} from {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=args.streaming,
        device=args.device,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
    )

    if args.rolekv_mode != "baseline":
        policy_mode = "baseline" if args.rolekv_mode == "norole" else args.rolekv_mode
        videoqa_model = apply_rolekv_policy(
            videoqa_model,
            head_classes_path=args.head_classes,
            mode=policy_mode,
            lambda_memory=args.lambda_memory,
            lambda_current=args.lambda_current,
            seed=args.seed,
        )
        config["rolekv_config"] = getattr(videoqa_model, "_rolekv_config", None)
    else:
        config["rolekv_config"] = None

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
    print("[RoleKV subset] result summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
