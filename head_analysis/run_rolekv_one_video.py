"""Run a one-video RoleKV smoke test on StreamingBench."""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.rolekv_policy import apply_rolekv_policy


def summarize_csv(csv_path, summary_path):
    rows = []
    with open(csv_path, newline="") as f:
        rows.extend(csv.DictReader(f))
    acc_values = [
        float(row["qa_acc"]) / 100.0
        for row in rows
        if row.get("qa_acc") not in (None, "")
    ]
    tasks = {}
    for row in rows:
        if row.get("task") and row.get("qa_acc") not in (None, ""):
            tasks.setdefault(row["task"], []).append(float(row["qa_acc"]) / 100.0)
    summary = {
        "n": len(rows),
        "overall": sum(acc_values) / len(acc_values) if acc_values else None,
        "tasks": {
            task: {"n": len(vals), "acc": sum(vals) / len(vals)}
            for task, vals in sorted(tasks.items())
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--video_index", type=int, default=0)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes", choices=["hermes", "streamingvlm"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head_classes", default="results/observations/head_classes_prev_current/head_classes.json")
    parser.add_argument("--rolekv_mode", default="rolekv", choices=["rolekv", "baseline", "random", "inverted"])
    parser.add_argument("--lambda_memory", type=float, default=0.2)
    parser.add_argument("--lambda_current", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from video_qa.base import MODELS
    from video_qa.hermes_vqa import HermesVQA

    with open(args.anno_path) as f:
        anno_all = json.load(f)
    anno = [anno_all[int(args.video_index)]]
    filtered_anno_path = os.path.join(args.save_dir, "one_video_anno.json")
    with open(filtered_anno_path, "w") as f:
        json.dump(anno, f, indent=2)

    model_path = MODELS[args.model]["model_path"]
    load_func = MODELS[args.model]["load_func"]
    print(f"[RoleKV runner] loading {args.model} from {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=True,
        device=args.device,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
    )
    videoqa_model = apply_rolekv_policy(
        videoqa_model,
        head_classes_path=args.head_classes,
        mode=args.rolekv_mode,
        lambda_memory=args.lambda_memory,
        lambda_current=args.lambda_current,
        seed=args.seed,
    )

    with open(os.path.join(args.save_dir, "run_config.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "anno": filtered_anno_path,
            "rolekv_config": getattr(videoqa_model, "_rolekv_config", None),
        }, f, indent=2)

    analyzer = HermesVQA(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        num_chunks=1,
        chunk_idx=0,
        save_dir=args.save_dir,
    )
    analyzer.analyze(debug=False)

    csv_path = os.path.join(args.save_dir, "1_0.csv")
    summary = summarize_csv(csv_path, os.path.join(args.save_dir, "summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
