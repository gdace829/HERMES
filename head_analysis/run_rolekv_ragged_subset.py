"""Run a task-balanced RoleKV-v2 ragged subset on StreamingBench.

This runner evaluates the physically meaningful method path:

  baseline: per-KV-head ragged retention with the configured budget table;
  rolekv:   same budget table, but RoleKV quota selection by profiled roles;
  random:   layer-matched random roles with the same quota mechanism;
  inverted: profiled memory/current roles swapped.

The subset filter is shared with ``run_rolekv_subset.py`` so every mode can use
the exact same videos/questions.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.run_rolekv_subset import (  # noqa: E402
    filter_task_balanced,
    parse_csv_list,
    summarize_annotation,
    summarize_csv,
)


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
        "--mode",
        default="rolekv",
        choices=["baseline", "rolekv", "rolekv_quota", "random", "random_quota", "inverted", "inverted_quota", "uniform", "norole"],
        help="Paired comparison mode. 'rolekv' maps to the stronger rolekv_quota policy.",
    )
    parser.add_argument(
        "--head_classes",
        default="results/observations/head_classes_prev_current/head_classes.json",
    )
    parser.add_argument(
        "--kv_profile_scores",
        default=None,
        help=(
            "Optional layer-KV-head profile CSV. If set, roles are built directly "
            "from [layer, kv_head] scores instead of query-head classes."
        ),
    )
    parser.add_argument("--kv_profile_metric", default="b_log_per_token_ratio")
    parser.add_argument("--kv_profile_quantile", type=float, default=0.2)
    parser.add_argument("--quota_ratio", type=float, default=0.7)
    parser.add_argument("--lambda_memory", type=float, default=0.2)
    parser.add_argument("--lambda_current", type=float, default=0.2)
    parser.add_argument("--role_min_votes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--max_questions_per_task", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--kv_head_budget_scores", default="sparsemm_qwen25")
    parser.add_argument(
        "--kv_head_budget_scheme",
        default="sparsemm",
        choices=[
            "relative",
            "sparsemm",
            "sparsemm_layer_total",
            "sparsemm_layer",
            "sparsemm_total",
            "sparsemm_per_layer_total",
            "sparsemm_layer_exact",
            "sparsemm_per_layer",
        ],
    )
    parser.add_argument("--kv_head_budget_sparsemm_ratio", type=float, default=0.1)
    parser.add_argument("--kv_head_budget_sparsemm_window_size", type=int, default=32)
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
        filtered_anno, filter_counts = filter_task_balanced(
            anno,
            tasks=parse_csv_list(args.tasks),
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

    print("[RoleKV-ragged subset] selected annotation:")
    print(json.dumps(anno_summary, indent=2))
    if args.dry_run:
        return

    from head_analysis.hermes_kv_head_budget import apply_kv_head_budget
    from head_analysis.hermes_kv_head_ragged_prefill import apply_kv_head_ragged_prefill
    from head_analysis.rolekv_ragged_policy import apply_rolekv_ragged_policy
    from video_qa.base import MODELS
    from video_qa.hermes_vqa import HermesVQA

    model_path = MODELS[args.model]["model_path"]
    load_func = MODELS[args.model]["load_func"]
    print(f"[RoleKV-ragged subset] loading {args.model} from {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=args.streaming,
        device=args.device,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
    )

    language_model = getattr(videoqa_model, "language_model", None)
    language_config = getattr(language_model, "config", None)
    num_query_heads = getattr(language_config, "num_attention_heads", None)
    num_kv_heads = getattr(language_config, "num_key_value_heads", None)
    if num_query_heads is None and hasattr(language_model, "model"):
        num_query_heads = getattr(language_model.model.config, "num_attention_heads", None)
    if num_kv_heads is None and hasattr(language_model, "model"):
        num_kv_heads = getattr(language_model.model.config, "num_key_value_heads", None)
    num_query_heads = int(num_query_heads or 28)
    num_kv_heads = int(num_kv_heads or 4)

    videoqa_model = apply_kv_head_budget(
        videoqa_model,
        kv_head_scores_path=args.kv_head_budget_scores,
        num_layers=videoqa_model.num_layers,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        budget_scheme=args.kv_head_budget_scheme,
        sparsemm_ratio=args.kv_head_budget_sparsemm_ratio,
        sparsemm_window_size=args.kv_head_budget_sparsemm_window_size,
    )
    videoqa_model = apply_kv_head_ragged_prefill(videoqa_model)

    if args.mode != "baseline":
        policy_mode = "rolekv_quota" if args.mode == "rolekv" else args.mode
        videoqa_model = apply_rolekv_ragged_policy(
            videoqa_model,
            head_classes_path=args.head_classes,
            mode=policy_mode,
            quota_ratio=args.quota_ratio,
            lambda_memory=args.lambda_memory,
            lambda_current=args.lambda_current,
            seed=args.seed,
            num_layers=videoqa_model.num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            role_min_votes=args.role_min_votes,
            kv_profile_path=args.kv_profile_scores,
            kv_profile_metric=args.kv_profile_metric,
            kv_profile_quantile=args.kv_profile_quantile,
        )

    config["kv_head_budget_config"] = getattr(videoqa_model, "_kv_head_budget_config", None)
    config["rolekv_ragged_config"] = getattr(videoqa_model, "_rolekv_ragged_config", None)
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
    print("[RoleKV-ragged subset] result summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
