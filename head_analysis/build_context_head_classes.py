"""Build memory/current head classes from eager previous-current profiling."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.context_denial import (
    build_head_classes_from_profile_csv,
    build_kv_head_classes_from_profile_csv,
    save_head_classes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_csv",
        default=(
            "results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full/"
            "head_profile_scores.csv"
        ),
    )
    parser.add_argument("--metric", default="b_log_per_token_ratio")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--head_granularity", choices=["query", "kv"], default="query")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
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
    parser.add_argument(
        "--output",
        default="results/observations/head_classes_prev_current/head_classes.json",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
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
    save_head_classes(head_classes, args.output)
    summary = {
        "output": args.output,
        "granularity": head_classes.get("granularity", args.head_granularity),
        "metric": head_classes["metric"],
        "quantile": head_classes["quantile"],
    }
    if head_classes.get("granularity") == "kv":
        summary.update({
            "aggregation": head_classes.get("aggregation"),
            "score_mode": head_classes.get("score_mode"),
            "num_query_heads": head_classes["num_query_heads"],
            "num_kv_heads": head_classes["num_kv_heads"],
            "group_size": head_classes["group_size"],
            "num_memory_oriented": len(head_classes["memory_kv_heads"]),
            "num_current_sensitive": len(head_classes["current_kv_heads"]),
            "num_mixed": len(head_classes["mixed_kv_heads"]),
        })
    else:
        summary.update({
            "num_memory_oriented": len(head_classes["memory_oriented"]),
            "num_current_sensitive": len(head_classes["current_sensitive"]),
            "num_mixed": len(head_classes["mixed"]),
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
