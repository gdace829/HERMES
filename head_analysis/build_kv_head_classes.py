"""Build layer-KV-head role classes from previous/current profiling results."""

import argparse
import csv
import json
import math
import os

import numpy as np
import pandas as pd


EPS = 1e-12


def _quantile_classes_from_scores(rows, quantile):
    rows = sorted(rows, key=lambda item: item["score"])
    total = len(rows)
    k = max(1, int(math.ceil(total * float(quantile))))
    k = min(k, total // 2)
    memory = rows[:k]
    current = rows[-k:]
    selected = {(row["layer"], row["kv_head"]) for row in memory + current}
    mixed = [
        row
        for row in rows
        if (row["layer"], row["kv_head"]) not in selected
    ]
    return memory, current, mixed


def _rows_from_profile(profile_csv, metric):
    rows = []
    with open(profile_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer", "kv_head", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{profile_csv} is missing columns: {sorted(missing)}")
        for row in reader:
            value = float(row[metric])
            if math.isfinite(value):
                rows.append(
                    {
                        "layer": int(row["layer"]),
                        "kv_head": int(row["kv_head"]),
                        "score": value,
                        "s_current_share": float(row.get("s_current_share", "nan")),
                        "r_per_token_ratio": float(row.get("r_per_token_ratio", "nan")),
                    }
                )
    return rows


def _rows_from_raw_median(raw_kv_csv):
    df = pd.read_csv(raw_kv_csv)
    local_ratio = np.maximum(
        df["local_current_to_prev_per_token_ratio"].to_numpy(dtype=np.float64),
        EPS,
    )
    global_ratio = np.maximum(
        df["global_current_to_prev_per_token_ratio"].to_numpy(dtype=np.float64),
        EPS,
    )
    df = df.copy()
    df["s_obs"] = 0.5 * (df["local_current_share"] + df["global_current_share"])
    df["b_obs"] = 0.5 * (np.log(local_ratio) + np.log(global_ratio))
    df["r_obs"] = np.exp(df["b_obs"])
    grouped = (
        df.groupby(["layer", "kv_head"])
        .agg(
            score=("b_obs", "median"),
            s_current_share=("s_obs", "median"),
            r_per_token_ratio=("r_obs", "median"),
            num_observations=("b_obs", "size"),
        )
        .reset_index()
    )
    return [
        {
            "layer": int(row.layer),
            "kv_head": int(row.kv_head),
            "score": float(row.score),
            "s_current_share": float(row.s_current_share),
            "r_per_token_ratio": float(row.r_per_token_ratio),
            "num_observations": int(row.num_observations),
        }
        for row in grouped.itertuples(index=False)
        if math.isfinite(float(row.score))
    ]


def build_classes(profile_csv=None, raw_kv_csv=None, output=None,
                  quantile=0.2, metric="b_log_per_token_ratio",
                  pooling="profile_mean"):
    if pooling == "pooled_median":
        if not raw_kv_csv:
            raise ValueError("--raw_kv_csv is required when --pooling=pooled_median")
        rows = _rows_from_raw_median(raw_kv_csv)
        source = raw_kv_csv
        score_metric = "median_b_obs"
    else:
        if not profile_csv:
            raise ValueError("--profile_csv is required when --pooling=profile_mean")
        rows = _rows_from_profile(profile_csv, metric)
        source = profile_csv
        score_metric = metric

    memory, current, mixed = _quantile_classes_from_scores(rows, quantile)
    num_layers = max(row["layer"] for row in rows) + 1 if rows else 0
    num_kv_heads = max(row["kv_head"] for row in rows) + 1 if rows else 0

    data = {
        "source": source,
        "profile_csv": profile_csv,
        "raw_kv_csv": raw_kv_csv,
        "pooling": pooling,
        "metric": score_metric,
        "quantile": float(quantile),
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "memory_oriented_kv": [[r["layer"], r["kv_head"]] for r in memory],
        "current_sensitive_kv": [[r["layer"], r["kv_head"]] for r in current],
        "mixed_kv": [[r["layer"], r["kv_head"]] for r in mixed],
        "scores": [
            {
                "layer": r["layer"],
                "kv_head": r["kv_head"],
                "score": r["score"],
                "s_current_share": r.get("s_current_share"),
                "r_per_token_ratio": r.get("r_per_token_ratio"),
            }
            for r in sorted(rows, key=lambda item: (item["layer"], item["kv_head"]))
        ],
        "counts": {
            "memory_oriented_kv": len(memory),
            "current_sensitive_kv": len(current),
            "mixed_kv": len(mixed),
            "total": len(rows),
        },
    }

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_csv", default=None)
    parser.add_argument("--raw_kv_csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--metric", default="b_log_per_token_ratio")
    parser.add_argument(
        "--pooling",
        default="pooled_median",
        choices=["profile_mean", "pooled_median"],
    )
    args = parser.parse_args()
    data = build_classes(
        profile_csv=args.profile_csv,
        raw_kv_csv=args.raw_kv_csv,
        output=args.output,
        quantile=args.quantile,
        metric=args.metric,
        pooling=args.pooling,
    )
    print(json.dumps({
        "output": args.output,
        "counts": data["counts"],
        "pooling": data["pooling"],
        "metric": data["metric"],
    }, indent=2))


if __name__ == "__main__":
    main()
