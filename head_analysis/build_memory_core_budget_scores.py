"""Build KV-head budget score CSVs from memory-core profiling results."""

import argparse
import os

import numpy as np
import pandas as pd


def normalize_positive(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if values.max() <= eps:
        return np.ones_like(values, dtype=np.float64)
    return values


def write_scores(df, out_path, scores, score_name):
    out = df[["layer", "kv_head"]].copy()
    out["score"] = np.asarray(scores, dtype=np.float64)
    out[score_name] = np.asarray(scores, dtype=np.float64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_csv",
        default="results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv",
    )
    parser.add_argument("--metric", default="internal_top100_mean_attention")
    parser.add_argument("--save_dir", default="results/observations/memory_core_budget_scores")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.profile_csv)
    required = {"layer", "kv_head", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.profile_csv} missing columns: {sorted(missing)}")
    df = df.sort_values(["layer", "kv_head"]).reset_index(drop=True)

    expected = int(args.num_layers) * int(args.num_kv_heads)
    if len(df) != expected:
        raise ValueError(f"Expected {expected} layer-KV rows, got {len(df)}")

    scores = normalize_positive(df[args.metric].to_numpy())
    os.makedirs(args.save_dir, exist_ok=True)
    write_scores(df, os.path.join(args.save_dir, "ours_top100.csv"), scores, args.metric)
    write_scores(df, os.path.join(args.save_dir, "uniform.csv"), np.ones_like(scores), args.metric)

    # Invert within each layer so every layer keeps the same score multiset.
    inv = np.zeros_like(scores)
    for layer in range(int(args.num_layers)):
        idx = np.where(df["layer"].to_numpy() == layer)[0]
        vals = scores[idx]
        inv[idx] = vals.max() + vals.min() - vals
    inv = normalize_positive(inv)
    write_scores(df, os.path.join(args.save_dir, "inverted_top100.csv"), inv, args.metric)

    # Random layer-matched control: permute the four KV-head scores within each layer.
    rng = np.random.default_rng(int(args.seed))
    rnd = np.zeros_like(scores)
    for layer in range(int(args.num_layers)):
        idx = np.where(df["layer"].to_numpy() == layer)[0]
        rnd[idx] = rng.permutation(scores[idx])
    write_scores(
        df,
        os.path.join(args.save_dir, f"random_top100_seed{int(args.seed)}.csv"),
        rnd,
        args.metric,
    )


if __name__ == "__main__":
    main()
