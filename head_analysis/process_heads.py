"""
Head 数据处理（SparseMM 风格）

输入: raw_obs.csv (从 analyze_heads_pseudo.py 采集的原始观测)
输出: head_scores.npz, head_scatter.png, head_heatmap.png

不做推理，纯数据处理。
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def process(data_path, save_dir, num_layers=28, num_heads=28):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} observations, columns: {list(df.columns)}")

    # 按层+头聚合
    cols = [c for c in df.columns if c not in ('layer', 'head', 'n_visual')]

    agg_mean = df.groupby(['layer', 'head'])[cols].mean()
    agg_std  = df.groupby(['layer', 'head'])[cols].std()
    agg_count = df.groupby(['layer', 'head']).size()

    # 转矩阵
    def to_matrix(series, fill=0):
        mat = np.full((num_layers, num_heads), fill, dtype=float)
        for (l, h), v in series.items():
            mat[l, h] = v
        return mat

    shift = to_matrix(agg_mean.get('shift', pd.Series(dtype=float)))
    local_early = to_matrix(agg_mean.get('local_early_ratio', pd.Series(dtype=float)), fill=0.5)
    global_early = to_matrix(agg_mean.get('global_early_ratio', pd.Series(dtype=float)), fill=0.5)
    count = to_matrix(agg_count)

    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. Scatter: shift vs consistency ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 用 shift 的绝对值作为 "specialization", global_early 作为 "长期倾向"
    x = global_early.ravel()   # x: 对全局问题的早期关注度
    y = local_early.ravel()    # y: 对局部问题的早期关注度
    c = np.array(['red' if shift.ravel()[i] > 0.02 else
                   'blue' if shift.ravel()[i] < -0.02 else 'gray'
                   for i in range(len(x))])

    ax = axes[0]
    ax.scatter(x[count.ravel() > 0], y[count.ravel() > 0],
               c=c[count.ravel() > 0], alpha=0.6, s=20)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Global Early Ratio')
    ax.set_ylabel('Local Early Ratio')
    ax.set_title('Head Temporal Specialization\nRed=Long-term, Blue=Short-term, Gray=Neutral')

    # ---- 2. Heatmap: shift ----
    ax = axes[1]
    im = ax.imshow(shift, cmap='RdBu_r', aspect='auto', origin='lower',
                   vmin=-0.08, vmax=0.08)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Shift (global_early - local_early)')
    plt.colorbar(im, ax=ax)

    # ---- 3. Per-layer shift distribution ----
    ax = axes[2]
    for l in range(num_layers):
        vals = shift[l][count[l] > 0]
        if len(vals) > 0:
            ax.scatter([l] * len(vals), vals, c='gray', alpha=0.3, s=10)
            ax.scatter([l], [vals.mean()], c='red', s=30, zorder=5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Shift')
    ax.set_title('Per-Layer Shift Distribution\nRed dot = layer mean')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "head_analysis.png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/head_analysis.png")

    # ---- 4. 头分类表 ----
    mask = count > max(5, count.max() * 0.3)  # 样本够多的头

    # 长期头: shift 显著为正
    long_term_heads = [(l, h, shift[l,h], global_early[l,h], local_early[l,h])
                       for l in range(num_layers) for h in range(num_heads)
                       if mask[l,h] and shift[l,h] > 0.02]
    long_term_heads.sort(key=lambda x: -x[2])

    # 短期头: shift 显著为负
    short_term_heads = [(l, h, shift[l,h], global_early[l,h], local_early[l,h])
                        for l in range(num_layers) for h in range(num_heads)
                        if mask[l,h] and shift[l,h] < -0.02]
    short_term_heads.sort(key=lambda x: x[2])

    out_path = os.path.join(save_dir, "head_scores.npz")
    np.savez(out_path,
             shift=shift, local_early=local_early, global_early=global_early,
             count=count, num_layers=num_layers, num_heads=num_heads)

    print(f"\nSaved: {out_path}")
    print(f"\nLong-term heads ({len(long_term_heads)}):")
    for l, h, s, ge, le in long_term_heads[:10]:
        print(f"  L{l:2d} H{h:2d}: shift={s:+.4f} global_early={ge:.3f} local_early={le:.3f}")
    print(f"\nShort-term heads ({len(short_term_heads)}):")
    for l, h, s, ge, le in short_term_heads[:10]:
        print(f"  L{l:2d} H{h:2d}: shift={s:+.4f} global_early={ge:.3f} local_early={le:.3f}")

    return shift, global_early, local_early


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="results/head_analysis/pseudo-qwen2.5_vl_7b-kv6000-hermes/raw_obs.csv")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_heads", type=int, default=28)
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(args.data)
    process(args.data, save_dir, args.num_layers, args.num_heads)
