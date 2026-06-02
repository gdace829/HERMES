"""
Head Analysis 可视化脚本。
读取 run_analysis.py 保存的 head_scores.npz。
now: recent_A = 记忆任务上的 recent attention 占比（低 = 更多看早期）
      recent_B = 近期任务上的 recent attention 占比（高 = 更看近期）
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt


def plot_heads(scores_path, save_dir=None):
    data = np.load(scores_path)
    recent_A = data['recent_A']  # [layers, heads]
    recent_B = data['recent_B']  # [layers, heads]

    num_layers, num_heads = recent_A.shape
    print(f"Layers: {num_layers}, Heads: {num_heads}")

    if save_dir is None:
        save_dir = os.path.dirname(scores_path)

    # ---- Three-panel heatmap ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: recent ratio on Probe A (memory tasks) — red = high recent (bad for memory)
    im0 = axes[0].imshow(recent_A, cmap='RdBu_r', aspect='auto', origin='lower',
                         vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title('Recent Ratio on Memory Tasks (CR+CT)\nLow=looks far back, High=looks recent', fontsize=10)
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    plt.colorbar(im0, ax=axes[0])

    # Middle: recent ratio on Probe B (recent tasks)
    im1 = axes[1].imshow(recent_B, cmap='RdBu_r', aspect='auto', origin='lower',
                         vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Recent Ratio on Recent Tasks (CS+PR)\nLow=looks far back, High=looks recent', fontsize=10)
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')
    plt.colorbar(im1, ax=axes[1])

    # Right: Task sensitivity = recent_B - recent_A
    # Positive = more recent on recent tasks (task-adaptive head)
    # Negative = more recent on memory tasks (counter-intuitive)
    # Near zero = fixed behavior regardless of task
    diff = recent_B - recent_A
    vmax = max(abs(diff.max()), abs(diff.min()), 0.01)
    im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', origin='lower',
                         vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[2].set_title('Task Sensitivity (recent_B - recent_A)\nRed=looks recent on recent tasks\nBlue=looks recent on memory tasks', fontsize=10)
    axes[2].set_xlabel('Head')
    axes[2].set_ylabel('Layer')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    save_path = os.path.join(save_dir, "head_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")

    # ---- Layer-level summary ----
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = range(num_layers)
    ax.plot(layers, recent_A.mean(axis=1), 'ro-', label='Recent ratio on Memory tasks (CR+CT)', markersize=4)
    ax.plot(layers, recent_B.mean(axis=1), 'bo-', label='Recent ratio on Recent tasks (CS+PR)', markersize=4)
    ax.axvline(x=int(num_layers*0.1), color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=int(num_layers*0.7), color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Recent Ratio')
    ax.set_title('Layer-wise Recent Attention: Memory vs Recent Tasks\nWindow = last 10 seconds before question')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path2 = os.path.join(save_dir, "layer_summary.png")
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path2}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default="results/head_analysis/head_scores.npz")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    plot_heads(args.scores, args.save_dir)
