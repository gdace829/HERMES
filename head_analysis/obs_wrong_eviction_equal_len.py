"""Safe wrapper for Observation 3 wrong-policy smoke tests.

The existing ``run_head_budget.py`` uses layer-adaptive budgets by default.
Those can create different KV lengths across decoder layers, which is not safe
for Qwen's shared attention mask path. This wrapper keeps every layer at the
same KV length and only changes per-head scoring/voting, making it suitable for
small wrong-policy ablations without modifying the original inference code.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import head_analysis.run_head_budget as run_head_budget
from head_analysis.hermes_head_budget import apply_head_budget as base_apply_head_budget


def equal_length_apply_head_budget(model, head_scores_path=None, scores=None,
                                   num_layers=28, num_heads=28, **kwargs):
    model = base_apply_head_budget(
        model,
        head_scores_path=head_scores_path,
        scores=scores,
        num_layers=num_layers,
        num_heads=num_heads,
        layer_budget_strength=0.0,
        head_budget_strength=1.0,
        layer_min_ratio=1.0,
        layer_max_ratio=1.0,
        head_min_ratio=0.25,
        head_max_ratio=2.0,
        min_head_budget=8,
    )
    original_prune = model.prune_kv_cache_by_attention

    def equal_length_prune(*args, **kwargs):
        keep_lists = original_prune(*args, **kwargs)
        non_long_lengths = [
            len(items)
            for layer_idx, items in enumerate(keep_lists)
            if layer_idx < model.long_term_threshold
        ]
        target = min(non_long_lengths) if non_long_lengths else min(len(items) for items in keep_lists)
        fixed = []
        for layer_idx, items in enumerate(keep_lists):
            # HERMES appends one summary token for long-term layers after pruning.
            # Keep one fewer index there so final cache lengths remain aligned.
            desired = target - 1 if layer_idx >= model.long_term_threshold else target
            fixed.append(items[:max(desired, 1)])
        return fixed

    model.prune_kv_cache_by_attention = equal_length_prune
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--scores", type=str, default=None)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=1)
    args = parser.parse_args()

    run_head_budget.apply_head_budget = equal_length_apply_head_budget
    run_head_budget.run(args)


if __name__ == "__main__":
    main()
