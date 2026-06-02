"""RoleKV-v1: role-aware head voting for streaming visual KV retention.

This is the minimal method prototype:
  * keep the standard dense per-layer KV cache;
  * profile query heads into memory-oriented/current-sensitive/mixed roles;
  * during compression, let each head vote for visual tokens with a
    role-specific previous-memory/latest-chunk bonus;
  * keep the same layer budget as the base streaming compression path.
"""

import json
import random
from collections import defaultdict
from types import MethodType

import torch

from head_analysis.context_denial import load_head_classes


def _heads_by_layer(heads):
    grouped = defaultdict(set)
    for layer, head in heads:
        grouped[int(layer)].add(int(head))
    return grouped


def _layer_matched_random(reference_heads, num_layers, num_heads, seed):
    rng = random.Random(int(seed))
    counts = defaultdict(int)
    for layer, _ in reference_heads:
        counts[int(layer)] += 1
    sampled = []
    for layer in range(num_layers):
        count = counts.get(layer, 0)
        if count > 0:
            sampled.extend([[layer, h] for h in rng.sample(list(range(num_heads)), count)])
    return sampled


def _normalize_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def _visual_scores(attn_weights, visual_start, question_len):
    # attn_weights: [batch, heads, q_len, kv_len + q_len]
    end = -int(question_len) if int(question_len) > 0 else None
    return attn_weights[0].mean(dim=1)[:, visual_start:end]


def apply_rolekv_policy(
    model,
    head_classes_path,
    mode="rolekv",
    lambda_memory=0.2,
    lambda_current=0.2,
    seed=0,
    num_layers=None,
    num_heads=None,
):
    """Install RoleKV pruning by replacing ``prune_kv_cache_by_attention``.

    Modes:
      rolekv   memory heads get previous-memory bonus; current heads get latest bonus
      inverted memory/current bonuses are swapped
      random   layer-matched random heads receive memory/current roles
      baseline no role bonus; uses the same head-voting scaffold
    """
    head_classes = load_head_classes(head_classes_path)
    num_layers = int(num_layers or head_classes.get("num_layers", model.num_layers))
    num_heads = int(num_heads or head_classes.get("num_heads", 28))

    memory_heads = head_classes.get("memory_oriented", [])
    current_heads = head_classes.get("current_sensitive", [])

    if mode == "random":
        memory_heads = _layer_matched_random(memory_heads, num_layers, num_heads, seed)
        current_heads = _layer_matched_random(current_heads, num_layers, num_heads, seed + 1009)
    elif mode == "inverted":
        memory_heads, current_heads = current_heads, memory_heads
    elif mode in ("rolekv", "baseline"):
        pass
    else:
        raise ValueError(f"Unknown RoleKV mode: {mode}")

    memory_by_layer = _heads_by_layer(memory_heads)
    current_by_layer = _heads_by_layer(current_heads)

    if not hasattr(model, "_rolekv_original_encode_video_chunk"):
        model._rolekv_original_encode_video_chunk = model.encode_video_chunk

        def encode_video_chunk_with_tracking(self, video_chunk):
            pre_lens = self._get_cache_seq_len_per_layer()
            result = self._rolekv_original_encode_video_chunk(video_chunk)
            post_lens = self._get_cache_seq_len_per_layer()
            self._rolekv_last_pre_lens = [int(x) for x in pre_lens]
            self._rolekv_last_post_lens = [int(x) for x in post_lens]
            return result

        model.encode_video_chunk = MethodType(encode_video_chunk_with_tracking, model)

    def rolekv_prune(attn_weights_local, attn_weights_global, attn_weights_mixed, num_keep=3000):
        device = model.device
        visual_start = int(model.visual_start_idx)
        n_layers = len(attn_weights_local)
        actual_heads = int(attn_weights_local[0].shape[1])
        if actual_heads != num_heads:
            raise ValueError(f"Configured num_heads={num_heads}, attention has {actual_heads}")

        q_len_local = int(attn_weights_local[0].shape[2])
        q_len_global = int(attn_weights_global[0].shape[2])
        q_len_mixed = int(attn_weights_mixed[0].shape[2])

        layer_budgets = model.allocate_budget_by_depth(int(num_keep) * n_layers, n_layers)
        keep_indices_all_layers = []

        pre_lens = getattr(model, "_rolekv_last_pre_lens", None)
        post_lens = getattr(model, "_rolekv_last_post_lens", None)

        for layer_idx in range(n_layers):
            local_visual = _visual_scores(attn_weights_local[layer_idx], visual_start, q_len_local)
            global_visual = _visual_scores(attn_weights_global[layer_idx], visual_start, q_len_global)
            mixed_visual = _visual_scores(attn_weights_mixed[layer_idx], visual_start, q_len_mixed)

            num_visual = int(local_visual.shape[1])
            if num_visual <= 0:
                keep_indices_all_layers.append(torch.arange(visual_start, device=device).tolist())
                continue

            layer_budget = int(layer_budgets[layer_idx])
            if layer_idx >= model.long_term_threshold:
                layer_budget = max(0, layer_budget - 1)
            layer_budget = min(layer_budget, num_visual)
            head_budget = max(1, layer_budget // max(actual_heads, 1))

            if pre_lens is not None and post_lens is not None:
                pre_rel = max(0, min(int(pre_lens[layer_idx]) - visual_start, num_visual))
                post_rel = max(pre_rel, min(int(post_lens[layer_idx]) - visual_start, num_visual))
            else:
                pre_rel = num_visual
                post_rel = num_visual

            vote = torch.zeros(num_visual, device=device, dtype=torch.float32)
            fallback = torch.zeros(num_visual, device=device, dtype=torch.float32)
            memory_heads_layer = memory_by_layer.get(layer_idx, set())
            current_heads_layer = current_by_layer.get(layer_idx, set())

            for head_idx in range(actual_heads):
                if head_idx in memory_heads_layer:
                    score = _normalize_01(global_visual[head_idx].float())
                    if mode != "baseline" and pre_rel > 0:
                        score[:pre_rel] = score[:pre_rel] + float(lambda_memory)
                elif head_idx in current_heads_layer:
                    score = _normalize_01(local_visual[head_idx].float())
                    if mode != "baseline" and post_rel > pre_rel:
                        score[pre_rel:post_rel] = score[pre_rel:post_rel] + float(lambda_current)
                else:
                    score = _normalize_01(
                        0.5 * local_visual[head_idx].float()
                        + 0.5 * global_visual[head_idx].float()
                        + 0.1 * mixed_visual[head_idx].float()
                    )

                fallback += score
                k = min(head_budget, num_visual)
                top = torch.topk(score, k, sorted=False).indices
                vote[top] += 1.0 / max(k, 1)

            fallback = _normalize_01(fallback / max(actual_heads, 1))
            combined = vote + 1e-3 * fallback
            keep_rel = torch.topk(combined, layer_budget, sorted=False).indices
            keep_abs = torch.sort(keep_rel + visual_start)[0]
            full_keep = torch.cat([torch.arange(visual_start, device=device), keep_abs]).unique(sorted=True)
            keep_indices_all_layers.append(full_keep.tolist())

        model._rolekv_last_stats = {
            "mode": mode,
            "lambda_memory": float(lambda_memory),
            "lambda_current": float(lambda_current),
            "num_memory_heads": int(sum(len(v) for v in memory_by_layer.values())),
            "num_current_heads": int(sum(len(v) for v in current_by_layer.values())),
        }
        return keep_indices_all_layers

    model.prune_kv_cache_by_attention = rolekv_prune
    model._rolekv_config = {
        "mode": str(mode),
        "head_classes_path": head_classes_path,
        "lambda_memory": float(lambda_memory),
        "lambda_current": float(lambda_current),
        "seed": int(seed),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "num_memory_heads": int(sum(len(v) for v in memory_by_layer.values())),
        "num_current_heads": int(sum(len(v) for v in current_by_layer.values())),
    }
    print(
        "[RoleKV] Installed role-aware head voting: "
        f"mode={mode}, memory_heads={model._rolekv_config['num_memory_heads']}, "
        f"current_heads={model._rolekv_config['num_current_heads']}, "
        f"lambda=({float(lambda_memory):g}, {float(lambda_current):g})"
    )
    return model
