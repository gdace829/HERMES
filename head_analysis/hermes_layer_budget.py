"""
Layer-adaptive HERMES pruning.

Qwen/HF builds one attention mask for the whole decoder forward, so the default
path keeps layer-wise KV cache lengths aligned and uses SparseMM/head scores to
change each layer's recency-vs-attention preference.

Set variable_lengths=True to allow true per-layer KV lengths. That path installs
decoder-layer pre-hooks that replace Qwen's shared causal mask with a mask sized
for each layer's current cache length.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from head_analysis.hermes_head_budget import load_head_scores


def _allocate_integer_budget(weights, total_budget, min_budget=0, max_budget=None):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)

    n_items = int(weights.size)
    total_budget = int(total_budget)
    if n_items == 0:
        return np.zeros(0, dtype=int)
    if total_budget <= 0:
        return np.zeros(n_items, dtype=int)

    min_budget = max(0, int(min_budget))
    if min_budget * n_items > total_budget:
        min_budget = total_budget // n_items

    if max_budget is None:
        max_budget = total_budget
    max_budget = max(int(max_budget), min_budget)

    raw = weights / weights.sum() * total_budget
    raw = np.clip(raw, min_budget, max_budget)
    if raw.sum() > 0:
        raw = raw * (total_budget / raw.sum())

    budgets = np.floor(raw).astype(int)
    budgets = np.clip(budgets, min_budget, max_budget)

    diff = total_budget - int(budgets.sum())
    frac = raw - np.floor(raw)

    while diff > 0:
        candidates = np.where(budgets < max_budget)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(frac[candidates])]
        budgets[idx] += 1
        diff -= 1

    while diff < 0:
        candidates = np.where(budgets > min_budget)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmin(frac[candidates])]
        budgets[idx] -= 1
        diff += 1

    return budgets.astype(int)


def build_layer_budgets(scores, num_keep, num_layers=28, num_heads=28,
                        strength=0.5, min_ratio=0.75, max_ratio=1.25):
    """
    Convert head scores into per-layer token budgets.

    scores can be signed; budget uses score magnitude because both long-memory
    and short-recency specialized heads indicate that the layer is useful.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (num_layers, num_heads):
        raise ValueError(
            f"scores shape must be {(num_layers, num_heads)}, got {scores.shape}"
        )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = max(abs(scores.max()), abs(scores.min()), 1e-6)
    layer_signal = np.abs(scores / vmax).mean(axis=1)

    if layer_signal.max() < 1e-6:
        weights = np.ones(num_layers, dtype=np.float64)
    else:
        global_mean = layer_signal.mean() + 1e-6
        weights = 1.0 + strength * (layer_signal / global_mean - 1.0)
        weights = np.clip(weights, min_ratio, max_ratio)

    total_budget = int(num_keep) * int(num_layers)
    min_budget = max(1, int(round(num_keep * min_ratio)))
    max_budget = max(min_budget, int(round(num_keep * max_ratio)))

    return _allocate_integer_budget(
        weights,
        total_budget,
        min_budget=min_budget,
        max_budget=max_budget,
    )


def build_layer_alpha_adjustments(scores, num_layers=28, num_heads=28,
                                  strength=0.5, min_ratio=0.75,
                                  max_ratio=1.25):
    """
    Convert head scores into per-layer alpha/k adjustment factors.

    Larger score magnitude means the layer has stronger visual-head evidence, so
    we reduce recency bias and rely more on attention. Smaller score magnitude
    receives stronger recency bias. Output is centered around 1.0.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (num_layers, num_heads):
        raise ValueError(
            f"scores shape must be {(num_layers, num_heads)}, got {scores.shape}"
        )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = max(abs(scores.max()), abs(scores.min()), 1e-6)
    layer_signal = np.abs(scores / vmax).mean(axis=1)
    if layer_signal.max() < 1e-6:
        return np.ones(num_layers, dtype=np.float64)

    layer_strength = layer_signal / (layer_signal.mean() + 1e-6)
    alpha_adjust = 1.0 - strength * (layer_strength - 1.0)
    return np.clip(alpha_adjust, min_ratio, max_ratio)


def _get_layer_past_len(past_key_values, layer_idx):
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, DynamicCache):
        if layer_idx >= len(past_key_values.layers):
            return 0
        layer = past_key_values.layers[layer_idx]
        if not getattr(layer, "is_initialized", False):
            return 0
        return layer.keys.shape[-2]
    return past_key_values[layer_idx][0].shape[-2]


def _build_layer_causal_mask(batch, q_len, past_len, dtype, device):
    kv_len = past_len + q_len
    mask = torch.zeros((batch, 1, q_len, kv_len), dtype=dtype, device=device)
    if q_len <= 1:
        return mask

    min_value = torch.finfo(dtype).min
    future = torch.triu(
        torch.ones((q_len, q_len), dtype=torch.bool, device=device),
        diagonal=1,
    )
    current_mask = torch.zeros((q_len, q_len), dtype=dtype, device=device)
    current_mask = current_mask.masked_fill(future, min_value)
    mask[:, :, :, past_len:] = current_mask.view(1, 1, q_len, q_len)
    return mask


def _install_variable_layer_hooks(model):
    """
    Override per-layer attention masks and position embeddings.

    Qwen's text model constructs one shared mask before the decoder loop. With
    variable layer cache lengths that mask can be too short/long for a given
    layer. This hook replaces it inside each decoder layer.
    """
    if not hasattr(model.language_model, "layers"):
        raise ValueError("Variable layer budgets currently require Qwen-style language_model.layers")

    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            batch, q_len = hidden_states.shape[:2]
            past_key_values = kwargs.get("past_key_values", None)
            past_len = _get_layer_past_len(past_key_values, layer_idx)

            kwargs["attention_mask"] = _build_layer_causal_mask(
                batch,
                q_len,
                past_len,
                hidden_states.dtype,
                hidden_states.device,
            )

            if layer_idx in getattr(model, "_layer_position_ids", {}):
                position_ids = model._layer_position_ids[layer_idx]
                kwargs["position_ids"] = position_ids
                kwargs["position_embeddings"] = model.language_model.rotary_emb(
                    hidden_states,
                    position_ids,
                )
            return args, kwargs

        return hook

    for layer_idx, layer in enumerate(model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

    model._variable_layer_budget_hook_handles = handles


def apply_layer_budget(model, layer_scores_path=None, scores=None,
                       num_layers=None, num_heads=None,
                       strength=0.5, min_ratio=0.75, max_ratio=1.25,
                       variable_lengths=False):
    """
    Patch model.prune_kv_cache_by_attention() with layer-adaptive scoring.

    The resulting keep_indices have the same target length for every layer, so
    Qwen's shared attention mask remains valid.
    """
    num_layers = num_layers or model.num_layers

    if num_heads is None:
        language_model = getattr(model, "language_model", None)
        language_config = getattr(language_model, "config", None)
        num_heads = getattr(language_config, "num_attention_heads", None)
        if num_heads is None and hasattr(language_model, "model"):
            num_heads = getattr(language_model.model.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = 28

    if scores is None and layer_scores_path is not None:
        scores = load_head_scores(layer_scores_path, num_layers, num_heads)

    if scores is None:
        scores = np.zeros((num_layers, num_heads), dtype=np.float64)
        print("[layer_budget] No scores provided, using uniform budgets")

    scores = np.asarray(scores, dtype=np.float64)
    print(f"[layer_budget] Loaded scores. Range: [{scores.min():.4f}, {scores.max():.4f}]")

    if variable_lengths:
        def allocate_budget_by_depth(total_budget, requested_layers):
            if requested_layers != num_layers:
                raise ValueError(
                    f"Layer budget configured for {num_layers} layers, "
                    f"but requested {requested_layers}"
                )
            num_keep = int(total_budget) // int(requested_layers)
            budgets = build_layer_budgets(
                scores,
                num_keep,
                num_layers=num_layers,
                num_heads=num_heads,
                strength=strength,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
            model._layer_budget_values = budgets
            return budgets.tolist()

        model.allocate_budget_by_depth = allocate_budget_by_depth
        _install_variable_layer_hooks(model)
        preview = build_layer_budgets(
            scores,
            getattr(model, "kv_size", 6000),
            num_layers=num_layers,
            num_heads=num_heads,
            strength=strength,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )
        model._layer_budget_values = preview
        model._layer_budget_config = {
            "strength": strength,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "mode": "variable_lengths_with_per_layer_masks",
        }
        print(
            "[layer_budget] Installed VARIABLE layer budgets. "
            f"budget=[{preview.min()}, {preview.max()}], "
            "using per-layer attention masks."
        )
        return model

    alpha_adjust = build_layer_alpha_adjustments(
        scores,
        num_layers=num_layers,
        num_heads=num_heads,
        strength=strength,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )

    def layer_adaptive_prune(attn_weights_local, attn_weights_global,
                             attn_weights_mixed, num_keep=3000):
        device = model.device
        visual_start_idx = model.visual_start_idx
        n_layers = len(attn_weights_local)

        if n_layers != num_layers:
            raise ValueError(
                f"Layer adapter configured for {num_layers} layers, "
                f"but attention has {n_layers} layers"
            )

        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        layer_raw_scores = []
        layer_configs = []

        for layer_idx in range(n_layers):
            if layer_idx < model.short_term_threshold:
                layer_type = "short-term"
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
                layer_recency_alpha = 1.0
                k_decay = 20.0
            elif layer_idx >= model.long_term_threshold:
                layer_type = "long-term"
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
                layer_recency_alpha = 0.0
                k_decay = 0.0
            else:
                layer_type = "mid-term"
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                progress = (
                    (layer_idx - model.short_term_threshold)
                    / (model.long_term_threshold - model.short_term_threshold)
                )
                layer_recency_alpha = 0.75 - 0.6 * progress
                k_decay = 20.0 - 12.0 * progress

            adj = float(alpha_adjust[layer_idx])
            layer_recency_alpha = max(0.0, min(1.0, layer_recency_alpha * adj))
            k_decay = max(0.0, k_decay * adj)

            visual_attn_weights = (
                layer_attn_weights[0]
                .mean(dim=0)[:, visual_start_idx:-1 * question_len]
                .mean(dim=0)
            )
            num_visual_tokens = visual_attn_weights.shape[0]
            if num_visual_tokens <= 0:
                layer_raw_scores.append(torch.empty(0, device=device))
                layer_configs.append({
                    "budget": 0,
                    "layer_type": layer_type,
                    "visual_start_idx": visual_start_idx,
                })
                continue

            layer_budget = int(num_keep)
            if layer_type == "long-term":
                # _shrink_positions_and_rerotate_keys adds one summary token for
                # long-term layers. Subtract here to keep final layer lengths
                # aligned with short/mid-term layers.
                layer_budget = max(0, layer_budget - 1)
            layer_budget = min(layer_budget, num_visual_tokens)

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            recency_weights = torch.exp(-k_decay * time_distances)

            attn_norm = (visual_attn_weights - visual_attn_weights.min()) / (
                visual_attn_weights.max() - visual_attn_weights.min() + 1e-6
            )
            recency_norm = (recency_weights - recency_weights.min()) / (
                recency_weights.max() - recency_weights.min() + 1e-6
            )
            raw_score = attn_norm * (1 - layer_recency_alpha) + recency_norm * layer_recency_alpha

            layer_raw_scores.append(raw_score)
            layer_configs.append({
                "budget": layer_budget,
                "layer_type": layer_type,
                "visual_start_idx": visual_start_idx,
            })

        refined_scores = [s.clone() for s in layer_raw_scores]
        for i in range(len(refined_scores) - 2, -1, -1):
            current_type = layer_configs[i]["layer_type"]
            if current_type == "long-term":
                gamma = 0.4
            elif current_type == "mid-term":
                gamma = 0.3
            else:
                gamma = 0.1

            score_current = refined_scores[i]
            score_next = refined_scores[i + 1]
            if score_current.numel() == 0 or score_next.numel() == 0:
                continue

            if score_current.shape[0] != score_next.shape[0]:
                score_next_reshaped = score_next.view(1, 1, -1)
                score_next_interp = F.interpolate(
                    score_next_reshaped,
                    size=score_current.shape[0],
                    mode="linear",
                    align_corners=False,
                ).view(-1)
                refined_scores[i] = (1 - gamma) * score_current + gamma * score_next_interp
            else:
                refined_scores[i] = (1 - gamma) * score_current + gamma * score_next

        keep_indices_all_layers = []
        for layer_idx, score in enumerate(refined_scores):
            start_idx = layer_configs[layer_idx]["visual_start_idx"]
            actual_num_keep = layer_configs[layer_idx]["budget"]

            if score.numel() == 0 or actual_num_keep <= 0:
                keep_indices = torch.arange(start_idx, device=device)
            else:
                topk_indices_relative = torch.topk(score, actual_num_keep, sorted=False)[1]
                topk_indices_absolute = topk_indices_relative + start_idx
                topk_indices_absolute_sorted = torch.sort(topk_indices_absolute)[0]
                keep_indices = torch.cat([
                    torch.arange(start_idx, device=device),
                    topk_indices_absolute_sorted,
                ])

            keep_indices_all_layers.append(keep_indices.tolist())

        return keep_indices_all_layers

    model.prune_kv_cache_by_attention = layer_adaptive_prune
    model._layer_alpha_adjust = alpha_adjust
    model._layer_budget_values = np.full(num_layers, model.kv_size, dtype=int)
    model._layer_budget_config = {
        "strength": strength,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "mode": "safe_equal_length_layer_adaptive",
    }
    print(
        "[layer_budget] Installed safe equal-length layer-adaptive pruning. "
        f"alpha_adjust=[{alpha_adjust.min():.2f}, {alpha_adjust.max():.2f}]"
    )
    return model
