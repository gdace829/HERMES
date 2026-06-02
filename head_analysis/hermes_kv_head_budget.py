"""
Per-KV-head HERMES pruning prototype.

This module lets each KV head choose a different visual-token keep set. The
first implementation is intentionally conservative: the physical cache remains
dense and stores the union of all KV-head keep sets, while a per-query-head
attention mask makes each query-head group see only the tokens selected for its
own KV head.

It is a stepping stone toward HybridKV-style flat ragged caches. It validates
head-wise eviction quality without replacing Qwen's full attention/cache stack.
"""

import numpy as np
import torch

from head_analysis.hermes_head_budget import load_head_scores


def _finite_positive_weights(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, 1.0)


def _allocate_integer_budget(weights, total_budget, min_budget=0, max_budget=None):
    weights = _finite_positive_weights(weights)
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
    frac = raw - np.floor(raw)

    diff = total_budget - int(budgets.sum())
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


def _get_language_config(model):
    language_model = getattr(model, "language_model", None)
    config = getattr(language_model, "config", None)
    if config is None and hasattr(language_model, "model"):
        config = getattr(language_model.model, "config", None)
    return config


def _get_attention_shape(model, num_query_heads=None, num_kv_heads=None):
    config = _get_language_config(model)
    if num_query_heads is None:
        num_query_heads = getattr(config, "num_attention_heads", None) if config is not None else None
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "num_key_value_heads", None) if config is not None else None

    num_query_heads = int(num_query_heads or 28)
    num_kv_heads = int(num_kv_heads or 4)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    return num_query_heads, num_kv_heads, num_query_heads // num_kv_heads


def _aggregate_query_scores_to_kv(scores, num_layers, num_query_heads, num_kv_heads):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (num_layers, num_query_heads):
        raise ValueError(
            f"scores shape must be {(num_layers, num_query_heads)}, got {scores.shape}"
        )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    groups = num_query_heads // num_kv_heads
    return np.abs(scores).reshape(num_layers, num_kv_heads, groups).mean(axis=-1)


def build_kv_head_budget_table(scores, num_keep, num_layers=28,
                               num_query_heads=28, num_kv_heads=4,
                               strength=0.75, min_ratio=0.75,
                               max_ratio=1.25,
                               scheme="relative",
                               sparsemm_ratio=0.1,
                               sparsemm_window_size=32):
    """
    Build [layer, kv_head] budgets.

    scheme="relative" keeps the old HERMES prototype behavior: each layer has
    average num_keep and heads are clipped by min_ratio/max_ratio.

    scheme="sparsemm" follows SparseMM's static visual-score formula:
        base = num_keep - window_size
        min_cache = base * ratio
        head_budget = score * ((base - min_cache) * L * H_kv) + min_cache
        final_budget = head_budget + window_size

    During pruning, the window_size portion is treated as protected recent
    cache, while the remaining head_budget is used for historical top-k
    selection.

    SparseMM normalizes visual scores globally over all layers/KV-heads, so the
    average budget is num_keep per layer-KV-head.

    scheme="sparsemm_layer_total" keeps the same SparseMM score ranking and
    per-head minimum/window, but constrains the global total visual budget to
    num_keep * L. For Qwen2.5-VL-7B with 4 KV heads, num_keep=6000 means an
    average of 1500 visual tokens per layer-KV-head, or 6000 per layer in total.

    scheme="sparsemm_per_layer_total" constrains each layer separately: the
    layer's KV-head budgets sum to num_keep exactly, while SparseMM scores still
    decide how the layer budget is split among that layer's KV heads.
    """
    kv_scores = _aggregate_query_scores_to_kv(
        scores, num_layers, num_query_heads, num_kv_heads
    )
    scheme = str(scheme or "relative").lower()
    if scheme == "sparsemm":
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        remain_capacity = (
            max(base_capacity - min_cache, 0)
            * int(num_layers)
            * int(num_kv_heads)
        )
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()
        budgets = np.round(weights * remain_capacity + min_cache).astype(int)
        budgets = budgets + int(sparsemm_window_size)
        return np.maximum(budgets, 1).astype(int)

    if scheme in ("sparsemm_layer_total", "sparsemm_layer", "sparsemm_total"):
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        min_budget = max(1, min_cache + int(sparsemm_window_size))
        target_total = int(num_keep) * int(num_layers)
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        budgets_flat = _allocate_integer_budget(
            weights.reshape(-1),
            total_budget=target_total,
            min_budget=min_budget,
            max_budget=None,
        )
        return budgets_flat.reshape(int(num_layers), int(num_kv_heads)).astype(int)

    if scheme in ("sparsemm_per_layer_total", "sparsemm_layer_exact", "sparsemm_per_layer"):
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        min_budget = max(1, min_cache + int(sparsemm_window_size))
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        budgets = np.zeros((int(num_layers), int(num_kv_heads)), dtype=int)
        for layer_idx in range(int(num_layers)):
            budgets[layer_idx] = _allocate_integer_budget(
                weights[layer_idx],
                total_budget=int(num_keep),
                min_budget=min_budget,
                max_budget=None,
            )
        return budgets.astype(int)

    if scheme != "relative":
        raise ValueError(f"Unsupported KV-head budget scheme: {scheme}")

    if kv_scores.max() < 1e-8:
        weights = np.ones_like(kv_scores)
    else:
        global_mean = kv_scores.mean() + 1e-8
        weights = 1.0 + strength * (kv_scores / global_mean - 1.0)
        weights = np.clip(weights, min_ratio, max_ratio)

    budgets = np.zeros((num_layers, num_kv_heads), dtype=int)
    total_per_layer = int(num_keep) * int(num_kv_heads)
    min_h = max(1, int(round(num_keep * min_ratio)))
    max_h = max(min_h, int(round(num_keep * max_ratio)))
    for layer_idx in range(num_layers):
        budgets[layer_idx] = _allocate_integer_budget(
            weights[layer_idx],
            total_per_layer,
            min_budget=min_h,
            max_budget=max_h,
        )
    return budgets


def _get_layer_past_len(past_key_values, layer_idx):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "layers"):
        if layer_idx >= len(past_key_values.layers):
            return 0
        layer = past_key_values.layers[layer_idx]
        if not getattr(layer, "is_initialized", False):
            return 0
        return layer.keys.shape[-2]
    return past_key_values[layer_idx][0].shape[-2]


def _build_group_mask(batch, q_len, past_len, hidden_states, layer_idx,
                      model, num_query_heads, num_kv_heads, group_size):
    device = hidden_states.device
    mask = torch.zeros(
        (batch, num_query_heads, q_len, past_len + q_len),
        dtype=torch.bool,
        device=device,
    )

    layer_positions = getattr(model, "_kv_head_budget_cache_positions", {}).get(layer_idx)
    union_len = getattr(model, "_kv_head_budget_union_lens", {}).get(layer_idx, past_len)

    if layer_positions is None:
        mask[:, :, :, :past_len] = True
    else:
        for q_head in range(num_query_heads):
            kv_head = q_head // group_size
            allowed = layer_positions[kv_head]
            if not isinstance(allowed, torch.Tensor):
                allowed = torch.as_tensor(allowed, device=device, dtype=torch.long)
            else:
                allowed = allowed.to(device=device, dtype=torch.long)
            allowed = allowed[(allowed >= 0) & (allowed < past_len)]
            if allowed.numel() > 0:
                mask[:, q_head, :, allowed] = True

        # HERMES long-term layers can append summary tokens after the union.
        # Let every head see those extra tail tokens.
        if union_len < past_len:
            mask[:, :, :, union_len:past_len] = True

    if q_len > 0:
        causal_current = torch.tril(
            torch.ones((q_len, q_len), dtype=torch.bool, device=device)
        )
        mask[:, :, :, past_len:past_len + q_len] = causal_current.view(1, 1, q_len, q_len)

    attn_impl = getattr(_get_language_config(model), "_attn_implementation", "sdpa")
    if attn_impl == "eager":
        additive = torch.zeros(mask.shape, dtype=hidden_states.dtype, device=device)
        additive = additive.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
        return additive

    return mask


def _install_kv_head_mask_hooks(model, num_query_heads, num_kv_heads, group_size):
    if not hasattr(model.language_model, "layers"):
        raise ValueError("KV-head budget masks require Qwen-style language_model.layers")

    old_handles = getattr(model, "_kv_head_budget_hook_handles", [])
    for handle in old_handles:
        handle.remove()

    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            batch, q_len = hidden_states.shape[:2]
            past_key_values = kwargs.get("past_key_values", None)
            past_len = _get_layer_past_len(past_key_values, layer_idx)

            max_mask_q_len = getattr(model, "_kv_head_budget_max_mask_q_len", 128)
            if q_len <= max_mask_q_len:
                kwargs["attention_mask"] = _build_group_mask(
                    batch=batch,
                    q_len=q_len,
                    past_len=past_len,
                    hidden_states=hidden_states,
                    layer_idx=layer_idx,
                    model=model,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    group_size=group_size,
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

    model._kv_head_budget_hook_handles = handles


def apply_kv_head_budget(model, kv_head_scores_path=None, scores=None,
                         num_layers=None, num_query_heads=None,
                         num_kv_heads=None, strength=0.75,
                         min_ratio=0.75, max_ratio=1.25,
                         budget_scheme="relative",
                         sparsemm_ratio=0.1,
                         sparsemm_window_size=32,
                         union_cap_ratio=1.0,
                         max_mask_q_len=128):
    """
    Install per-KV-head logical eviction.

    union_cap_ratio caps the dense physical union length relative to num_keep.
    The default 1.0 keeps the physical layer cache close to the original HERMES
    length while still applying different per-head visibility masks.
    """
    num_layers = int(num_layers or model.num_layers)
    num_query_heads, num_kv_heads, group_size = _get_attention_shape(
        model, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads
    )

    if scores is None and kv_head_scores_path is not None:
        scores = load_head_scores(kv_head_scores_path, num_layers, num_query_heads)
    if scores is None:
        scores = np.zeros((num_layers, num_query_heads), dtype=np.float64)
        print("[kv_head_budget] No scores provided, using uniform KV-head budgets")

    scores = np.asarray(scores, dtype=np.float64)
    print(f"[kv_head_budget] Loaded scores. Range: [{scores.min():.4f}, {scores.max():.4f}]")

    _install_kv_head_mask_hooks(model, num_query_heads, num_kv_heads, group_size)
    model._kv_head_budget_cache_positions = {}
    model._kv_head_budget_union_lens = {}
    model._kv_head_budget_max_mask_q_len = int(max_mask_q_len)
    model._kv_head_budget_scores = scores
    budget_scheme_name = str(budget_scheme or "relative").lower()
    protected_recent_window = (
        max(0, int(sparsemm_window_size))
        if budget_scheme_name.startswith("sparsemm")
        else 0
    )

    def kv_head_budget_prune(attn_weights_local, attn_weights_global,
                             attn_weights_mixed, num_keep=3000):
        device = model.device
        visual_start_idx = model.visual_start_idx
        n_layers = len(attn_weights_local)

        if n_layers != num_layers:
            raise ValueError(
                f"KV-head budget configured for {num_layers} layers, got {n_layers}"
            )

        actual_heads = attn_weights_local[0].shape[1]
        if actual_heads != num_query_heads:
            raise ValueError(
                f"Configured num_query_heads={num_query_heads}, attention has {actual_heads}"
            )

        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        budget_table = build_kv_head_budget_table(
            scores,
            num_keep,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            strength=strength,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            scheme=budget_scheme,
            sparsemm_ratio=sparsemm_ratio,
            sparsemm_window_size=sparsemm_window_size,
        )
        model._kv_head_budget_values = budget_table

        keep_indices_all_layers = []
        cache_positions = {}
        union_lens = {}
        union_lengths_for_log = []
        visible_lengths_for_log = []

        for layer_idx in range(n_layers):
            if layer_idx < model.short_term_threshold:
                layer_type = "short-term"
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
                recency_alpha = 1.0
                k_decay = 20.0
            elif layer_idx >= model.long_term_threshold:
                layer_type = "long-term"
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
                recency_alpha = 0.0
                k_decay = 0.0
            else:
                layer_type = "mid-term"
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                progress = (
                    (layer_idx - model.short_term_threshold)
                    / (model.long_term_threshold - model.short_term_threshold)
                )
                recency_alpha = 0.75 - 0.6 * progress
                k_decay = 20.0 - 12.0 * progress

            visual_attn_q = layer_attn_weights[0].mean(dim=1)[:, visual_start_idx:-question_len]
            if visual_attn_q.shape[0] != num_query_heads:
                raise ValueError(
                    f"Layer {layer_idx}: attention heads {visual_attn_q.shape[0]} "
                    f"!= configured {num_query_heads}"
                )

            num_visual_tokens = visual_attn_q.shape[1]
            if num_visual_tokens <= 0:
                keep = torch.arange(visual_start_idx, device=device)
                keep_indices_all_layers.append(keep.tolist())
                all_text = keep.clone()
                cache_positions[layer_idx] = [all_text for _ in range(num_kv_heads)]
                union_lens[layer_idx] = int(keep.numel())
                continue

            visual_attn_kv = visual_attn_q.view(
                num_kv_heads, group_size, num_visual_tokens
            ).mean(dim=1)

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            recency = torch.exp(-k_decay * time_distances)
            recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)

            kv_selected = []
            vote = torch.zeros(num_visual_tokens, device=device)
            score_sum = torch.zeros(num_visual_tokens, device=device)
            effective_budgets = budget_table[layer_idx].copy()
            if layer_type == "long-term":
                effective_budgets = np.maximum(effective_budgets - 1, 0)

            for kv_head in range(num_kv_heads):
                h_attn = visual_attn_kv[kv_head]
                attn_norm = (h_attn - h_attn.min()) / (h_attn.max() - h_attn.min() + 1e-6)
                score = attn_norm * (1.0 - recency_alpha) + recency * recency_alpha
                score_sum += score

                k = min(int(effective_budgets[kv_head]), num_visual_tokens)
                if k <= 0:
                    selected = torch.empty(0, device=device, dtype=torch.long)
                else:
                    recent_k = min(protected_recent_window, k, num_visual_tokens)
                    history_len = max(0, num_visual_tokens - recent_k)
                    history_k = min(max(k - recent_k, 0), history_len)

                    pieces = []
                    if history_k > 0:
                        pieces.append(torch.topk(score[:history_len], history_k, sorted=False)[1])
                    if recent_k > 0:
                        pieces.append(
                            torch.arange(
                                num_visual_tokens - recent_k,
                                num_visual_tokens,
                                device=device,
                                dtype=torch.long,
                            )
                        )
                    selected = torch.unique(torch.cat(pieces), sorted=True)
                    vote[selected] += 1.0
                kv_selected.append(selected)

            combined = vote + 1e-3 * (
                (score_sum - score_sum.min()) / (score_sum.max() - score_sum.min() + 1e-6)
            )

            union_rel = torch.unique(torch.cat(kv_selected), sorted=True)
            union_cap = int(round(float(num_keep) * float(union_cap_ratio)))
            if layer_type == "long-term":
                union_cap = max(0, union_cap - 1)
            union_cap = min(max(union_cap, 0), num_visual_tokens)

            if union_cap > 0 and union_rel.numel() > union_cap:
                recent_k = min(protected_recent_window, union_cap, num_visual_tokens)
                if recent_k > 0:
                    protected_rel = torch.arange(
                        num_visual_tokens - recent_k,
                        num_visual_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                    remaining_cap = union_cap - int(protected_rel.numel())
                    if remaining_cap > 0:
                        all_rel = torch.arange(num_visual_tokens, device=device, dtype=torch.long)
                        candidate_rel = all_rel[~torch.isin(all_rel, protected_rel)]
                        _, capped_pos = torch.topk(combined[candidate_rel], remaining_cap, sorted=False)
                        capped = candidate_rel[capped_pos]
                        union_rel = torch.unique(torch.cat([protected_rel, capped]), sorted=True)
                    else:
                        union_rel = protected_rel
                else:
                    _, capped = torch.topk(combined, union_cap, sorted=False)
                    union_rel = torch.sort(capped)[0]
            elif union_cap == 0:
                union_rel = torch.empty(0, device=device, dtype=torch.long)

            union_abs = union_rel + visual_start_idx
            text_abs = torch.arange(visual_start_idx, device=device)
            full_keep = torch.cat([text_abs, union_abs]).unique(sorted=True)

            union_lens[layer_idx] = int(full_keep.numel())
            keep_indices_all_layers.append(full_keep.tolist())
            union_lengths_for_log.append(int(full_keep.numel()))

            layer_positions = []
            for kv_head, selected_rel in enumerate(kv_selected):
                selected_abs = selected_rel + visual_start_idx
                selected_abs = selected_abs[torch.isin(selected_abs, union_abs)]
                allowed_abs = torch.cat([text_abs, torch.sort(selected_abs)[0]]).unique(sorted=True)

                # full_keep is sorted, so searchsorted maps old absolute indices to
                # positions in the compacted dense-union cache.
                compact_pos = torch.searchsorted(full_keep, allowed_abs)
                compact_pos = compact_pos[compact_pos < full_keep.numel()]
                layer_positions.append(compact_pos.detach())
                visible_lengths_for_log.append(int(compact_pos.numel()))

            cache_positions[layer_idx] = layer_positions

        model._kv_head_budget_cache_positions = cache_positions
        model._kv_head_budget_union_lens = union_lens
        model._kv_head_budget_last_stats = {
            "union_min": min(union_lengths_for_log) if union_lengths_for_log else 0,
            "union_max": max(union_lengths_for_log) if union_lengths_for_log else 0,
            "visible_min": min(visible_lengths_for_log) if visible_lengths_for_log else 0,
            "visible_max": max(visible_lengths_for_log) if visible_lengths_for_log else 0,
            "budget_min": int(budget_table.min()),
            "budget_max": int(budget_table.max()),
        }
        print(
            "[kv_head_budget] prune stats: "
            f"kv_budget=[{budget_table.min()}, {budget_table.max()}], "
            f"visible=[{model._kv_head_budget_last_stats['visible_min']}, "
            f"{model._kv_head_budget_last_stats['visible_max']}], "
            f"union=[{model._kv_head_budget_last_stats['union_min']}, "
            f"{model._kv_head_budget_last_stats['union_max']}]"
        )

        return keep_indices_all_layers

    model.prune_kv_cache_by_attention = kv_head_budget_prune
    model._kv_head_budget_config = {
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "group_size": group_size,
        "strength": strength,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "budget_scheme": str(budget_scheme or "relative").lower(),
        "sparsemm_ratio": float(sparsemm_ratio),
        "sparsemm_window_size": int(sparsemm_window_size),
        "union_cap_ratio": union_cap_ratio,
        "max_mask_q_len": int(max_mask_q_len),
        "mode": "logical_per_kv_head_dense_union",
    }
    print(
        "[kv_head_budget] Installed logical per-KV-head eviction. "
        f"q_heads={num_query_heads}, kv_heads={num_kv_heads}, "
        f"group={group_size}, budget_scheme={model._kv_head_budget_config['budget_scheme']}, "
        f"sparsemm_ratio={float(sparsemm_ratio):g}, sparsemm_window={int(sparsemm_window_size)}, "
        f"union_cap_ratio={union_cap_ratio}, "
        f"max_mask_q_len={int(max_mask_q_len)}."
    )
    return model
