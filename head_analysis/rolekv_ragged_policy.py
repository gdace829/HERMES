"""RoleKV-v2: role-aware ragged retention for streaming visual KV cache.

This module is the method-side counterpart of the context-access ablation:

  * query heads are profiled offline as memory-oriented/current-sensitive/mixed;
  * query-head roles are aggregated to Qwen's physical KV heads;
  * during compression, each layer-KV-head keeps its own visual token indices;
  * memory-oriented KV heads can be forced to reserve a quota for previous
    retained memory, while current-sensitive heads reserve a quota for the
    latest appended chunk.

The implementation is deliberately isolated from the base ragged cache code.
Installing the policy only sets ``model._rolekv_ragged_config``; the actual
selection is called from ``hermes_kv_head_ragged_prefill`` when that config is
present.
"""

from collections import Counter, defaultdict
import csv
import math
import random

import torch

from head_analysis.context_denial import load_head_classes


MEMORY_ROLE = "memory"
CURRENT_ROLE = "current"
MIXED_ROLE = "mixed"


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


def _normalize_heads(heads, num_layers, num_heads):
    out = []
    seen = set()
    for layer, head in heads:
        layer = int(layer)
        head = int(head)
        key = (layer, head)
        if 0 <= layer < num_layers and 0 <= head < num_heads and key not in seen:
            out.append([layer, head])
            seen.add(key)
    return out


def _layer_matched_random(reference_heads, num_layers, num_heads, seed, exclude_heads=None):
    rng = random.Random(int(seed))
    exclude = set((int(layer), int(head)) for layer, head in (exclude_heads or []))
    counts = defaultdict(int)
    for layer, _ in reference_heads:
        counts[int(layer)] += 1

    sampled = []
    for layer in range(int(num_layers)):
        count = int(counts.get(layer, 0))
        if count <= 0:
            continue
        candidates = [head for head in range(int(num_heads)) if (layer, head) not in exclude]
        if len(candidates) < count:
            candidates = list(range(int(num_heads)))
        sampled.extend([[layer, head] for head in rng.sample(candidates, count)])
    return sampled


def _build_query_role_heads(head_classes, mode, num_layers, num_heads, seed):
    mode = str(mode or "rolekv_quota").lower()
    memory_heads = _normalize_heads(
        head_classes.get("memory_oriented", []),
        num_layers,
        num_heads,
    )
    current_heads = _normalize_heads(
        head_classes.get("current_sensitive", []),
        num_layers,
        num_heads,
    )

    if mode in ("baseline", "uniform", "none", "norole"):
        return [], []
    if mode in ("random", "random_quota"):
        random_memory = _layer_matched_random(memory_heads, num_layers, num_heads, seed)
        random_current = _layer_matched_random(
            current_heads,
            num_layers,
            num_heads,
            seed + 1009,
            exclude_heads=random_memory,
        )
        return random_memory, random_current
    if mode in ("inverted", "inverted_quota"):
        return current_heads, memory_heads
    if mode in ("rolekv", "rolekv_quota"):
        return memory_heads, current_heads
    raise ValueError(f"Unknown RoleKV ragged mode: {mode}")


def build_kv_role_table(
    head_classes,
    mode,
    num_layers,
    num_query_heads,
    num_kv_heads,
    seed=0,
    role_min_votes=1,
):
    """Aggregate query-head roles to physical KV-head roles.

    Qwen2.5-VL-7B has 28 query heads and 4 KV heads, so each KV head is shared
    by a group of 7 query heads. Since top/bottom quantile roles are sparse,
    the role decision is made by majority among selected query heads inside the
    group, not by majority among all 7 query heads.
    """
    num_layers = int(num_layers)
    num_query_heads = int(num_query_heads)
    num_kv_heads = int(num_kv_heads)
    group_size = num_query_heads // num_kv_heads
    role_min_votes = max(1, int(role_min_votes))

    memory_heads, current_heads = _build_query_role_heads(
        head_classes,
        mode=mode,
        num_layers=num_layers,
        num_heads=num_query_heads,
        seed=seed,
    )
    memory_set = set((int(layer), int(head)) for layer, head in memory_heads)
    current_set = set((int(layer), int(head)) for layer, head in current_heads)

    role_table = []
    group_counts = []
    role_counter = Counter()
    for layer_idx in range(num_layers):
        layer_roles = []
        layer_counts = []
        for kv_head in range(num_kv_heads):
            q_start = kv_head * group_size
            q_end = q_start + group_size
            memory_votes = sum((layer_idx, head) in memory_set for head in range(q_start, q_end))
            current_votes = sum((layer_idx, head) in current_set for head in range(q_start, q_end))
            if memory_votes >= role_min_votes and memory_votes > current_votes:
                role = MEMORY_ROLE
            elif current_votes >= role_min_votes and current_votes > memory_votes:
                role = CURRENT_ROLE
            else:
                role = MIXED_ROLE
            layer_roles.append(role)
            layer_counts.append(
                {
                    "memory_votes": int(memory_votes),
                    "current_votes": int(current_votes),
                }
            )
            role_counter[role] += 1
        role_table.append(layer_roles)
        group_counts.append(layer_counts)

    return {
        "role_table": role_table,
        "group_counts": group_counts,
        "query_memory_heads": memory_heads,
        "query_current_heads": current_heads,
        "role_counts": dict(role_counter),
        "num_layers": num_layers,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "group_size": group_size,
        "role_min_votes": role_min_votes,
    }


def _load_kv_profile_scores(kv_profile_path, metric, num_layers, num_kv_heads):
    rows = []
    with open(kv_profile_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer", "kv_head", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"KV profile {kv_profile_path} is missing columns: {sorted(missing)}"
            )
        for row in reader:
            layer = int(row["layer"])
            kv_head = int(row["kv_head"])
            if not (0 <= layer < int(num_layers) and 0 <= kv_head < int(num_kv_heads)):
                continue
            value = float(row[metric])
            rows.append((layer, kv_head, value))

    expected = int(num_layers) * int(num_kv_heads)
    seen = {(layer, kv_head) for layer, kv_head, _ in rows}
    if len(seen) != expected:
        raise ValueError(
            f"KV profile {kv_profile_path} covers {len(seen)} layer-KV-heads, "
            f"expected {expected}"
        )
    return rows


def build_kv_role_table_from_profile_csv(
    kv_profile_path,
    mode,
    num_layers,
    num_query_heads,
    num_kv_heads,
    metric="b_log_per_token_ratio",
    quantile=0.2,
    seed=0,
):
    """Build physical KV-head roles directly from a [layer, kv_head] profile.

    Lower ``metric`` values are treated as more memory-oriented; higher values
    are treated as relatively current-sensitive.
    """
    num_layers = int(num_layers)
    num_query_heads = int(num_query_heads)
    num_kv_heads = int(num_kv_heads)
    group_size = num_query_heads // num_kv_heads
    mode = str(mode or "rolekv_quota").lower()

    rows = _load_kv_profile_scores(
        kv_profile_path,
        metric=metric,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
    )
    rows_sorted = sorted(rows, key=lambda item: item[2])
    total = num_layers * num_kv_heads
    k = max(1, int(math.ceil(total * float(quantile))))
    k = min(k, len(rows_sorted) // 2)

    memory_kv_heads = [[layer, kv_head] for layer, kv_head, _ in rows_sorted[:k]]
    current_kv_heads = [[layer, kv_head] for layer, kv_head, _ in rows_sorted[-k:]]

    if mode in ("baseline", "uniform", "none", "norole"):
        memory_kv_heads = []
        current_kv_heads = []
    elif mode in ("random", "random_quota"):
        memory_kv_heads = _layer_matched_random(
            memory_kv_heads,
            num_layers,
            num_kv_heads,
            seed,
        )
        current_kv_heads = _layer_matched_random(
            current_kv_heads,
            num_layers,
            num_kv_heads,
            seed + 1009,
            exclude_heads=memory_kv_heads,
        )
    elif mode in ("inverted", "inverted_quota"):
        memory_kv_heads, current_kv_heads = current_kv_heads, memory_kv_heads
    elif mode in ("rolekv", "rolekv_quota"):
        pass
    else:
        raise ValueError(f"Unknown RoleKV ragged mode: {mode}")

    memory_set = set((int(layer), int(kv_head)) for layer, kv_head in memory_kv_heads)
    current_set = set((int(layer), int(kv_head)) for layer, kv_head in current_kv_heads)
    score_map = {(int(layer), int(kv_head)): float(value) for layer, kv_head, value in rows}

    role_table = []
    group_counts = []
    role_counter = Counter()
    for layer_idx in range(num_layers):
        layer_roles = []
        layer_counts = []
        for kv_head in range(num_kv_heads):
            key = (layer_idx, kv_head)
            if key in memory_set:
                role = MEMORY_ROLE
            elif key in current_set:
                role = CURRENT_ROLE
            else:
                role = MIXED_ROLE
            layer_roles.append(role)
            layer_counts.append({"profile_score": score_map.get(key)})
            role_counter[role] += 1
        role_table.append(layer_roles)
        group_counts.append(layer_counts)

    return {
        "role_table": role_table,
        "group_counts": group_counts,
        "query_memory_heads": [],
        "query_current_heads": [],
        "kv_memory_heads": memory_kv_heads,
        "kv_current_heads": current_kv_heads,
        "role_counts": dict(role_counter),
        "num_layers": num_layers,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "group_size": group_size,
        "role_min_votes": None,
        "kv_profile_path": kv_profile_path,
        "kv_profile_metric": metric,
        "kv_profile_quantile": float(quantile),
    }


def apply_rolekv_ragged_policy(
    model,
    head_classes_path=None,
    mode="rolekv_quota",
    quota_ratio=0.7,
    lambda_memory=0.2,
    lambda_current=0.2,
    seed=0,
    num_layers=None,
    num_query_heads=None,
    num_kv_heads=None,
    role_min_votes=1,
    kv_profile_path=None,
    kv_profile_metric="b_log_per_token_ratio",
    kv_profile_quantile=0.2,
):
    """Install RoleKV-v2 configuration on a model using ragged prefill.

    The caller should also install ``apply_kv_head_ragged_prefill``. This
    function does not patch model forwards by itself, so it is safe to call
    before or after the ragged prefill hook is installed.
    """
    actual_query_heads, actual_kv_heads, _ = _get_attention_shape(
        model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )
    num_layers = int(num_layers or getattr(model, "num_layers", 28))
    num_query_heads = int(num_query_heads or actual_query_heads)
    num_kv_heads = int(num_kv_heads or actual_kv_heads)

    if kv_profile_path:
        role_info = build_kv_role_table_from_profile_csv(
            kv_profile_path,
            mode=mode,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            metric=kv_profile_metric,
            quantile=kv_profile_quantile,
            seed=seed,
        )
        role_source = "kv_profile"
    else:
        if not head_classes_path:
            raise ValueError("Either head_classes_path or kv_profile_path must be provided")
        head_classes = load_head_classes(head_classes_path)
        num_layers = int(num_layers or head_classes.get("num_layers", getattr(model, "num_layers", 28)))
        num_query_heads = int(num_query_heads or head_classes.get("num_heads", actual_query_heads))
        role_info = build_kv_role_table(
            head_classes,
            mode=mode,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            seed=seed,
            role_min_votes=role_min_votes,
        )
        role_source = "query_head_classes"

    mode = str(mode or "rolekv_quota").lower()
    use_quota = mode in ("rolekv_quota", "random", "random_quota", "inverted", "inverted_quota")
    model._rolekv_ragged_config = {
        "mode": mode,
        "head_classes_path": head_classes_path,
        "kv_profile_path": kv_profile_path,
        "kv_profile_metric": kv_profile_metric,
        "kv_profile_quantile": float(kv_profile_quantile),
        "role_source": role_source,
        "quota_ratio": float(quota_ratio),
        "lambda_memory": float(lambda_memory),
        "lambda_current": float(lambda_current),
        "seed": int(seed),
        "use_quota": bool(use_quota),
        "num_layers": int(num_layers),
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(num_kv_heads),
        "group_size": int(role_info["group_size"]),
        "role_min_votes": int(role_min_votes),
        "role_table": role_info["role_table"],
        "group_counts": role_info["group_counts"],
        "query_memory_heads": role_info["query_memory_heads"],
        "query_current_heads": role_info["query_current_heads"],
        "kv_memory_heads": role_info.get("kv_memory_heads", []),
        "kv_current_heads": role_info.get("kv_current_heads", []),
        "role_counts": role_info["role_counts"],
        "mode_detail": "physical_per_kv_head_ragged_retention",
    }
    print(
        "[RoleKV-ragged] Installed role-aware ragged retention: "
        f"mode={mode}, quota={float(quota_ratio):g}, "
        f"source={role_source}, kv_roles={role_info['role_counts']}, "
        f"query_memory={len(role_info['query_memory_heads'])}, "
        f"query_current={len(role_info['query_current_heads'])}, "
        f"kv_memory={len(role_info.get('kv_memory_heads', []))}, "
        f"kv_current={len(role_info.get('kv_current_heads', []))}"
    )
    return model


def _normalize_01(values):
    if values.numel() == 0:
        return values.float()
    values = values.float()
    vmin = values.min()
    vmax = values.max()
    return (values - vmin) / (vmax - vmin + 1e-6)


def _score_to_length(score, head_len, device):
    score = score.to(device=device).float()
    if score.numel() < head_len:
        pad = torch.zeros(head_len - score.numel(), device=device, dtype=score.dtype)
        score = torch.cat([score, pad], dim=0)
    return score[:head_len]


def _role_base_score(model, layer_idx, role, local_score, global_score, mixed_score, text_keep):
    head_len = int(local_score.numel())
    combined = torch.zeros(head_len, device=local_score.device, dtype=torch.float32)
    if head_len <= text_keep:
        return combined

    if role == MEMORY_ROLE:
        combined[text_keep:] = _normalize_01(global_score[text_keep:])
        return combined
    if role == CURRENT_ROLE:
        combined[text_keep:] = _normalize_01(local_score[text_keep:])
        return combined

    if layer_idx < model.short_term_threshold:
        source = local_score
        recency_alpha = 1.0
        k_decay = 20.0
    elif layer_idx >= model.long_term_threshold:
        source = global_score
        recency_alpha = 0.0
        k_decay = 0.0
    else:
        source = mixed_score
        progress = (
            (layer_idx - model.short_term_threshold)
            / max(model.long_term_threshold - model.short_term_threshold, 1)
        )
        recency_alpha = 0.75 - 0.6 * progress
        k_decay = 20.0 - 12.0 * progress

    visual_score = source[text_keep:]
    num_visual = int(visual_score.numel())
    positions = torch.arange(num_visual, device=source.device, dtype=torch.float32)
    time_distances = (num_visual - 1 - positions) / max(num_visual - 1, 1)
    recency = torch.exp(-float(k_decay) * time_distances)
    recency = _normalize_01(recency)
    attn_norm = _normalize_01(visual_score)
    combined[text_keep:] = attn_norm * (1.0 - float(recency_alpha)) + recency * float(recency_alpha)
    return combined


def _arange_region(start, end, device):
    start = int(start)
    end = int(end)
    if end <= start:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.arange(start, end, device=device, dtype=torch.long)


def _topk_from_candidates(score, candidates, k):
    k = int(k)
    if k <= 0 or candidates.numel() == 0:
        return torch.empty(0, device=score.device, dtype=torch.long)
    k = min(k, int(candidates.numel()))
    local_rank = torch.topk(score.index_select(0, candidates), k, sorted=False).indices
    return candidates.index_select(0, local_rank)


def _select_with_quota(score, visual_positions, target_positions, budget, quota_ratio):
    budget = int(budget)
    if budget <= 0:
        return torch.empty(0, device=score.device, dtype=torch.long)
    budget = min(budget, int(visual_positions.numel()))

    quota = int(round(budget * float(quota_ratio)))
    quota = min(max(quota, 0), budget, int(target_positions.numel()))
    selected = _topk_from_candidates(score, target_positions, quota)

    remaining = budget - int(selected.numel())
    if remaining > 0:
        free_score = score.clone()
        if selected.numel() > 0:
            free_score[selected] = -torch.finfo(free_score.dtype).max
        free = _topk_from_candidates(free_score, visual_positions, remaining)
        selected = torch.cat([selected, free])

    return torch.unique(selected, sorted=True)


def _nested_lens_value(nested_lens, layer_idx, kv_head, fallback):
    if nested_lens is None:
        return int(fallback)
    try:
        layer_lens = nested_lens[int(layer_idx)]
        if isinstance(layer_lens, torch.Tensor):
            return int(layer_lens[int(kv_head)].item())
        return int(layer_lens[int(kv_head)])
    except Exception:
        return int(fallback)


def select_rolekv_ragged_keep_indices(
    model,
    local_scores,
    global_scores,
    mixed_scores,
    num_keep,
    budget_table,
):
    """Select physical keep indices for every layer-KV-head under RoleKV-v2."""
    config = getattr(model, "_rolekv_ragged_config", None)
    if not config:
        raise ValueError("RoleKV ragged config is not installed on the model")

    device = model.device
    visual_start_idx = int(model.visual_start_idx)
    num_layers = len(model.kv_cache.layers)
    num_kv_heads = int(model.kv_cache.layers[0].head_lens.numel())
    role_table = config["role_table"]
    mode = str(config.get("mode", "rolekv_quota")).lower()
    use_quota = bool(config.get("use_quota", False))
    quota_ratio = float(config.get("quota_ratio", 0.7))
    lambda_memory = float(config.get("lambda_memory", 0.2))
    lambda_current = float(config.get("lambda_current", 0.2))

    pre_lens = getattr(model, "_rolekv_last_pre_lens_ragged", None)
    post_lens = getattr(model, "_rolekv_last_post_lens_ragged", None)

    keep_indices = []
    visible_lengths = []
    selected_counter = Counter()
    quota_counter = Counter()
    budget_values = []

    for layer_idx in range(num_layers):
        layer_keep = []
        for kv_head in range(num_kv_heads):
            head_len = int(model.kv_cache.layers[layer_idx].head_lens[kv_head].item())
            text_keep = min(visual_start_idx, head_len)
            num_visual_tokens = max(0, head_len - text_keep)
            if num_visual_tokens <= 0:
                keep = torch.arange(head_len, device=device, dtype=torch.long)
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                continue

            local_score = _score_to_length(local_scores[layer_idx][kv_head], head_len, device)
            global_score = _score_to_length(global_scores[layer_idx][kv_head], head_len, device)
            mixed_score = _score_to_length(mixed_scores[layer_idx][kv_head], head_len, device)

            role = role_table[layer_idx][kv_head]
            if mode in ("baseline", "uniform", "none", "norole"):
                role = MIXED_ROLE

            combined = _role_base_score(
                model,
                layer_idx,
                role,
                local_score,
                global_score,
                mixed_score,
                text_keep,
            )

            pre_len = _nested_lens_value(pre_lens, layer_idx, kv_head, head_len)
            post_len = _nested_lens_value(post_lens, layer_idx, kv_head, head_len)
            pre_len = min(max(pre_len, text_keep), head_len)
            post_len = min(max(post_len, pre_len), head_len)
            previous_positions = _arange_region(text_keep, pre_len, device)
            current_positions = _arange_region(pre_len, post_len, device)
            visual_positions = _arange_region(text_keep, head_len, device)

            if role == MEMORY_ROLE and previous_positions.numel() > 0:
                combined[previous_positions] += lambda_memory
            elif role == CURRENT_ROLE and current_positions.numel() > 0:
                combined[current_positions] += lambda_current

            budget = min(int(budget_table[layer_idx, kv_head]), num_visual_tokens)
            budget = max(0, budget)
            budget_values.append(int(budget))

            if budget <= 0:
                selected_visual = torch.empty(0, device=device, dtype=torch.long)
            elif use_quota and role == MEMORY_ROLE and previous_positions.numel() > 0:
                selected_visual = _select_with_quota(
                    combined,
                    visual_positions,
                    previous_positions,
                    budget,
                    quota_ratio,
                )
                quota_counter["memory_quota_heads"] += 1
            elif use_quota and role == CURRENT_ROLE and current_positions.numel() > 0:
                selected_visual = _select_with_quota(
                    combined,
                    visual_positions,
                    current_positions,
                    budget,
                    quota_ratio,
                )
                quota_counter["current_quota_heads"] += 1
            else:
                selected_visual = _topk_from_candidates(combined, visual_positions, budget)

            selected_counter[role] += 1
            keep = torch.cat([
                torch.arange(text_keep, device=device, dtype=torch.long),
                torch.sort(selected_visual)[0],
            ])
            keep = torch.unique(keep, sorted=True)
            layer_keep.append(keep)
            visible_lengths.append(int(keep.numel()))
        keep_indices.append(layer_keep)

    model._rolekv_ragged_last_stats = {
        "mode": mode,
        "use_quota": bool(use_quota),
        "quota_ratio": quota_ratio,
        "lambda_memory": lambda_memory,
        "lambda_current": lambda_current,
        "selected_kv_roles": dict(selected_counter),
        "quota_stats": dict(quota_counter),
        "budget_min": min(budget_values) if budget_values else 0,
        "budget_max": max(budget_values) if budget_values else 0,
        "visible_min": min(visible_lengths) if visible_lengths else 0,
        "visible_max": max(visible_lengths) if visible_lengths else 0,
        "num_keep": int(num_keep),
    }
    return keep_indices, visible_lengths
