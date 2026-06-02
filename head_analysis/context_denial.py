"""Head-wise context-access ablation for Qwen2.5-VL HERMES.

This module implements the Forcing-KV-style functional probe for streaming
video QA:

  * profile heads offline into memory-oriented/current-sensitive groups;
  * during QA only, deny selected query heads access to previous visual memory
    and/or the latest encoded visual chunk;
  * keep text prompt tokens, answer prompt tokens, generated tokens, and HERMES
    long-term summary tokens visible.

The implementation uses Qwen's existing per-query-head attention-mask path.
It does not physically edit the KV cache, so it is suitable as a controlled
ablation rather than a speedup method.
"""

import csv
import json
import math
import random
from collections import defaultdict
from types import MethodType

try:
    import torch
except ImportError:  # Allows offline head-class construction without torch.
    torch = None


EPS = 1e-12


def _require_torch():
    if torch is None:
        raise ImportError("context-denial hooks require torch in the active environment")


def _get_language_config(model):
    language_model = getattr(model, "language_model", None)
    config = getattr(language_model, "config", None)
    if config is None and hasattr(language_model, "model"):
        config = getattr(language_model.model, "config", None)
    return config


def _get_attention_shape(model, num_query_heads=None):
    config = _get_language_config(model)
    if num_query_heads is None:
        num_query_heads = getattr(config, "num_attention_heads", None) if config is not None else None
    return int(num_query_heads or 28)


def _get_attention_shape_full(model, num_query_heads=None, num_kv_heads=None):
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


def _normalize_heads(heads, num_layers, num_heads):
    out = []
    seen = set()
    for item in heads:
        layer, head = int(item[0]), int(item[1])
        if 0 <= layer < num_layers and 0 <= head < num_heads and (layer, head) not in seen:
            out.append([layer, head])
            seen.add((layer, head))
    return out


def build_head_classes_from_profile_csv(
    csv_path,
    metric="b_log_per_token_ratio",
    quantile=0.2,
    num_layers=28,
    num_heads=28,
):
    """Build bottom/top-quantile head classes from head_profile_scores.csv.

    Lower b_h means the head assigns more per-token attention density to the
    previous retained memory. Higher b_h means it is relatively more sensitive
    to the latest chunk.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric not in row:
                raise ValueError(f"Metric column not found in {csv_path}: {metric}")
            layer = int(row["layer"])
            head = int(row["head"])
            if row[metric] in (None, ""):
                continue
            value = float(row[metric])
            if math.isfinite(value):
                rows.append((layer, head, value))

    if not rows:
        raise ValueError(f"No finite head scores found in {csv_path}")

    rows.sort(key=lambda item: item[2])
    total_heads = int(num_layers) * int(num_heads)
    k = max(1, int(round(total_heads * float(quantile))))
    k = min(k, len(rows) // 2)
    memory = [[layer, head] for layer, head, _ in rows[:k]]
    current = [[layer, head] for layer, head, _ in rows[-k:]]
    selected = set((layer, head) for layer, head in memory + current)
    mixed = [
        [layer, head]
        for layer in range(int(num_layers))
        for head in range(int(num_heads))
        if (layer, head) not in selected
    ]

    return {
        "source_csv": csv_path,
        "metric": metric,
        "quantile": float(quantile),
        "granularity": "query",
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "memory_oriented": _normalize_heads(memory, num_layers, num_heads),
        "current_sensitive": _normalize_heads(current, num_layers, num_heads),
        "mixed": _normalize_heads(mixed, num_layers, num_heads),
    }


def _aggregate_values(values, aggregation):
    aggregation = str(aggregation or "mean").lower()
    if not values:
        return None
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "median":
        values = sorted(values)
        mid = len(values) // 2
        if len(values) % 2 == 1:
            return values[mid]
        return 0.5 * (values[mid - 1] + values[mid])
    if aggregation == "min":
        return min(values)
    if aggregation == "max":
        return max(values)
    raise ValueError(f"Unsupported KV-head aggregation: {aggregation}")


def build_kv_head_classes_from_profile_csv(
    csv_path,
    metric="b_log_per_token_ratio",
    quantile=0.2,
    num_layers=28,
    num_query_heads=28,
    num_kv_heads=4,
    aggregation="mean",
    score_mode="aggregate",
):
    """Build KV-head classes by aggregating query-head profile scores.

    Qwen2.5-VL uses grouped-query attention: multiple query heads share one KV
    head. For KV-cache functional probes, we first aggregate the query-head
    preference score inside each KV group, then classify KV heads by quantile.

    ``score_mode="aggregate"`` consumes head_profile_scores.csv and aggregates
    the per-query-head b_h values inside each KV group. ``score_mode="pooled"``
    consumes raw_prev_current_attention.csv and first pools attention masses
    across all query heads sharing the same KV head for each observation, then
    computes the per-token current/previous density ratio.
    """
    num_layers = int(num_layers)
    num_query_heads = int(num_query_heads)
    num_kv_heads = int(num_kv_heads)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    group_size = num_query_heads // num_kv_heads

    score_mode = str(score_mode or "aggregate").lower()
    if score_mode not in ("aggregate", "pooled"):
        raise ValueError(f"Unsupported KV-head score mode: {score_mode}")

    grouped = defaultdict(list)
    if score_mode == "pooled":
        required = {
            "layer",
            "head",
            "prev_visual_tokens",
            "current_chunk_tokens",
            "local_prev_mass",
            "local_current_mass",
            "global_prev_mass",
            "global_current_mass",
        }
        obs_fields = [
            "video_idx",
            "chunk_idx",
            "question_idx",
            "frame_start",
            "frame_end",
            "task",
        ]
        pooled = defaultdict(lambda: {
            "local_prev_mass": 0.0,
            "local_current_mass": 0.0,
            "global_prev_mass": 0.0,
            "global_current_mass": 0.0,
            "prev_visual_tokens": 0.0,
            "current_chunk_tokens": 0.0,
        })
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "Pooled KV-head scoring requires raw per-observation CSV; "
                    f"missing columns: {sorted(missing)}"
                )
            for row in reader:
                layer = int(row["layer"])
                q_head = int(row["head"])
                if not (0 <= layer < num_layers and 0 <= q_head < num_query_heads):
                    continue
                obs_key = tuple(row.get(field, "") for field in obs_fields)
                key = (layer, q_head // group_size, obs_key)
                try:
                    local_prev_mass = float(row["local_prev_mass"])
                    local_current_mass = float(row["local_current_mass"])
                    global_prev_mass = float(row["global_prev_mass"])
                    global_current_mass = float(row["global_current_mass"])
                    prev_visual_tokens = float(row["prev_visual_tokens"])
                    current_chunk_tokens = float(row["current_chunk_tokens"])
                except (TypeError, ValueError):
                    continue
                if not all(
                    math.isfinite(value)
                    for value in (
                        local_prev_mass,
                        local_current_mass,
                        global_prev_mass,
                        global_current_mass,
                        prev_visual_tokens,
                        current_chunk_tokens,
                    )
                ):
                    continue
                item = pooled[key]
                item["local_prev_mass"] += local_prev_mass
                item["local_current_mass"] += local_current_mass
                item["global_prev_mass"] += global_prev_mass
                item["global_current_mass"] += global_current_mass
                item["prev_visual_tokens"] = max(
                    item["prev_visual_tokens"],
                    prev_visual_tokens,
                )
                item["current_chunk_tokens"] = max(
                    item["current_chunk_tokens"],
                    current_chunk_tokens,
                )

        for (layer, kv_head, _), item in pooled.items():
            n_prev = max(item["prev_visual_tokens"], 1.0)
            n_curr = max(item["current_chunk_tokens"], 1.0)
            local_prev_density = item["local_prev_mass"] / n_prev
            local_curr_density = item["local_current_mass"] / n_curr
            global_prev_density = item["global_prev_mass"] / n_prev
            global_curr_density = item["global_current_mass"] / n_curr
            local_log = math.log((local_curr_density + EPS) / (local_prev_density + EPS))
            global_log = math.log((global_curr_density + EPS) / (global_prev_density + EPS))
            grouped[(layer, kv_head)].append(0.5 * (local_log + global_log))
    else:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if metric not in row:
                    raise ValueError(f"Metric column not found in {csv_path}: {metric}")
                layer = int(row["layer"])
                q_head = int(row["head"])
                if not (0 <= layer < num_layers and 0 <= q_head < num_query_heads):
                    continue
                if row[metric] in (None, ""):
                    continue
                value = float(row[metric])
                if math.isfinite(value):
                    grouped[(layer, q_head // group_size)].append(value)

    rows = []
    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            value = _aggregate_values(grouped.get((layer, kv_head), []), aggregation)
            if value is not None and math.isfinite(value):
                rows.append((layer, kv_head, float(value)))

    if not rows:
        raise ValueError(f"No finite KV-head scores found in {csv_path}")

    rows.sort(key=lambda item: item[2])
    total_heads = num_layers * num_kv_heads
    k = max(1, int(round(total_heads * float(quantile))))
    k = min(k, len(rows) // 2)
    memory = [[layer, kv_head] for layer, kv_head, _ in rows[:k]]
    current = [[layer, kv_head] for layer, kv_head, _ in rows[-k:]]
    selected = set((layer, head) for layer, head in memory + current)
    mixed = [
        [layer, kv_head]
        for layer in range(num_layers)
        for kv_head in range(num_kv_heads)
        if (layer, kv_head) not in selected
    ]

    return {
        "source_csv": csv_path,
        "metric": metric,
        "quantile": float(quantile),
        "granularity": "kv",
        "aggregation": str(aggregation or "mean").lower(),
        "score_mode": score_mode,
        "num_layers": num_layers,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "group_size": group_size,
        "memory_kv_heads": _normalize_heads(memory, num_layers, num_kv_heads),
        "current_kv_heads": _normalize_heads(current, num_layers, num_kv_heads),
        "mixed_kv_heads": _normalize_heads(mixed, num_layers, num_kv_heads),
    }


def save_head_classes(head_classes, path):
    with open(path, "w") as f:
        json.dump(head_classes, f, indent=2)


def load_head_classes(path):
    with open(path) as f:
        data = json.load(f)
    num_layers = int(data.get("num_layers", 28))
    granularity = str(data.get("granularity", "query")).lower()
    num_heads = int(data.get("num_heads", data.get("num_query_heads", 28)))
    data["memory_oriented"] = _normalize_heads(
        data.get("memory_oriented", data.get("memory_bottom", [])),
        num_layers,
        num_heads,
    )
    data["current_sensitive"] = _normalize_heads(
        data.get("current_sensitive", data.get("current_top", [])),
        num_layers,
        num_heads,
    )
    data["mixed"] = _normalize_heads(data.get("mixed", []), num_layers, num_heads)
    data["num_layers"] = num_layers
    data["num_heads"] = num_heads
    data["granularity"] = granularity

    if granularity == "kv" or "memory_kv_heads" in data or "current_kv_heads" in data:
        num_query_heads = int(data.get("num_query_heads", num_heads))
        num_kv_heads = int(data.get("num_kv_heads", 4))
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads={num_query_heads} must be divisible by "
                f"num_kv_heads={num_kv_heads}"
            )
        data["granularity"] = "kv"
        data["num_query_heads"] = num_query_heads
        data["num_kv_heads"] = num_kv_heads
        data["group_size"] = int(data.get("group_size", num_query_heads // num_kv_heads))
        data["memory_kv_heads"] = _normalize_heads(
            data.get("memory_kv_heads", data.get("memory_oriented", [])),
            num_layers,
            num_kv_heads,
        )
        data["current_kv_heads"] = _normalize_heads(
            data.get("current_kv_heads", data.get("current_sensitive", [])),
            num_layers,
            num_kv_heads,
        )
        data["mixed_kv_heads"] = _normalize_heads(
            data.get("mixed_kv_heads", []),
            num_layers,
            num_kv_heads,
        )
    return data


def _heads_by_layer(heads):
    grouped = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    return {layer: sorted(set(values)) for layer, values in grouped.items()}


def _layer_matched_random(reference_heads, num_layers, num_heads, seed, exclude_heads=None):
    rng = random.Random(int(seed))
    exclude = set((int(l), int(h)) for l, h in (exclude_heads or []))
    counts = defaultdict(int)
    for layer, _ in reference_heads:
        counts[int(layer)] += 1

    sampled = []
    for layer in range(num_layers):
        count = counts.get(layer, 0)
        if count <= 0:
            continue
        candidates = [h for h in range(num_heads) if (layer, h) not in exclude]
        if len(candidates) < count:
            candidates = list(range(num_heads))
        sampled.extend([[layer, h] for h in rng.sample(candidates, count)])
    return sampled


def _expand_kv_heads_to_query_heads(kv_heads, group_size, num_query_heads):
    expanded = []
    for layer, kv_head in kv_heads:
        start = int(kv_head) * int(group_size)
        end = min(start + int(group_size), int(num_query_heads))
        expanded.extend([[int(layer), q_head] for q_head in range(start, end)])
    return expanded


def select_denial_heads(
    head_classes,
    setting,
    seed=0,
    head_granularity=None,
    num_query_heads=None,
    num_kv_heads=None,
):
    """Return per-layer query heads for previous/current denial."""
    setting = str(setting)
    head_granularity = str(
        head_granularity or head_classes.get("granularity", "query")
    ).lower()
    num_layers = int(head_classes.get("num_layers", 28))

    if head_granularity == "kv":
        num_query_heads = int(num_query_heads or head_classes.get("num_query_heads", 28))
        num_kv_heads = int(num_kv_heads or head_classes.get("num_kv_heads", 4))
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads={num_query_heads} must be divisible by "
                f"num_kv_heads={num_kv_heads}"
            )
        group_size = int(head_classes.get("group_size", num_query_heads // num_kv_heads))
        memory = _normalize_heads(
            head_classes.get("memory_kv_heads", head_classes.get("memory_oriented", [])),
            num_layers,
            num_kv_heads,
        )
        current = _normalize_heads(
            head_classes.get("current_kv_heads", head_classes.get("current_sensitive", [])),
            num_layers,
            num_kv_heads,
        )
        selector_num_heads = num_kv_heads
    else:
        num_query_heads = int(num_query_heads or head_classes.get("num_heads", 28))
        num_kv_heads = int(num_kv_heads or head_classes.get("num_kv_heads", 0) or 0)
        group_size = None
        memory = _normalize_heads(head_classes.get("memory_oriented", []), num_layers, num_query_heads)
        current = _normalize_heads(head_classes.get("current_sensitive", []), num_layers, num_query_heads)
        selector_num_heads = num_query_heads

    deny_previous = []
    deny_current = []
    selected = []
    deny_previous_source = []
    deny_current_source = []
    selected_source = []

    if setting in ("full", "none", "no_ablation"):
        pass
    elif setting in ("deny_memory_to_memory_heads", "deny_previous_to_memory_kv_heads"):
        deny_previous_source = memory
        selected_source = memory
    elif setting in ("deny_current_to_current_heads", "deny_current_to_current_kv_heads"):
        deny_current_source = current
        selected_source = current
    elif setting in ("deny_current_to_memory_heads", "deny_current_to_memory_kv_heads"):
        deny_current_source = memory
        selected_source = memory
    elif setting in ("deny_memory_to_current_heads", "deny_previous_to_current_kv_heads"):
        deny_previous_source = current
        selected_source = current
    elif setting in ("random_layer_matched_memory_denial", "random_layer_matched_previous_kv_denial"):
        selected_source = _layer_matched_random(
            memory,
            num_layers,
            selector_num_heads,
            seed,
            exclude_heads=memory,
        )
        deny_previous_source = selected_source
    elif setting in ("random_layer_matched_current_denial", "random_layer_matched_current_kv_denial"):
        selected_source = _layer_matched_random(
            current,
            num_layers,
            selector_num_heads,
            seed,
            exclude_heads=current,
        )
        deny_current_source = selected_source
    else:
        raise ValueError(f"Unknown context-denial setting: {setting}")

    if head_granularity == "kv":
        deny_previous = _expand_kv_heads_to_query_heads(
            deny_previous_source,
            group_size,
            num_query_heads,
        )
        deny_current = _expand_kv_heads_to_query_heads(
            deny_current_source,
            group_size,
            num_query_heads,
        )
        selected = _expand_kv_heads_to_query_heads(
            selected_source,
            group_size,
            num_query_heads,
        )
    else:
        deny_previous = deny_previous_source
        deny_current = deny_current_source
        selected = selected_source

    return {
        "setting": setting,
        "deny_previous_by_layer": _heads_by_layer(deny_previous),
        "deny_current_by_layer": _heads_by_layer(deny_current),
        "selected_heads": _normalize_heads(selected, num_layers, num_query_heads),
        "num_selected_heads": int(len(selected)),
        "head_granularity": head_granularity,
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(num_kv_heads) if num_kv_heads else None,
        "group_size": int(group_size) if group_size else None,
        "selected_source_heads": _normalize_heads(
            selected_source,
            num_layers,
            selector_num_heads,
        ),
        "num_selected_source_heads": int(len(selected_source)),
    }


def _tensor_positions(start, end, device):
    start = int(start)
    end = int(end)
    if end <= start:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.arange(start, end, device=device, dtype=torch.long)


def _store_dense_last_chunk_positions(model, pre_lens, post_lens):
    """Store previous/latest positions before any HERMES compression."""
    _require_torch()
    device = model.device
    visual_start = int(model.visual_start_idx)
    spans = {}
    for layer_idx, pre_len in enumerate(pre_lens):
        post_len = int(post_lens[layer_idx])
        pre_len = int(pre_len)
        prev_start = min(visual_start, post_len)
        prev_end = min(max(pre_len, visual_start), post_len)
        curr_start = min(max(pre_len, visual_start), post_len)
        curr_end = max(curr_start, post_len)
        spans[layer_idx] = {
            "previous_positions": _tensor_positions(prev_start, prev_end, device),
            "current_positions": _tensor_positions(curr_start, curr_end, device),
            "mode": "dense_before_or_without_compression",
        }
    model._context_denial_spans = spans
    model._context_denial_last_pre_lens = [int(x) for x in pre_lens]
    model._context_denial_last_post_lens = [int(x) for x in post_lens]


def _store_compacted_last_chunk_positions(model, keep_indices_all_layers):
    """Map previous/latest positions through HERMES strict-shrink keep indices.

    Long-term summary tokens appended by HERMES are intentionally left visible:
    they can mix previous and current evidence, so assigning them to one side
    would make the ablation harder to interpret.
    """
    if not hasattr(model, "_context_denial_last_pre_lens"):
        return

    _require_torch()
    device = model.device
    visual_start = int(model.visual_start_idx)
    pre_lens = model._context_denial_last_pre_lens
    post_lens = model._context_denial_last_post_lens
    curr_lens = model._get_cache_seq_len_per_layer()
    spans = {}

    for layer_idx, keep in enumerate(keep_indices_all_layers):
        seq_len = int(curr_lens[layer_idx])
        keep_tensor = torch.as_tensor(keep, device=device, dtype=torch.long)
        if hasattr(model, "_sanitize_keep_indices"):
            safe = model._sanitize_keep_indices(keep_tensor, seq_len)
        else:
            safe = keep_tensor[(keep_tensor >= 0) & (keep_tensor < seq_len)]
            safe = torch.unique(safe, sorted=True)

        pre_len = int(pre_lens[layer_idx])
        post_len = int(post_lens[layer_idx])
        prev_mask = (safe >= visual_start) & (safe < pre_len)
        curr_mask = (safe >= pre_len) & (safe < post_len)
        compact_pos = torch.arange(safe.numel(), device=device, dtype=torch.long)
        spans[layer_idx] = {
            "previous_positions": compact_pos[prev_mask],
            "current_positions": compact_pos[curr_mask],
            "mode": "compacted_after_hermes_pruning",
        }

    model._context_denial_spans = spans


def _build_context_denial_mask(
    model,
    layer_idx,
    batch,
    q_len,
    past_len,
    hidden_states,
    num_query_heads,
):
    device = hidden_states.device
    mask = torch.zeros(
        (batch, num_query_heads, q_len, past_len + q_len),
        dtype=torch.bool,
        device=device,
    )

    if past_len > 0:
        mask[:, :, :, :past_len] = True

    spans = getattr(model, "_context_denial_spans", {}).get(layer_idx, {})
    previous_positions = spans.get("previous_positions", None)
    current_positions = spans.get("current_positions", None)

    deny_previous = getattr(model, "_context_denial_deny_previous_by_layer", {}).get(layer_idx, [])
    deny_current = getattr(model, "_context_denial_deny_current_by_layer", {}).get(layer_idx, [])

    if previous_positions is not None and len(deny_previous) > 0:
        previous_positions = previous_positions.to(device=device, dtype=torch.long)
        previous_positions = previous_positions[(previous_positions >= 0) & (previous_positions < past_len)]
        if previous_positions.numel() > 0:
            for head in deny_previous:
                if 0 <= int(head) < num_query_heads:
                    mask[:, int(head), :, previous_positions] = False

    if current_positions is not None and len(deny_current) > 0:
        current_positions = current_positions.to(device=device, dtype=torch.long)
        current_positions = current_positions[(current_positions >= 0) & (current_positions < past_len)]
        if current_positions.numel() > 0:
            for head in deny_current:
                if 0 <= int(head) < num_query_heads:
                    mask[:, int(head), :, current_positions] = False

    if q_len > 0:
        causal_current = torch.tril(
            torch.ones((q_len, q_len), dtype=torch.bool, device=device)
        )
        mask[:, :, :, past_len:past_len + q_len] = causal_current.view(1, 1, q_len, q_len)

    attn_impl = getattr(_get_language_config(model), "_attn_implementation", "sdpa")
    if attn_impl == "eager":
        additive = torch.zeros(mask.shape, dtype=hidden_states.dtype, device=device)
        return additive.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
    return mask


def _install_context_denial_hooks(model, num_query_heads, max_mask_q_len):
    if not hasattr(model.language_model, "layers"):
        raise ValueError("Context denial hooks require Qwen-style language_model.layers")

    for handle in getattr(model, "_context_denial_hook_handles", []):
        handle.remove()

    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            if not getattr(model, "_context_denial_enabled", False):
                return args, kwargs

            hidden_states = args[0] if args else kwargs["hidden_states"]
            batch, q_len = hidden_states.shape[:2]
            if max_mask_q_len >= 0 and q_len > max_mask_q_len:
                return args, kwargs

            past_key_values = kwargs.get("past_key_values", None)
            past_len = _get_layer_past_len(past_key_values, layer_idx)
            if past_len <= 0:
                return args, kwargs

            kwargs["attention_mask"] = _build_context_denial_mask(
                model=model,
                layer_idx=layer_idx,
                batch=batch,
                q_len=q_len,
                past_len=past_len,
                hidden_states=hidden_states,
                num_query_heads=num_query_heads,
            )
            return args, kwargs

        return hook

    for layer_idx, layer in enumerate(model.language_model.layers):
        handles.append(layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

    model._context_denial_hook_handles = handles


def apply_context_denial_ablation(
    model,
    head_classes,
    setting,
    seed=0,
    num_query_heads=None,
    num_kv_heads=None,
    max_mask_q_len=512,
    head_granularity=None,
):
    """Install context-denial ablation on a HERMES model.

    The ablation is active only inside ``model.question_answering``. Video
    encoding and HERMES pseudo-query compression remain unchanged.
    """
    _require_torch()
    num_query_heads, detected_num_kv_heads, group_size = _get_attention_shape_full(
        model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )
    selected = select_denial_heads(
        head_classes,
        setting,
        seed=seed,
        head_granularity=head_granularity,
        num_query_heads=num_query_heads,
        num_kv_heads=detected_num_kv_heads,
    )

    model._context_denial_enabled = False
    model._context_denial_setting = str(setting)
    model._context_denial_deny_previous_by_layer = selected["deny_previous_by_layer"]
    model._context_denial_deny_current_by_layer = selected["deny_current_by_layer"]
    model._context_denial_selected_heads = selected["selected_heads"]
    model._context_denial_max_mask_q_len = int(max_mask_q_len)
    model._context_denial_num_query_heads = int(num_query_heads)

    _install_context_denial_hooks(model, num_query_heads, int(max_mask_q_len))

    if not hasattr(model, "_context_denial_original_encode_video_chunk"):
        model._context_denial_original_encode_video_chunk = model.encode_video_chunk

        def encode_video_chunk_with_tracking(self, video_chunk):
            pre_lens = self._get_cache_seq_len_per_layer()
            result = self._context_denial_original_encode_video_chunk(video_chunk)
            post_lens = self._get_cache_seq_len_per_layer()
            _store_dense_last_chunk_positions(self, pre_lens, post_lens)
            return result

        model.encode_video_chunk = MethodType(encode_video_chunk_with_tracking, model)

    if not hasattr(model, "_context_denial_original_apply_kv_cache_pruning_strict"):
        model._context_denial_original_apply_kv_cache_pruning_strict = model.apply_kv_cache_pruning_strict

        def apply_kv_cache_pruning_strict_with_tracking(self, keep_indices_all_layers):
            _store_compacted_last_chunk_positions(self, keep_indices_all_layers)
            return self._context_denial_original_apply_kv_cache_pruning_strict(keep_indices_all_layers)

        model.apply_kv_cache_pruning_strict = MethodType(
            apply_kv_cache_pruning_strict_with_tracking,
            model,
        )

    if not hasattr(model, "_context_denial_original_question_answering"):
        model._context_denial_original_question_answering = model.question_answering

        def question_answering_with_context_denial(self, *args, **kwargs):
            prev_enabled = getattr(self, "_context_denial_enabled", False)
            self._context_denial_enabled = True
            try:
                return self._context_denial_original_question_answering(*args, **kwargs)
            finally:
                self._context_denial_enabled = prev_enabled

        model.question_answering = MethodType(question_answering_with_context_denial, model)

    model._context_denial_config = {
        "setting": str(setting),
        "seed": int(seed),
        "head_granularity": selected["head_granularity"],
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(detected_num_kv_heads),
        "group_size": int(group_size),
        "max_mask_q_len": int(max_mask_q_len),
        "num_selected_heads": int(selected["num_selected_heads"]),
        "selected_heads": selected["selected_heads"],
        "num_selected_source_heads": int(selected["num_selected_source_heads"]),
        "selected_source_heads": selected["selected_source_heads"],
        "deny_previous_by_layer": selected["deny_previous_by_layer"],
        "deny_current_by_layer": selected["deny_current_by_layer"],
        "mode": "qa_only_headwise_context_access_ablation",
    }
    print(
        "[context_denial] Installed QA-only context denial: "
        f"setting={setting}, granularity={selected['head_granularity']}, "
        f"source_heads={selected['num_selected_source_heads']}, "
        f"query_heads={selected['num_selected_heads']}, "
        f"max_mask_q_len={int(max_mask_q_len)}"
    )
    return model
