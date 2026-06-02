"""
HERMES + SparseMM-style head-level KV budget allocation

在 prune_kv_cache_by_attention 里，每个 attention 头用不同的权重做 token 重要性打分，
然后用投票机制（voting）把所有头的 top-K token 合并成每层的 keep_indices。

不动原代码，纯外挂。

原理:
  长期偏好头 → 给 global_attention 高分, recency 低权重
  近期偏好头 → 给 local_attention 高分, attention 低权重
  无关头 → 均权

用法:
  from head_analysis.hermes_head_budget import apply_head_budget
  model = apply_head_budget(model, head_scores_path)

  # 加载 SparseMM 的头分数 (qwen.json)，或自己的 head_pseudo.npz
  head_scores_path = "/home/sjs/SparseMM/visual_head/head_score/qwen.json"
"""

import json
import os
import numpy as np
import torch


def load_sparsemm_scores(json_path, num_layers=28, num_heads=28):
    """加载 SparseMM 风格的 head score JSON.

    SparseMM 的 visual-head JSON 中，每个 layer-head 对应一个统计序列；
    原实现使用 np.mean(list) 作为该 head 的 score。这里保持同一语义，
    避免只取某个位置的统计值导致 score 过稀疏。
    """
    with open(json_path) as f:
        raw = json.load(f)

    scores = np.zeros((num_layers, num_heads))
    for key, val in raw.items():
        l, h = map(int, key.split('-'))
        if isinstance(val, list) and len(val) > 0:
            numeric_vals = [float(x) for x in val if isinstance(x, (int, float))]
            scores[l, h] = float(np.mean(numeric_vals)) if numeric_vals else 0.0
        elif isinstance(val, (int, float)):
            scores[l, h] = float(val)
    return scores


def load_pseudo_scores(npz_path, num_layers=28, num_heads=28):
    """加载 head analysis 打出的 shift 分数"""
    data = np.load(npz_path)
    shift = data['shift']  # [28, 28]
    return shift


def load_csv_scores(csv_path, num_layers=28, num_heads=28):
    """Load head scores from CSV.

    Supported schemas:
      - layer,head,<score-column>
      - layer,kv_head,<score-column>

    For KV-head CSVs, each KV score is expanded to the query heads that share
    that KV cache. This keeps downstream code compatible with the existing
    [num_layers, num_query_heads] score interface.
    """
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "layer" not in fieldnames:
        raise ValueError(f"{csv_path} must contain a 'layer' column")

    preferred = [
        "internal_top100_mean_attention",
        "spiky_readout_score",
        "readout_shape_score",
        "score",
        "score_sum",
        "score_mean_q",
        "score_max_q",
    ]
    score_col = next((c for c in preferred if c in fieldnames), None)
    if score_col is None:
        numeric = [c for c in fieldnames if c not in ("layer", "head", "kv_head", "q_heads")]
        if not numeric:
            raise ValueError(f"{csv_path} has no recognizable score column")
        score_col = numeric[0]

    scores = np.zeros((int(num_layers), int(num_heads)), dtype=np.float64)
    if "head" in fieldnames:
        for row in rows:
            layer = int(row["layer"])
            head = int(row["head"])
            if 0 <= layer < num_layers and 0 <= head < num_heads:
                scores[layer, head] = float(row[score_col])
        return scores

    if "kv_head" not in fieldnames:
        raise ValueError(f"{csv_path} must contain either 'head' or 'kv_head'")

    # Qwen2.5-VL-7B uses 28 query heads / 4 KV heads. Keep this generic when
    # possible by inferring KV count from the CSV.
    max_kv = max(int(row["kv_head"]) for row in rows) if rows else 0
    num_kv_heads = max_kv + 1
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot expand {num_kv_heads} KV heads to {num_heads} query heads"
        )
    group_size = int(num_heads) // int(num_kv_heads)
    for row in rows:
        layer = int(row["layer"])
        kv_head = int(row["kv_head"])
        if not (0 <= layer < num_layers and 0 <= kv_head < num_kv_heads):
            continue
        value = float(row[score_col])
        start = kv_head * group_size
        end = min(start + group_size, int(num_heads))
        scores[layer, start:end] = value
    return scores


def load_head_scores(path_or_alias, num_layers=28, num_heads=28):
    """加载 head score，支持 pseudo / sparsemm / sparsemm_qwen25 短别名。"""
    if path_or_alias == "sparsemm":
        path_or_alias = "/home/sjs/SparseMM/visual_head/head_score/qwen.json"
    elif path_or_alias in ("sparsemm_qwen25", "sparsemm_qwen2.5", "qwen25"):
        path_or_alias = "/home/sjs/SparseMM/visual_head/head_score/qwen2.5-vl.json"
    elif path_or_alias == "pseudo":
        path_or_alias = (
            "results/head_analysis/pseudo-qwen2.5_vl_7b-kv6000-hermes/"
            "head_pseudo.npz"
        )

    if not os.path.exists(path_or_alias):
        raise FileNotFoundError(f"Head score file not found: {path_or_alias}")

    if path_or_alias.endswith('.json'):
        return load_sparsemm_scores(path_or_alias, num_layers, num_heads)
    if path_or_alias.endswith('.npz'):
        return load_pseudo_scores(path_or_alias, num_layers, num_heads)
    if path_or_alias.endswith('.csv'):
        return load_csv_scores(path_or_alias, num_layers, num_heads)
    raise ValueError(f"Unknown score format: {path_or_alias}")


def build_head_weights(scores, num_layers=28, num_heads=28):
    """
    把头分数转成 per-head 的 (attn_weight, recency_weight) 对

    scores: [28, 28], 正值=长期偏好, 负值=近期偏好, 零=无关
    """
    # 归一化到 [-1, 1]
    vmax = max(abs(scores.max()), abs(scores.min()), 1e-6)
    norm = scores / vmax

    # 长期头（norm > 0.1）: attn 高权重，recency 低权重
    # 近期头（norm < -0.1）: recency 高权重，attn 低权重
    # 无关头（|norm| <= 0.1）: 均权
    attn_w = np.ones((num_layers, num_heads)) * 0.5
    recency_w = np.ones((num_layers, num_heads)) * 0.5

    long_mask = norm > 0.1
    short_mask = norm < -0.1

    # 长期头：attn 权重增加
    attn_w[long_mask] = 0.5 + 0.5 * norm[long_mask].clip(0, 1)
    recency_w[long_mask] = 1.0 - attn_w[long_mask]

    # 短期头：recency 权重增加
    recency_w[short_mask] = 0.5 + 0.5 * (-norm[short_mask]).clip(0, 1)
    attn_w[short_mask] = 1.0 - recency_w[short_mask]

    return attn_w, recency_w


def _allocate_integer_budget(weights, total_budget, min_budget=0, max_budget=None):
    """按正权重分配整数预算，并尽量保持总和等于 total_budget。"""
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


def build_budget_tables(scores, num_keep, num_layers=28, num_heads=28,
                        layer_budget_strength=0.5,
                        head_budget_strength=1.0,
                        layer_min_ratio=0.75,
                        layer_max_ratio=1.25,
                        head_min_ratio=0.25,
                        head_max_ratio=2.0,
                        min_head_budget=8):
    """
    从 head score 构造异构预算表。

    返回:
      layer_budgets: [num_layers]，每层最终保留的视觉 token 预算
      head_budgets:  [num_layers, num_heads]，每个 Q-head 的投票 top-K 预算

    注意：最终 KV cache 仍然是每层一个统一 keep set，不是 ragged per-head KV。
    这样能保持 transformers/HERMES 原有 forward 兼容。
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (num_layers, num_heads):
        raise ValueError(
            f"scores shape must be {(num_layers, num_heads)}, got {scores.shape}"
        )

    if not np.isfinite(scores).all():
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = max(abs(scores.max()), abs(scores.min()), 1e-6)
    magnitude = np.abs(scores / vmax)

    if magnitude.max() < 1e-6:
        layer_weights = np.ones(num_layers, dtype=np.float64)
        head_weights = np.ones((num_layers, num_heads), dtype=np.float64)
    else:
        global_mean = magnitude.mean() + 1e-6
        layer_weights = 1.0 + layer_budget_strength * (
            magnitude.mean(axis=1) / global_mean - 1.0
        )
        layer_weights = np.clip(layer_weights, layer_min_ratio, layer_max_ratio)

        head_weights = np.ones((num_layers, num_heads), dtype=np.float64)
        for layer_idx in range(num_layers):
            layer_mean = magnitude[layer_idx].mean() + 1e-6
            head_weights[layer_idx] = 1.0 + head_budget_strength * (
                magnitude[layer_idx] / layer_mean - 1.0
            )
            head_weights[layer_idx] = np.clip(
                head_weights[layer_idx], head_min_ratio, head_max_ratio
            )

    total_layer_budget = int(num_keep) * int(num_layers)
    min_layer_budget = max(1, int(round(num_keep * layer_min_ratio)))
    max_layer_budget = max(min_layer_budget, int(round(num_keep * layer_max_ratio)))
    layer_budgets = _allocate_integer_budget(
        layer_weights,
        total_layer_budget,
        min_budget=min_layer_budget,
        max_budget=max_layer_budget,
    )

    head_budgets = np.zeros((num_layers, num_heads), dtype=int)
    for layer_idx in range(num_layers):
        layer_budget = int(layer_budgets[layer_idx])
        base_head_budget = max(1, layer_budget / max(num_heads, 1))
        min_h = max(0, min(int(min_head_budget), layer_budget // max(num_heads, 1)))
        min_h = max(min_h, int(round(base_head_budget * head_min_ratio)))
        max_h = max(min_h, int(round(base_head_budget * head_max_ratio)))
        head_budgets[layer_idx] = _allocate_integer_budget(
            head_weights[layer_idx],
            layer_budget,
            min_budget=min_h,
            max_budget=max_h,
        )

    return layer_budgets, head_budgets


def apply_head_budget(model, head_scores_path=None, scores=None,
                       num_layers=28, num_heads=28,
                       layer_budget_strength=0.5,
                       head_budget_strength=1.0,
                       layer_min_ratio=0.75,
                       layer_max_ratio=1.25,
                       head_min_ratio=0.25,
                       head_max_ratio=2.0,
                       min_head_budget=8):
    """
    给模型挂上头级预算的 prune_kv_cache_by_attention

    model: QwenVL_Hermes 实例
    head_scores_path: SparseMM qwen.json 或 head_pseudo.npz 的路径
    scores: 直接传入 [28, 28] 分数矩阵，优先级高于 head_scores_path
    """
    if scores is None and head_scores_path is not None:
        scores = load_head_scores(head_scores_path, num_layers, num_heads)

    if scores is None:
        # 默认：均匀（等于原始行为）
        scores = np.zeros((num_layers, num_heads))
        print("[head_budget] No scores provided, using uniform (original behavior)")

    print(f"[head_budget] Loaded scores. Range: [{scores.min():.4f}, {scores.max():.4f}]")

    attn_w, recency_w = build_head_weights(scores, num_layers, num_heads)

    def head_budget_prune(attn_weights_local, attn_weights_global,
                           attn_weights_mixed, num_keep=3000):
        """
        Per-head weighted scoring + voting:
        1. 每个头用自己的 (attn_w, recency_w) 计算 token 重要性
        2. 每个头选 top-K（K = 头级预算）
        3. 投票合并 → 每层 keep_indices
        """
        device = model.device
        visual_start_idx = model.visual_start_idx
        n_layers = len(attn_weights_local)

        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        actual_num_heads = attn_weights_local[0].shape[1]
        if actual_num_heads != num_heads:
            raise ValueError(
                f"Configured num_heads={num_heads}, but attention has "
                f"{actual_num_heads} heads"
            )

        layer_budgets, head_budget_table = build_budget_tables(
            scores,
            num_keep,
            num_layers=n_layers,
            num_heads=actual_num_heads,
            layer_budget_strength=layer_budget_strength,
            head_budget_strength=head_budget_strength,
            layer_min_ratio=layer_min_ratio,
            layer_max_ratio=layer_max_ratio,
            head_min_ratio=head_min_ratio,
            head_max_ratio=head_max_ratio,
            min_head_budget=min_head_budget,
        )
        model._head_budget_layer_budgets = layer_budgets
        model._head_budget_head_budgets = head_budget_table

        keep_indices_all_layers = []

        for layer_idx in range(n_layers):
            # 原始层类型判定（沿用 HERMES 的分层）
            if layer_idx < model.short_term_threshold:
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
            elif layer_idx >= model.long_term_threshold:
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
            else:
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                progress = (layer_idx - model.short_term_threshold) / (
                    model.long_term_threshold - model.short_term_threshold)

            # [heads, n_visual]: 取视觉部分，对 query 维度平均
            visual_attn = layer_attn_weights[0].mean(dim=1)[:, visual_start_idx:-question_len]

            num_visual_tokens = visual_attn.shape[1]
            if num_visual_tokens <= 0:
                keep_indices_all_layers.append(
                    torch.arange(visual_start_idx, device=device).tolist()
                )
                continue

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)

            # ---- Per-head scoring ----
            head_scores_list = []  # 每个头的 token 分数
            head_budgets = head_budget_table[layer_idx]

            for head_idx in range(visual_attn.shape[0]):
                # 该头的权重
                aw = float(attn_w[layer_idx, head_idx])
                rw = float(recency_w[layer_idx, head_idx])

                h_attn = visual_attn[head_idx]

                # attention 分数
                attn_norm = (h_attn - h_attn.min()) / (h_attn.max() - h_attn.min() + 1e-6)

                # recency 分数 (HERMES 公式: exponential decay)
                k = 20 if layer_idx < model.short_term_threshold else (
                    0 if layer_idx >= model.long_term_threshold else 20 - 12 * progress)
                recency_w_raw = torch.exp(-k * time_distances)
                recency_norm = (recency_w_raw - recency_w_raw.min()) / (
                    recency_w_raw.max() - recency_w_raw.min() + 1e-6)

                # 加权合并
                head_score = attn_norm * aw + recency_norm * rw
                head_scores_list.append(head_score)

            # ---- Voting: 各头选 top-K，投票决定每层的 keep token ----
            all_votes = torch.zeros(num_visual_tokens, device=device)
            layer_score = torch.zeros(num_visual_tokens, device=device)

            for head_idx in range(visual_attn.shape[0]):
                score = head_scores_list[head_idx]
                layer_score += score

                k = min(int(head_budgets[head_idx]), num_visual_tokens)
                if k <= 0:
                    continue
                _, topk = torch.topk(score, k)
                all_votes[topk] += 1.0 / max(k, 1)  # 每个头总投票质量约为 1

            # 用很小的 layer_score 补齐未被任何头投票的 token，避免 topk 随机选 0 票项。
            layer_score = layer_score / max(visual_attn.shape[0], 1)
            layer_score = (layer_score - layer_score.min()) / (
                layer_score.max() - layer_score.min() + 1e-6
            )
            combined_score = all_votes + 1e-3 * layer_score

            # 取该层预算对应的 keep tokens；不同层可以不同。
            actual_keep = min(int(layer_budgets[layer_idx]), num_visual_tokens)
            _, keep_indices_rel = torch.topk(combined_score, actual_keep)
            keep_indices = torch.sort(keep_indices_rel + visual_start_idx)[0]

            # 加上 text token
            full_keep = torch.cat([
                torch.arange(visual_start_idx, device=device),
                keep_indices
            ]).unique()

            keep_indices_all_layers.append(full_keep.tolist())

        return keep_indices_all_layers

    model.prune_kv_cache_by_attention = head_budget_prune
    model._head_attn_w = attn_w
    model._head_recency_w = recency_w
    model._head_budget_config = {
        "layer_budget_strength": layer_budget_strength,
        "head_budget_strength": head_budget_strength,
        "layer_min_ratio": layer_min_ratio,
        "layer_max_ratio": layer_max_ratio,
        "head_min_ratio": head_min_ratio,
        "head_max_ratio": head_max_ratio,
        "min_head_budget": min_head_budget,
    }
    print("[head_budget] Per-head prune_kv_cache_by_attention installed.")

    return model
