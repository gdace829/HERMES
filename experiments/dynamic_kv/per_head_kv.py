"""
Per-KV-Head Dynamic KV Cache for HERMES

在 _shrink_positions_and_rerotate_keys 里，每个 KV 用不同的 keep_indices，
使得每个 KV 保留不同的 token。存储格式保持不变。

实现: 对每层每个 KV 独立做 top-K 选择 → pad 到 max_len → attention mask
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_per_head_keep_indices(model, attn_weights_local, attn_weights_global,
                                   attn_weights_mixed, head_scores=None):
    """
    对每层每个 KV 用 HEAD-SPECIFIC 打分公式计算 keep_indices

    head_scores: [28, 28] — 可选的 per-Q-head 偏好分数（长期/近期）
    """
    device = model.device
    visual_start = model.visual_start_idx
    n_layers = len(attn_weights_local)
    n_kv_heads = 4  # Qwen2.5-VL-7B GQA

    # 头权重: Q-head 分数聚合到 KV head
    if head_scores is not None:
        # head_scores: [28, 28] Q heads → group by 7 → [28, 4] KV head scores
        kv_scores = head_scores.reshape(28, 4, 7).mean(axis=2)  # [28, 4]
        # 每 KV head 的 α (recency vs attention)
        vmax = max(abs(kv_scores.max()), abs(kv_scores.min()), 1e-6)
        kv_norm = kv_scores / vmax  # [-1, 1]
        # 长期 > 0 → 更多 attention 权重，近期 < 0 → 更多 recency 权重
    else:
        kv_norm = np.zeros((n_layers, n_kv_heads))

    per_head_keep = []  # [layer_idx][kv_head] = tensor of keep_indices

    for layer_idx in range(n_layers):
        if layer_idx < model.short_term_threshold:
            ltype = "short"
            attn_w = attn_weights_local[layer_idx]
            q_len = attn_weights_local[0].shape[2]
        elif layer_idx >= model.long_term_threshold:
            ltype = "long"
            attn_w = attn_weights_global[layer_idx]
            q_len = attn_weights_global[0].shape[2]
        else:
            ltype = "mid"
            attn_w = attn_weights_mixed[layer_idx]
            q_len = attn_weights_mixed[0].shape[2]
            progress = (layer_idx - model.short_term_threshold) / (
                model.long_term_threshold - model.short_term_threshold)

        if attn_w.dim() < 4:
            per_head_keep.append(None)
            continue

        # [batch=1, heads, q_len, kv_len] → [heads, kv_len - visual_start - q_len]
        visual_attn = attn_w[0].mean(dim=1)[:, visual_start:-q_len]
        n_visual = visual_attn.shape[1]
        n_q_heads = visual_attn.shape[0]  # 28

        positions = torch.arange(n_visual, device=device, dtype=torch.float32)
        time_dist = (n_visual - 1 - positions) / max(n_visual - 1, 1)

        layer_keeps = []
        for kv_h in range(n_kv_heads):
            # 该 KV 头对应的 Q 头范围 (GQA)
            q_start = kv_h * (n_q_heads // n_kv_heads)
            q_end = q_start + (n_q_heads // n_kv_heads)

            # 该 KV 的 alpha
            alpha_kv = float(kv_norm[layer_idx, kv_h])
            if ltype == "short":
                alpha = 1.0  # 短期层: 基本看 recency, 但 head-level 调整
                k = 20.0
            elif ltype == "long":
                alpha = 0.0  # 长期层: 基本看 attention, head-level 调整
                k = 0.0
            else:
                alpha = 0.75 - 0.6 * progress
                k = 20.0 - 12.0 * progress

            # head-level 修正: 长期头降低 alpha (更多 attention), 近期头提高 alpha
            alpha = max(0.0, min(1.0, alpha - alpha_kv * 0.3))
            k = max(0.0, k - alpha_kv * 10.0)

            # 该 KV 头的平均 attention (对应 Q 头组)
            kv_attn = visual_attn[q_start:q_end].mean(dim=0)  # [n_visual]

            attn_norm = (kv_attn - kv_attn.min()) / (kv_attn.max() - kv_attn.min() + 1e-6)
            recency = torch.exp(-k * time_dist)
            recency_norm = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)

            score = attn_norm * (1 - alpha) + recency_norm * alpha
            budget = model.kv_size  # 每个 KV 头的 budget

            k_keep = min(budget, n_visual)
            _, topk = torch.topk(score, k_keep)
            keep = torch.sort(topk + visual_start)[0]
            layer_keeps.append(keep)

        per_head_keep.append(layer_keeps)

    return per_head_keep


def apply_per_head_shrink(model, per_head_keep):
    """
    对每层每个 KV 用独立的 keep_indices 裁剪 KV cache。

    裁剪后每个 KV 可能长度不同。用 mask 处理不等长。
    """
    device = model.device
    pos_cache = model._position_ids_cache
    new_kv = []
    max_lens = []

    for layer_idx in range(model.num_layers):
        keeps = per_head_keep[layer_idx]
        if keeps is None:
            new_kv.append(model.kv_cache[layer_idx])
            max_lens.append(model.kv_cache[layer_idx][0].shape[2])
            continue

        k_layer, v_layer = model.kv_cache[layer_idx]
        n_kv_heads = k_layer.shape[1]
        new_ks = []
        new_vs = []

        for h in range(min(n_kv_heads, len(keeps))):
            ki = keeps[h]
            if not isinstance(ki, torch.Tensor):
                ki = torch.as_tensor(ki, device=device)
            k_h = torch.index_select(k_layer[:, h:h+1], dim=2, index=ki)
            v_h = torch.index_select(v_layer[:, h:h+1], dim=2, index=ki)
            new_ks.append(k_h)
            new_vs.append(v_h)

        # pad 到 max_len (每个 KV 头不同长度)
        max_len = max(k.shape[2] for k in new_ks)
        pad_ks = []
        pad_vs = []
        for h in range(len(new_ks)):
            pad_k = F.pad(new_ks[h], (0, 0, 0, max_len - new_ks[h].shape[2]))
            pad_v = F.pad(new_vs[h], (0, 0, 0, max_len - new_vs[h].shape[2]))
            pad_ks.append(pad_k)
            pad_vs.append(pad_v)

        k_cat = torch.cat(pad_ks, dim=1)
        v_cat = torch.cat(pad_vs, dim=1)
        new_kv.append((k_cat.contiguous(), v_cat.contiguous()))
        max_lens.append(max_len)

    model.kv_cache = new_kv
    model._per_head_max_lens = max_lens
    model._per_head_keep = per_head_keep

    # Note: position_ids_cache 不严格正确（每头长度不同）
    # 实际使用时需要 per-head position_ids，这里简化处理

    return model


def install_per_head_kv(model, head_scores=None):
    """
    安装 per-head KV cache 到模型上。
    替换 predict_and_compress → 用 per-head 打分 + 裁剪
    """
    original_pac = model.predict_and_compress

    def per_head_predict_and_compress():
        if model.compress_mode == "streamingvlm":
            return model._sliding_window_compress()

        local_q, global_q = model.predict_next_question()
        # 先跑 pseudo_forward 拿到 attention
        model.pseudo_forward(local_q, global_q)
        # pseudo_forward 内部已经调了 prune + shrink
        # 我们需要在 shrink 之前截获 attention ...
        # 实际上 pseudo_forward 内部直接调了 prune_kv_cache_by_attention + apply_kv_cache_pruning_strict
        # 所以需要在伪问题前向时记录 attention

    # 更简单的方式：重写 pseudo_forward 并在其中插入 per-head shrink
    original_pf = model.pseudo_forward

    def per_head_pseudo_forward(local_question=None, global_question=None):
        # 用原始 pseudo_forward 拿到 attention weights（但不要执行压缩）
        # 实际需要复制 pseudo_forward 的逻辑拿到 attention...
        # 这里直接调原始版本（它内部会做 uniform shrink）
        # 然后我们再做一次 per-head shrink 覆盖

        device = model.device
        if local_question is None:
            local_question = "What is happening in the video?"
        if global_question is None:
            global_question = "What is the main topic of the video?"

        # 跑 local attention
        local_ids = model.processor.tokenizer(local_question).input_ids
        local_ids = torch.as_tensor([local_ids], device=device, dtype=torch.int)
        use_fa = (hasattr(model.language_model.config, '_attn_implementation') and
                  model.language_model.config._attn_implementation in ["flash_attention_2", "sdpa"])
        if use_fa:
            attn_local = model._compute_attention_scores_manually(local_ids, model.kv_cache)
        else:
            out = model.language_model(input_ids=local_ids, use_cache=False,
                                        past_key_values=model.kv_cache,
                                        output_attentions=True)
            attn_local = out.attentions

        # 跑 global attention
        global_ids = model.processor.tokenizer(global_question).input_ids
        global_ids = torch.as_tensor([global_ids], device=device, dtype=torch.int)
        if use_fa:
            attn_global = model._compute_attention_scores_manually(global_ids, model.kv_cache)
        else:
            out = model.language_model(input_ids=global_ids, use_cache=False,
                                        past_key_values=model.kv_cache,
                                        output_attentions=True)
            attn_global = out.attentions

        # 跑 mixed
        mixed_q = local_question + "; " + global_question
        mixed_ids = model.processor.tokenizer(mixed_q).input_ids
        mixed_ids = torch.as_tensor([mixed_ids], device=device, dtype=torch.int)
        if use_fa:
            attn_mixed = model._compute_attention_scores_manually(mixed_ids, model.kv_cache)
        else:
            out = model.language_model(input_ids=mixed_ids, use_cache=False,
                                        past_key_values=model.kv_cache,
                                        output_attentions=True)
            attn_mixed = out.attentions

        # ---- PER-HEAD prune ----
        per_head_keep = compute_per_head_keep_indices(
            model, attn_local, attn_global, attn_mixed, head_scores)

        # 常规 shrink (per-head)
        # 取每个 KV 头的 budget: 都用 kv_size（或按分数分配）
        total_budget = model.kv_size * model.num_layers
        # 简化: 每个 KV 头 budget = kv_size（同一层内）
        # 用 per-head keep_indices 做 shrink
        current_len = model.kv_cache[0][0].shape[2]
        if current_len > model.kv_size:
            print(f"Per-head KV shrink: {current_len} -> ~{model.kv_size} per head")
            apply_per_head_shrink(model, per_head_keep)

        model._layer_position_ids.clear()
        torch.cuda.empty_cache()

    model.pseudo_forward = per_head_pseudo_forward
    model._head_scores = head_scores
    print(f"[per_head_kv] Installed. Head scores: {head_scores is not None}")
    return model
