"""
层级异构预算 — 只改 HERMES prune_kv_cache_by_attention 的预算分配

每层根据 SparseMM 头分数分配不同 budget（层内 4 头统一）。
不动 attention forward, 不动 KV 格式, 不动原代码。
"""

import torch, numpy as np


def install(model, head_scores=None):
    """替换 model.prune_kv_cache_by_attention, 注入层级预算"""
    nl, nkv, nqpk = 28, 4, 7

    if head_scores is not None:
        ks = head_scores.reshape(28, nkv, nqpk).mean(axis=2)  # [28,4] per-KV-head
        ls = ks.mean(axis=1) / ks.mean()                       # [28] per-layer, 均值=1
    else:
        ls = np.ones(nl)

    # 预算统一 6000（避免层间不等长导致 HF mask 裂），α 逐层调整
    alpha_adj = np.clip(1.0 - (ls - 1) * 0.3, 0, 2)

    original_prune = model.prune_kv_cache_by_attention

    def layer_budget_prune(attn_weights_local, attn_weights_global,
                            attn_weights_mixed, num_keep=3000):
        device = model.device
        visual_start = model.visual_start_idx
        n_layers = len(attn_weights_local)

        keep_indices_all_layers = []

        for layer_idx in range(n_layers):
            if layer_idx < model.short_term_threshold:
                aw, ql = attn_weights_local[layer_idx], attn_weights_local[0].shape[2]
                alpha, k = 1.0, 20.0
            elif layer_idx >= model.long_term_threshold:
                aw, ql = attn_weights_global[layer_idx], attn_weights_global[0].shape[2]
                alpha, k = 0.0, 0.0
            else:
                aw, ql = attn_weights_mixed[layer_idx], attn_weights_mixed[0].shape[2]
                p = (layer_idx - model.short_term_threshold) / (
                    model.long_term_threshold - model.short_term_threshold)
                alpha, k = 0.75 - 0.6 * p, 20.0 - 12.0 * p

            # SparseMM α 调整
            aa = float(alpha_adj[layer_idx])
            alpha = max(0.0, min(1.0, alpha * aa))

            if aw.dim() < 4:
                keep_indices_all_layers.append(list(range(aw.shape[3])))
                continue

            visual_attn = aw[0].mean(dim=1)[:, visual_start:-ql]
            num_visual = visual_attn.shape[1]
            lb = min(model.kv_size, num_visual)
            lb = max(10, lb)

            positions = torch.arange(num_visual, device=device, dtype=torch.float32)
            time_dist = (num_visual - 1 - positions) / max(num_visual - 1, 1)

            attn_norm = (visual_attn.mean(dim=0) - visual_attn.mean(dim=0).min()) / (
                visual_attn.mean(dim=0).max() - visual_attn.mean(dim=0).min() + 1e-6)
            recency = torch.exp(-k * time_dist)
            recency_norm = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)

            score = attn_norm * (1 - alpha) + recency_norm * alpha
            _, topk = torch.topk(score, lb)
            keep = torch.sort(topk + visual_start)[0]

            keep_indices = torch.cat([
                torch.arange(visual_start, device=device), keep
            ]).unique().tolist()
            keep_indices_all_layers.append(keep_indices)

        return keep_indices_all_layers

    model.prune_kv_cache_by_attention = layer_budget_prune
    model._alpha_adj = alpha_adj
    model._alpha_adj = alpha_adj

    print(f"[layer_budget] Budget=6000, α range: [{alpha_adj.min():.2f}, {alpha_adj.max():.2f}]")
    return model
