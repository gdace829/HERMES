"""
SparseMM-style Per-Head Flattened KV Cache + Custom Attention

完全对齐 SparseMM:
  - PerHeadFlattenCache: 每头独立展平存储 [tokens, dim]
  - 自定义 attention forward: prefill 用 flash attn, decode 用 per-head gather
  - 不动原模型文件, monkey-patch attention forward
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv,
)


# ============================================================
# 1. Per-Head Flattened Cache (对齐 SparseMM DynamicCacheSplitHeadFlatten)
# ============================================================

class PerHeadFlattenCache:
    """每头独立存储 KV: key_cache[layer][head] = [tokens, dim] (SparseMM 风格)"""

    def __init__(self, num_layers=28, num_kv_heads=4):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.key_cache: List[List[Optional[torch.Tensor]]] = [
            [None] * num_kv_heads for _ in range(num_layers)]
        self.value_cache: List[List[Optional[torch.Tensor]]] = [
            [None] * num_kv_heads for _ in range(num_layers)]


def flatten_to_uniform(cache: PerHeadFlattenCache, layer_idx: int, device, dtype):
    """把 per-head 展平 KV 转换成统一格式 [1, n_kv, max_len, d]"""
    keys = cache.key_cache[layer_idx]
    vals = cache.value_cache[layer_idx]
    if keys[0] is None:
        return None, None

    max_len = max(k.shape[0] for k in keys)
    d = keys[0].shape[1]
    n_kv = len(keys)

    k_out = torch.zeros(1, n_kv, max_len, d, device=device, dtype=dtype)
    v_out = torch.zeros(1, n_kv, max_len, d, device=device, dtype=dtype)
    for h in range(n_kv):
        n = keys[h].shape[0]
        k_out[0, h, :n] = keys[h]
        v_out[0, h, :n] = vals[h]
    return k_out, v_out


# ============================================================
# 2. SparseMM-style Attention Forward
# ============================================================

def make_sparsemm_attn_forward(cache: PerHeadFlattenCache):
    """创建一个自定义 attention forward，处理 per-head 展平 KV cache"""

    def qwen2_5_vl_attn_forward_sparsemm(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        past_key_values = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        n_heads = self.num_heads
        n_kv_heads = self.num_key_value_heads
        n_kv_groups = self.num_key_value_groups
        head_dim = self.head_dim
        layer_idx = self.layer_idx

        # QKV 投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin,
            self.rope_scaling["mrope_section"])

        pkv = past_key_value or past_key_values
        is_decode = (q_len == 1)

        if pkv is not None and cache is not None:
            # 从 per_head cache 重建统一格式 KV
            if cache.key_cache[layer_idx][0] is not None:
                k_pad, v_pad = flatten_to_uniform(
                    cache, layer_idx, key_states.device, key_states.dtype)
                key_states = torch.cat([k_pad, key_states], dim=2)
                value_states = torch.cat([v_pad, value_states], dim=2)
        kv_seq_len = key_states.shape[2]

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, n_kv_groups)
        value_states = repeat_kv(value_states, n_kv_groups)

        # Attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # 更新 KV cache
        if use_cache:
            # 1. 更新 HF DynamicCache（模型需要它）
            from transformers.cache_utils import DynamicCache
            if isinstance(pkv, DynamicCache):
                pkv.update(key_states, value_states, layer_idx)
            # 2. 同步更新 per_head cache
            for h in range(n_kv_heads):
                k_h = key_states[0, h].contiguous()
                v_h = value_states[0, h].contiguous()
                cache.key_cache[layer_idx][h] = k_h
                cache.value_cache[layer_idx][h] = v_h

        return attn_output, None  # decoder 只需要 2 个返回值

    return qwen2_5_vl_attn_forward_sparsemm


# ============================================================
# 3. 安装 — monkey-patch attention + predict_and_compress
# ============================================================

def patch_model(model, head_scores=None):
    """monkey-patch Qwen2.5-VL-7B 的 attention forward 和 predict_and_compress"""

    n_layers = model.num_layers
    n_kv = 4

    # 创建 per-head cache 替换原有 kv_cache
    cache = PerHeadFlattenCache(num_layers=n_layers, num_kv_heads=n_kv)

    # Patch CLASS level (SparseMM 做法): 改类 → 所有实例永久生效
    attn_forward = make_sparsemm_attn_forward(cache)
    # 找到 attention 类并替换其 forward
    attn_cls = type(model.language_model.layers[0].self_attn)
    attn_cls.forward = attn_forward
    print(f"[sparsemm_kv] Patched {attn_cls.__name__}.forward")

    # 头分数
    if head_scores is not None:
        kv_sc = head_scores.reshape(28, n_kv, 7).mean(axis=2)
        total = kv_sc.sum()
        kv_n = kv_sc / total if total > 0 else np.ones((n_layers, n_kv)) / (n_layers * n_kv)
        total_b = model.kv_size * n_layers * n_kv
        min_b = max(10, model.kv_size // 2)
        per_head_budget = np.round(kv_n * (total_b - min_b * n_layers * n_kv) + min_b).astype(int)
    else:
        kv_n = np.zeros((n_layers, n_kv))
        per_head_budget = np.full((n_layers, n_kv), model.kv_size, dtype=int)

    # 替换 predict_and_compress
    def dyn_pac():
        if model.compress_mode == "streamingvlm":
            return

        # 同步 per_head cache → model.kv_cache（legacy tuple 格式）
        legacy_kv = []
        for li in range(n_layers):
            if cache.key_cache[li][0] is not None:
                k, v = flatten_to_uniform(cache, li, model.device, torch.float16)
                legacy_kv.append((k, v))
                ml = k.shape[2]
                p = torch.arange(ml, device=model.device, dtype=torch.float32)
                model._position_ids_cache[li] = p.unsqueeze(0).expand(3, -1).clone()
            else:
                legacy_kv.append(model.kv_cache[li])
        # 临时替换（_compute_attention_scores_manually 和 pseudo_forward 用 legacy 格式）
        saved_kv = model.kv_cache
        model.kv_cache = legacy_kv

        vs = model.visual_start_idx
        local_q, global_q = model.predict_next_question()
        local_ids = model.processor.tokenizer(local_q).input_ids
        local_ids = torch.as_tensor([local_ids], device=model.device, dtype=torch.int)
        global_ids = model.processor.tokenizer(global_q).input_ids
        global_ids = torch.as_tensor([global_ids], device=model.device, dtype=torch.int)

        al = model._compute_attention_scores_manually(local_ids, model.kv_cache)
        ag = model._compute_attention_scores_manually(global_ids, model.kv_cache)

        mixed_q = local_q + "; " + global_q
        mixed_ids = model.processor.tokenizer(mixed_q).input_ids
        mixed_ids = torch.as_tensor([mixed_ids], device=model.device, dtype=torch.int)
        am = model._compute_attention_scores_manually(mixed_ids, model.kv_cache)

        n_q_per_kv = 7

        # 每层 per-head prune
        for layer_idx in range(n_layers):
            # 检查是否需要压缩
            if cache.key_cache[layer_idx][0] is None:
                continue
            total_tokens = cache.key_cache[layer_idx][0].shape[0]
            if total_tokens <= model.kv_size:
                continue

            if layer_idx < model.short_term_threshold:
                aw = al[layer_idx]; ql = al[0].shape[2]; a_base, k_base = 1.0, 20.0
            elif layer_idx >= model.long_term_threshold:
                aw = ag[layer_idx]; ql = ag[0].shape[2]; a_base, k_base = 0.0, 0.0
            else:
                aw = am[layer_idx]; ql = am[0].shape[2]
                p = (layer_idx - model.short_term_threshold) / (
                    model.long_term_threshold - model.short_term_threshold)
                a_base, k_base = 0.75 - 0.6 * p, 20.0 - 12.0 * p

            if aw.dim() < 4:
                continue

            vis = aw[0].mean(dim=1)[:, vs:-ql]
            nv = vis.shape[1]
            pos_ = torch.arange(nv, device=model.device, dtype=torch.float32)
            td = (nv - 1 - pos_) / max(nv - 1, 1)

            for kh in range(n_kv):
                qs, qe = kh * n_q_per_kv, (kh + 1) * n_q_per_kv
                kv_a = vis[qs:qe].mean(dim=0)

                hb = float(kv_n[layer_idx, kh])
                a = max(0.0, min(1.0, a_base - hb * 0.3))
                k = max(0.0, k_base - hb * 10.0)

                an_ = (kv_a - kv_a.min()) / (kv_a.max() - kv_a.min() + 1e-6)
                rn_ = (torch.exp(-k * td) - torch.exp(-k * td).min()) / (
                    torch.exp(-k * td).max() - torch.exp(-k * td).min() + 1e-6)
                score = an_ * (1 - a) + rn_ * a

                bud = min(int(per_head_budget[layer_idx, kh]), nv)
                bud = max(10, bud)
                _, topk = torch.topk(score, bud)
                keep_idx = (topk + vs).sort()[0]

                # 直接索引 per-head KV
                old_k = cache.key_cache[layer_idx][kh]
                old_v = cache.value_cache[layer_idx][kh]
                cache.key_cache[layer_idx][kh] = torch.index_select(old_k, 0, keep_idx - vs)
                cache.value_cache[layer_idx][kh] = torch.index_select(old_v, 0, keep_idx - vs)

        # 同步 position_ids_cache (简化)
        for layer_idx in range(n_layers):
            if cache.key_cache[layer_idx][0] is not None:
                ml = max(cache.key_cache[layer_idx][h].shape[0] for h in range(n_kv))
                p = torch.arange(ml, device=model.device, dtype=torch.float32)
                model._position_ids_cache[layer_idx] = p.unsqueeze(0).expand(3, -1).clone()

        # 恢复
        model.kv_cache = saved_kv
        torch.cuda.empty_cache()

    model.predict_and_compress = dyn_pac
    model._sparsemm_cache = cache
    model._per_head_budget = per_head_budget

    print(f"[sparsemm_kv] Installed. head_scores={'yes' if head_scores is not None else 'no'}")
    return model
