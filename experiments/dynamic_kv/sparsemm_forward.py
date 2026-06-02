"""
完整 SparseMM 风格 attention forward for Qwen2.5-VL

替换 Qwen2_5_VLAttention.forward，自管 KV cache、mask、attention。
prefill 走 flash attn，decode 走 per-head 独立 KV。
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)


def make_sparsemm_forward(model, head_scores=None):
    """创建 attention forward + 缓存管理"""

    nl, nkv, nqpk = 28, 4, 7
    nq = 28

    # SparseMM budget
    if head_scores is not None:
        ks = head_scores.reshape(28, nkv, nqpk).mean(axis=2)
        ks = ks / ks.mean()
        base = model.kv_size
        pbudget = np.clip(np.round(base * ks), base//2, base*3//2).astype(int)
    else:
        ks = np.ones((nl, nkv))
        pbudget = np.full((nl, nkv), model.kv_size, dtype=int)

    # Per-head KV 存储 (SparseMM DynamicCacheSplitHeadFlatten 风格)
    # key_cache[layer] = [head0_K, head1_K, head2_K, head3_K] 每头独立 [tokens, dim]
    key_cache: list = [[None] * nkv for _ in range(nl)]
    value_cache: list = [[None] * nkv for _ in range(nl)]

    def PH_has(li):
        return key_cache[li][0] is not None

    def PH_get(li, h):
        return (key_cache[li][h], value_cache[li][h])

    def PH_set(li, h, k, v):
        key_cache[li][h] = k
        value_cache[li][h] = v

    def PH_clear():
        for li in range(nl):
            for h in range(nkv):
                key_cache[li][h] = None
                value_cache[li][h] = None

    def PH_items():
        for li in range(nl):
            if key_cache[li][0] is not None:
                for h in range(nkv):
                    yield li, h, key_cache[li][h], value_cache[li][h]

    # system prompt KV (深拷贝)
    init_kv = None

    # ---- predict_and_compress ----
    def pac():
        _rebuild_kv_cache()
        vs = model.visual_start_idx
        lq, gq = model.predict_next_question()
        lid = torch.as_tensor([model.processor.tokenizer(lq).input_ids], device=model.device, dtype=torch.int)
        gid = torch.as_tensor([model.processor.tokenizer(gq).input_ids], device=model.device, dtype=torch.int)

        # 截断 KV 到 30K 防 OOM
        saved = list(model.kv_cache)
        for i in range(nl):
            k, v = saved[i]
            if k.shape[2] > 30000:
                saved[i] = (k[:,:,-30000:], v[:,:,-30000:])
        model.kv_cache = tuple(saved)
        al = model._compute_attention_scores_manually(lid, model.kv_cache)
        ag = model._compute_attention_scores_manually(gid, model.kv_cache)
        mid = torch.as_tensor([model.processor.tokenizer(lq+"; "+gq).input_ids], device=model.device, dtype=torch.int)
        am = model._compute_attention_scores_manually(mid, model.kv_cache)
        _rebuild_kv_cache()

        for li in range(nl):
            if not PH_has(li): continue
            if key_cache[li][0].shape[0] <= model.kv_size: continue

            if li < 2: aw, ql, ab, kb = al[li], al[0].shape[2], 1.0, 20.0
            elif li >= 20: aw, ql, ab, kb = ag[li], ag[0].shape[2], 0.0, 0.0
            else:
                aw, ql = am[li], am[0].shape[2]
                p = (li - 2) / 18; ab, kb = 0.75 - 0.6 * p, 20 - 12 * p

            if aw.dim() < 4: continue
            vis = aw[0].mean(dim=1)[:, vs:-ql]
            nv = vis.shape[1]
            pos = torch.arange(nv, device=model.device, dtype=torch.float32)
            td = (nv - 1 - pos) / max(nv - 1, 1)

            for kh in range(nkv):
                kva = vis[kh*nqpk:(kh+1)*nqpk].mean(dim=0)
                hb = float(ks[li, kh] / ks.mean()) if ks.mean() > 0 else 1.0
                a = max(0.0, min(1.0, ab - (hb - 1) * 0.1))
                kv_val = max(0.0, kb - (hb - 1) * 3.0)
                an = (kva - kva.min()) / (kva.max() - kva.min() + 1e-6)
                rn = (torch.exp(-kv_val * td) - torch.exp(-kv_val * td).min()) / (
                    torch.exp(-kv_val * td).max() - torch.exp(-kv_val * td).min() + 1e-6)
                score = an * (1 - a) + rn * a
                bud = min(int(pbudget[li, kh]), nv); bud = max(10, bud)
                _, tk = torch.topk(score, bud)
                ki = (tk + vs).sort()[0]
                ok, ov = PH_get(li, kh)
                safe = (ki - vs).clamp(0, ok.shape[0] - 1)
                PH_set(li, kh, torch.index_select(ok, 0, safe),
                       torch.index_select(ov, 0, safe))

    model.predict_and_compress = pac

    # ---- KV cache 重建 ----
    def _rebuild_kv_cache():
        legacy = ()
        for li in range(nl):
            if not PH_has(li):
                continue
            ml = max(PH_get(li, h)[0].shape[0] for h in range(nkv))
            d = PH_get(li, 0)[0].shape[1]
            ku = torch.zeros(1, nkv, ml, d, device=PH_get(li, 0)[0].device, dtype=PH_get(li, 0)[0].dtype)
            vu = torch.zeros_like(ku)
            for h in range(nkv):
                k, v = PH_get(li, h); n = k.shape[0]
                ku[0, h, :n] = k; vu[0, h, :n] = v
            legacy += ((ku, vu),)
        model.kv_cache = legacy
        # 不重索引位置: PH 里 K/V 保持原始绝对位置（同 HybridKV pre-rotate）

    # ---- Attention Forward (完整 SparseMM 风格) ----
    def sparsemm_attn_forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, position_embeddings=None, **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()
        nh, nkg_val, hd = self.num_heads, self.num_key_value_groups, self.head_dim
        li = self.layer_idx

        # QKV
        q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)

        # M-RoPE：全局位置，不重索引 → 对所有头正确
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_scaling["mrope_section"])

        k_cache = k  # 原始 K (4头), DynamicCache 更新用
        # ---- Per-head: prefill + decode（仿 SparseMM: 先更新再读）----
        if PH_has(li):
            # decode: 先更新 PH + DynamicCache
            if use_cache and q_len == 1:
                for h in range(nkv):
                    PH_set(li, h, torch.cat([PH_get(li, h)[0], k[0, h].clone()], dim=0),
                           torch.cat([PH_get(li, h)[1], v[0, h].clone()], dim=0))
                pkv = kwargs.get('past_key_values', past_key_value)
                if pkv is not None and hasattr(pkv, 'update'):
                    pkv.update(k, v, li)

            if q_len == 1:
                # ==== HybridKV 风格 decode: flash_attn_varlen_func ====
                from flash_attn import flash_attn_varlen_func
                # 展平 per-head KV: [total_tokens, dim]
                k_flat = torch.cat([PH_get(li, h)[0] for h in range(nkv)], dim=0)
                v_flat = torch.cat([PH_get(li, h)[1] for h in range(nkv)], dim=0)
                # cu_seq_lens_k: 每 KV 头累计长度
                head_lens = torch.tensor([PH_get(li, h)[0].shape[0] for h in range(nkv)],
                                         device=k.device, dtype=torch.int32)
                cu_k = torch.cat([torch.zeros(1, device=k.device, dtype=torch.int32),
                                  head_lens.cumsum(0).to(torch.int32)])
                cu_q = (torch.arange(0, nkv+1, device=k.device, dtype=torch.int32) * nkg_val).to(torch.int32)
                max_q, max_k = nkg_val, head_lens.max().item()
                # Q: [nkv*nkg, 1, d] → 展平
                q_flat = q[0, :, 0, :].contiguous()  # [28, d]
                # 同步 DC（HF 需要跟踪位置）
                pkv = kwargs.get('past_key_values', past_key_value)
                if pkv is not None and hasattr(pkv, 'update'):
                    pkv.update(k_cache, v, li)

                attn_out = flash_attn_varlen_func(
                    q_flat.unsqueeze(1), k_flat.unsqueeze(1), v_flat.unsqueeze(1),
                    cu_q, cu_k, max_q, max_k, causal=True)
                attn_out = attn_out.view(1, nh, 1, hd)
            else:
                # ==== prefill: 对齐 HybridKV，统一 KV + SDPA（flashattn 后端）====
                pkv = kwargs.get('past_key_values', past_key_value)
                if pkv is not None:
                    try:
                        ok, ov = pkv[li]; k = torch.cat([ok, k], dim=2); v = torch.cat([ov, v], dim=2)
                    except: pass
                k = repeat_kv(k, nkg_val); v = repeat_kv(v, nkg_val)
                from flash_attn import flash_attn_func
                attn_out = flash_attn_func(
                    q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                    dropout_p=0.0, causal=True)
                attn_out = attn_out.transpose(1,2)
        else:
            # PH 不存在时: 拼旧 KV + 更新 DynamicCache
            pkv = kwargs.get('past_key_values', past_key_value)
            if pkv is not None and hasattr(pkv, 'get_seq_length') and pkv.get_seq_length(li) > 0:
                try:
                    ok, ov = pkv[li]; k = torch.cat([ok, k], dim=2); v = torch.cat([ov, v], dim=2)
                except: pass
            k_rep = repeat_kv(k, nkg_val); v_rep = repeat_kv(v, nkg_val)
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, scale=hd**-0.5)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_out = self.o_proj(attn_out)

        # 更新 DynamicCache（非 per-head 路径）
        # 更新 DynamicCache（所有路径都需要）
        if use_cache:
            pkv2 = kwargs.get('past_key_values', past_key_value)
            if pkv2 is not None and hasattr(pkv2, 'update'):
                pkv2.update(k_cache, v, li)

        return attn_out, None

    # 安装
    attn_cls = type(model.language_model.layers[0].self_attn)
    attn_cls.forward = sparsemm_attn_forward

    # ---- encode hooks ----
    oi = model.encode_init_prompt
    def ni():
        oi()
        nonlocal init_kv
        init_kv = tuple((model.kv_cache[l][0].clone(), model.kv_cache[l][1].clone()) for l in range(nl))
    model.encode_init_prompt = ni

    oe = model.encode_video_chunk
    def ne(vc):
        nonlocal init_kv
        # 拼 system prompt + 上一轮压缩 PH → 保留时序上下文
        if init_kv is not None:
            legacy = list((k.clone(), v.clone()) for k, v in init_kv)
            for li in range(nl):
                if PH_has(li):
                    ok, ov = PH_get(li, 0)  # [tokens_0, d]
                    ml = max(PH_get(li, h)[0].shape[0] for h in range(nkv))
                    d = PH_get(li, 0)[0].shape[1]
                    ku = torch.zeros(1, nkv, ml, d, device=PH_get(li,0)[0].device, dtype=PH_get(li,0)[0].dtype)
                    vu = torch.zeros_like(ku)
                    for h in range(nkv):
                        k, v = PH_get(li, h); n = k.shape[0]
                        ku[0, h, :n] = k; vu[0, h, :n] = v
                    ik, iv = init_kv[li]
                    ku = torch.cat([ik, ku], dim=2)
                    vu = torch.cat([iv, vu], dim=2)
                    legacy[li] = (ku, vu)
            model.kv_cache = tuple(legacy)
        else:
            model.kv_cache = None
        oe(vc)
        PH_clear()
        for li in range(nl):
            kl, vl = model.kv_cache[li]
            for h in range(nkv):
                PH_set(li, h, kl[0, h].clone(), vl[0, h].clone())
        pac()
        _rebuild_kv_cache()
    model.encode_video_chunk = ne

    oq = model.question_answering
    def nq(*args, **kwargs):
        _rebuild_kv_cache()
        # QA prefill 前设 _ph_active=False → 不加到 PH
        model._ph_active = True
        result = oq(*args, **kwargs)
        return result
    model.question_answering = nq

    model._per_head_bud = pbudget
    print(f"[sparsemm_forward] OK. budget=[{pbudget.min()},{pbudget.max()}]")
    return model
