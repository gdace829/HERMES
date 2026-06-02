"""
Per-Head KV Cache — HybridKV 对齐 Qwen2.5-VL
"""
import torch, numpy as np
from flash_attn import flash_attn_func, flash_attn_varlen_func
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb, repeat_kv)


def install(model, head_scores=None):
    nl, nkv, nqpk = 28, 4, 7
    if head_scores is not None:
        ks = head_scores.reshape(28, nkv, nqpk).mean(axis=2)
        ks = ks / ks.mean()
        pbudget = np.clip(np.round(model.kv_size * ks), model.kv_size//2, model.kv_size*3//2).astype(int)
    else:
        ks = np.ones((nl, nkv))
        pbudget = np.full((nl, nkv), model.kv_size, dtype=int)

    key_cache = [[None]*nkv for _ in range(nl)]
    value_cache = [[None]*nkv for _ in range(nl)]

    def has_ph(li): return key_cache[li][0] is not None

    # ---- 压缩 ----
    def compress():
        vs = model.visual_start_idx
        lq, gq = model.predict_next_question()
        lid = torch.as_tensor([model.processor.tokenizer(lq).input_ids], device=model.device, dtype=torch.int)
        gid = torch.as_tensor([model.processor.tokenizer(gq).input_ids], device=model.device, dtype=torch.int)
        saved = model.kv_cache  # 保存完整 DynamicCache
        _rebuild()              # 临时替换为 padded KV（仅用于伪查询注意力）
        al = model._compute_attention_scores_manually(lid, model.kv_cache)
        ag = model._compute_attention_scores_manually(gid, model.kv_cache)
        mq = lq+"; "+gq
        mid = torch.as_tensor([model.processor.tokenizer(mq).input_ids], device=model.device, dtype=torch.int)
        am = model._compute_attention_scores_manually(mid, model.kv_cache)
        for li in range(nl):
            if not has_ph(li) or key_cache[li][0].shape[0] <= model.kv_size: continue
            if li < 2: aw, ql, ab, kb = al[li], al[0].shape[2], 1.0, 20.0
            elif li >= 20: aw, ql, ab, kb = ag[li], ag[0].shape[2], 0.0, 0.0
            else: aw, ql = am[li], am[0].shape[2]; p = (li-2)/18; ab, kb = 0.75-0.6*p, 20-12*p
            if aw.dim() < 4: continue
            vis = aw[0].mean(dim=1)[:, vs:-ql]; nv = vis.shape[1]
            pos = torch.arange(nv, device=model.device, dtype=torch.float32); td = (nv-1-pos)/max(nv-1,1)
            # 层级预算: 层内 4 头等长 → 不 pad
            layer_score = torch.zeros(nv, device=model.device)
            for kh in range(nkv):
                kva = vis[kh*nqpk:(kh+1)*nqpk].mean(dim=0)
                hb = float(ks[li,kh]/ks.mean()) if ks.mean()>0 else 1.0
                a = max(0.0, min(1.0, ab-(hb-1)*0.1)); kv = max(0.0, kb-(hb-1)*3.0)
                an = (kva-kva.min())/(kva.max()-kva.min()+1e-6)
                rn = (torch.exp(-kv*td)-torch.exp(-kv*td).min())/(torch.exp(-kv*td).max()-torch.exp(-kv*td).min()+1e-6)
                layer_score += an*(1-a)+rn*a
            layer_bud = min(int(pbudget[li].mean()), nv); layer_bud = max(10, layer_bud)
            _, tk = torch.topk(layer_score, layer_bud)
            ki = (tk+vs).sort()[0]
            for kh in range(nkv):
                si = (ki-vs).clamp(0, key_cache[li][kh].shape[0]-1)
                key_cache[li][kh] = torch.index_select(key_cache[li][kh], 0, si)
                value_cache[li][kh] = torch.index_select(value_cache[li][kh], 0, si)
        model.kv_cache = saved  # 恢复完整 DC: prefill 永远用完整历史
        torch.cuda.empty_cache()

    def _rebuild():
        legacy = ()
        for li in range(nl):
            if not has_ph(li): continue
            ml = max(key_cache[li][h].shape[0] for h in range(nkv))
            d = key_cache[li][0].shape[1]
            ku = torch.zeros(1, nkv, ml, d, device=key_cache[li][0].device, dtype=key_cache[li][0].dtype)
            vu = torch.zeros_like(ku)
            for h in range(nkv): k,v = key_cache[li][h], value_cache[li][h]; n=k.shape[0]; ku[0,h,:n]=k; vu[0,h,:n]=v
            legacy += ((ku, vu),)
        model.kv_cache = legacy

    model.predict_and_compress = compress

    # ---- attention forward ----
    attn_cls = type(model.language_model.layers[0].self_attn)

    def fwd(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        nh, nkg, hd = self.num_heads, self.num_key_value_groups, self.head_dim
        li = self.layer_idx

        q = self.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1,2)
        k = self.k_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1,2)
        v = self.v_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1,2)

        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_scaling["mrope_section"])
        k_orig, v_orig = k.clone(), v.clone()

        # 统一 flash_attn_func: prefill + decode 全部等长
        pkv = kwargs.get('past_key_values', past_key_value)
        if pkv is not None:
            try: ok, ov = pkv[li]; k = torch.cat([ok, k], dim=2); v = torch.cat([ov, v], dim=2)
            except: pass
        k = repeat_kv(k, nkg); v = repeat_kv(v, nkg)
        attn_out = flash_attn_func(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
            dropout_p=0.0, causal=True).transpose(1,2)
        attn_out = attn_out.transpose(1,2).contiguous().reshape(bsz, q_len, -1)
        attn_out = self.o_proj(attn_out)

        # DC update + 同步 PH
        pkv = kwargs.get('past_key_values', past_key_value)
        if pkv is not None and use_cache:
            if hasattr(pkv, 'update'): pkv.update(k_orig, v_orig, li)
            if has_ph(li):
                for h in range(nkv):
                    key_cache[li][h] = torch.cat([key_cache[li][h], k_orig[0,h].clone()], dim=0)
                    value_cache[li][h] = torch.cat([value_cache[li][h], v_orig[0,h].clone()], dim=0)
        elif use_cache:
            for h in range(nkv):
                key_cache[li][h] = k_orig[0,h].clone() if key_cache[li][h] is None else torch.cat([key_cache[li][h], k_orig[0,h].clone()], dim=0)
                value_cache[li][h] = v_orig[0,h].clone() if value_cache[li][h] is None else torch.cat([value_cache[li][h], v_orig[0,h].clone()], dim=0)

        return attn_out, None

    attn_cls.forward = fwd

    # ---- hooks ----
    init_kv = None
    oi = model.encode_init_prompt
    def ni():
        nonlocal init_kv; oi()
        init_kv = tuple((model.kv_cache[l][0].clone(), model.kv_cache[l][1].clone()) for l in range(nl))
    model.encode_init_prompt = ni

    oe = model.encode_video_chunk
    def ne(vc):
        # 拼 system prompt + 上一轮压缩后的等长 KV
        if init_kv is not None:
            legacy = list((k.clone(),v.clone()) for k,v in init_kv)
            for li in range(nl):
                if has_ph(li):
                    ml = max(key_cache[li][h].shape[0] for h in range(nkv))
                    d = key_cache[li][0].shape[1]
                    ku = torch.zeros(1, nkv, ml, d, device=key_cache[li][0].device, dtype=key_cache[li][0].dtype)
                    vu = torch.zeros_like(ku)
                    for h in range(nkv): k,v = key_cache[li][h], value_cache[li][h]; n=k.shape[0]; ku[0,h,:n]=k; vu[0,h,:n]=v
                    ik, iv = legacy[li]
                    legacy[li] = (torch.cat([ik, ku], dim=2), torch.cat([iv, vu], dim=2))
            model.kv_cache = tuple(legacy)
        else:
            model.kv_cache = None
        oe(vc)
        for li in range(nl):
            kl, vl = model.kv_cache[li]
            for h in range(nkv):
                key_cache[li][h] = kl[0,h].clone(); value_cache[li][h] = vl[0,h].clone()
        compress()
    model.encode_video_chunk = ne

    oq = model.question_answering
    def nq(*a,**kw):
        _rebuild()  # QA 前重建 PH → padded legacy（含全部历史）
        return oq(*a,**kw)
    model.question_answering = nq

    print(f"[per_head_cache] OK budget=[{pbudget.min()},{pbudget.max()}]")
    return model
