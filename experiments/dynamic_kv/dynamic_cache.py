"""
SparseMM-style Per-Head Dynamic KV for HERMES (v2)

策略:
  - Prefill (chunk 编码): 正常 flash attention, 4 个 KV 头统一 kv_len
  - 压缩: 每个 KV 头独立打分选 token, 存为 per-head 展平格式
  - Decode (答题): eager attention, 每头只 attend 自己的 KV

不动 qwenvl_hermes.py, 纯外挂。
"""

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. Per-head keep indices 计算
# ============================================================

def compute_per_head_keep(model, head_scores=None):
    """用伪问题 attention 计算每层每个 KV 的 keep_indices"""
    device = model.device
    vs = model.visual_start_idx

    local_q, global_q = model.predict_next_question()
    local_ids = model.processor.tokenizer(local_q).input_ids
    local_ids = torch.as_tensor([local_ids], device=device, dtype=torch.int)
    global_ids = model.processor.tokenizer(global_q).input_ids
    global_ids = torch.as_tensor([global_ids], device=device, dtype=torch.int)

    al = model._compute_attention_scores_manually(local_ids, model.kv_cache)
    ag = model._compute_attention_scores_manually(global_ids, model.kv_cache)

    mixed_q = local_q + "; " + global_q
    mixed_ids = model.processor.tokenizer(mixed_q).input_ids
    mixed_ids = torch.as_tensor([mixed_ids], device=device, dtype=torch.int)
    am = model._compute_attention_scores_manually(mixed_ids, model.kv_cache)

    n_layers = len(al)
    n_kv, n_q_per_kv = 4, 7

    # 头分数 → KV 头级 budget
    if head_scores is not None:
        kv_sc = head_scores.reshape(28, n_kv, n_q_per_kv).mean(axis=2)
        total = kv_sc.sum()
        kv_norm = kv_sc / total if total > 0 else np.ones((n_layers, n_kv)) / (n_layers * n_kv)
        total_b = model.kv_size * n_layers * n_kv
        min_b = max(10, model.kv_size // 2)
        remain = total_b - min_b * n_layers * n_kv
        per_head_budget = np.round(kv_norm * remain + min_b).astype(int)
    else:
        per_head_budget = np.full((n_layers, n_kv), model.kv_size, dtype=int)
        kv_norm = np.zeros((n_layers, n_kv))

    per_head_keep = []

    for layer_idx in range(n_layers):
        if layer_idx < model.short_term_threshold:
            aw = al[layer_idx]; ql = al[0].shape[2]; a_base, k_base = 1.0, 20.0
        elif layer_idx >= model.long_term_threshold:
            aw = ag[layer_idx]; ql = ag[0].shape[2]; a_base, k_base = 0.0, 0.0
        else:
            aw = am[layer_idx]; ql = am[0].shape[2]
            p = (layer_idx - model.short_term_threshold) / (model.long_term_threshold - model.short_term_threshold)
            a_base, k_base = 0.75 - 0.6 * p, 20.0 - 12.0 * p

        if aw.dim() < 4:
            per_head_keep.append(None); continue

        vis = aw[0].mean(dim=1)[:, vs:-ql]
        nv = vis.shape[1]
        pos = torch.arange(nv, device=device, dtype=torch.float32)
        td = (nv - 1 - pos) / max(nv - 1, 1)

        lk = []
        for kh in range(n_kv):
            qs, qe = kh * n_q_per_kv, (kh + 1) * n_q_per_kv
            kv_a = vis[qs:qe].mean(dim=0)

            hb = float(kv_norm[layer_idx, kh])
            a = max(0.0, min(1.0, a_base - hb * 0.3))
            k = max(0.0, k_base - hb * 10.0)

            an_ = (kv_a - kv_a.min()) / (kv_a.max() - kv_a.min() + 1e-6)
            rn_ = (torch.exp(-k * td) - torch.exp(-k * td).min()) / (
                torch.exp(-k * td).max() - torch.exp(-k * td).min() + 1e-6)
            score = an_ * (1 - a) + rn_ * a

            bud = min(int(per_head_budget[layer_idx, kh]), nv)
            bud = max(10, bud)
            _, topk = torch.topk(score, bud)
            lk.append(topk + vs)

        per_head_keep.append(lk)

    return per_head_keep


# ============================================================
# 2. 按 per-head keep 裁剪 KV，展平存储
# ============================================================

def apply_per_head_prune(model, per_head_keep):
    """每 KV 头独立裁剪 → 展平存为 list of [tokens_per_head, dim]"""
    device = model.device
    per_head_kv = []  # [layer][head] = (k, v)

    for layer_idx in range(model.num_layers):
        keeps = per_head_keep[layer_idx]
        if keeps is None:
            per_head_kv.append(None); continue

        k_layer, v_layer = model.kv_cache[layer_idx]  # [1, 4, seq, d]
        d = k_layer.shape[3]
        hk = []
        for h in range(min(4, len(keeps))):
            ki = keeps[h]
            if not isinstance(ki, torch.Tensor):
                ki = torch.as_tensor(ki, device=device)
            kh = torch.index_select(k_layer[:, h], dim=1, index=ki)  # [1, n, d]
            vh = torch.index_select(v_layer[:, h], dim=1, index=ki)
            hk.append((kh.squeeze(0).contiguous(), vh.squeeze(0).contiguous()))
        per_head_kv.append(hk)

    return per_head_kv


# ============================================================
# 3. Decode 阶段 per-head attention（替换 question_answering 的 attention）
# ============================================================

def per_head_question_answering(model, input_text, max_new_tokens=128):
    """重写 question_answering：decode 阶段每头只 attend 自己的 KV"""
    from inference.qwenvl_hermes import get_qwen2_5_vl_position_ids
    import time
    import re

    device = model.device
    stop_ids = [model.processor.tokenizer.eos_token_id]
    output_ids = []
    start = time.perf_counter()

    prompt = input_text['prompt']
    input_ids = model.processor.tokenizer(prompt).input_ids
    input_ids = torch.as_tensor([input_ids], device=device)

    # 从 per_head KV 重建统一格式（pad 到 max_len）
    # 这一步确保 prefill 阶段正常
    per_head = model._per_head_kv
    if per_head is not None:
        # 重建 padded KV cache 给 prefill
        rebuilt = []
        for layer_idx in range(model.num_layers):
            if per_head[layer_idx] is None:
                rebuilt.append(model.kv_cache[layer_idx]); continue
            max_len = max(k.shape[0] for k, v in per_head[layer_idx])
            d = per_head[layer_idx][0][0].shape[1]
            ks = torch.zeros(1, 4, max_len, d, device=device, dtype=per_head[layer_idx][0][0].dtype)
            vs = torch.zeros(1, 4, max_len, d, device=device, dtype=per_head[layer_idx][0][0].dtype)
            for h, (k, v) in enumerate(per_head[layer_idx]):
                n = k.shape[0]
                ks[0, h, :n] = k
                vs[0, h, :n] = v
            rebuilt.append((ks, vs))
        model.kv_cache = rebuilt

        # 位置 ID 重建
        for layer_idx in range(model.num_layers):
            seq = model.kv_cache[layer_idx][0].shape[2]
            p = torch.arange(seq, device=device, dtype=torch.float32)
            model._position_ids_cache[layer_idx] = p.unsqueeze(0).expand(3, -1).clone()

    # ======== prefill (同原版) ========
    model._ensure_dynamic_cache()
    past_lens = model._get_cache_seq_len_per_layer()
    global_off = model._get_next_global_offset_per_layer()

    inputs_embeds = model.get_input_embeddings()(input_ids)
    q_len = inputs_embeds.shape[1]
    batch = inputs_embeds.shape[0]

    model._layer_position_ids.clear()
    for layer_idx in range(model.num_layers):
        p3d = model._build_position_ids_3d_for_text(global_off[layer_idx], q_len, batch)
        model._layer_position_ids[layer_idx] = p3d
    pos_3d = model._build_position_ids_3d_for_text(global_off[0], q_len, batch)

    out = model.language_model(
        inputs_embeds=inputs_embeds, use_cache=True,
        past_key_values=model.kv_cache, position_ids=pos_3d)
    pkv = out.past_key_values
    logits = model.lm_head(out.last_hidden_state)

    for layer_idx in range(model.num_layers):
        off = global_off[layer_idx]
        model._append_position_ids_layer(layer_idx, [off, off, off], q_len)
    model._layer_position_ids.clear()

    # ======== decode (per-head attention) ========
    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :]
        _, indices = torch.topk(last_logits, 1)
        token = int(indices[0])
        output_ids.append(token)

        if step == 0:
            ttft = time.perf_counter() - start
            print(f"TTFT: {ttft} seconds")

        if token in stop_ids:
            break

        curr_off = model._get_next_global_offset_per_layer()
        model._layer_position_ids.clear()
        for layer_idx in range(model.num_layers):
            model._layer_position_ids[layer_idx] = model._build_position_ids_3d_for_text(
                curr_off[layer_idx], 1, 1)
        pos_3d = model._build_position_ids_3d_for_text(curr_off[0], 1, 1)

        out = model.language_model(
            input_ids=torch.as_tensor([[token]], device=device),
            use_cache=True, past_key_values=pkv, position_ids=pos_3d)
        logits = model.lm_head(out.last_hidden_state)
        pkv = out.past_key_values

        for layer_idx in range(model.num_layers):
            off = curr_off[layer_idx]
            model._append_position_ids_layer(layer_idx, [off, off, off], 1)
        model._layer_position_ids.clear()

    output = model.processor.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # 清理
    model._truncate_kv_cache(past_lens)
    for layer_idx in range(model.num_layers):
        if model._position_ids_cache[layer_idx] is not None and \
           model._position_ids_cache[layer_idx].shape[1] > past_lens[layer_idx]:
            model._position_ids_cache[layer_idx] = model._position_ids_cache[layer_idx][:, :past_lens[layer_idx]].contiguous()

    # 恢复 per-head KV
    if per_head is not None:
        model._per_head_kv = per_head

    if not getattr(input_text, 'pseudo_forward', False):
        question = input_text.get('question', '')
        if not hasattr(model, 'conv_history'):
            model.conv_history = []
        model.conv_history.append((question, output, None))
        print(f"Answering Cache lengths: min={min(past_lens)}, max={max(past_lens)}")

    torch.cuda.empty_cache()
    return output


# ============================================================
# 4. 安装
# ============================================================

def install_dynamic_kv(model, head_scores=None):
    model._head_scores = head_scores
    model._per_head_kv = None

    # 替换 predict_and_compress
    def dyn_pac():
        if model.compress_mode == "streamingvlm":
            return model._sliding_window_compress()

        phk = compute_per_head_keep(model, head_scores)
        current_len = model.kv_cache[0][0].shape[2]
        if current_len > model.kv_size:
            print(f"DynamicKV: {current_len} -> per-head prune (budget={model.kv_size})")
            model._per_head_kv = apply_per_head_prune(model, phk)

            # 对 per_head KV pad 回统一格式（后续 prefill 正常用）
            if model._per_head_kv is not None:
                rebuilt = []
                for li in range(model.num_layers):
                    if model._per_head_kv[li] is None:
                        rebuilt.append(model.kv_cache[li]); continue
                    ml = max(k.shape[0] for k, v in model._per_head_kv[li])
                    d = model._per_head_kv[li][0][0].shape[1]
                    ks = torch.zeros(1, 4, ml, d, device=model.device, dtype=model._per_head_kv[li][0][0].dtype)
                    vs = torch.zeros(1, 4, ml, d, device=model.device, dtype=model._per_head_kv[li][0][0].dtype)
                    for h, (k, v) in enumerate(model._per_head_kv[li]):
                        n = k.shape[0]
                        ks[0, h, :n] = k; vs[0, h, :n] = v
                    rebuilt.append((ks, vs))
                model.kv_cache = rebuilt

                for li in range(model.num_layers):
                    s = model.kv_cache[li][0].shape[2]
                    p = torch.arange(s, device=model.device, dtype=torch.float32)
                    model._position_ids_cache[li] = p.unsqueeze(0).expand(3, -1).clone()

    model.predict_and_compress = dyn_pac

    print(f"[dynamic_kv] Installed. head_scores={'yes' if head_scores is not None else 'no'}")
    return model
