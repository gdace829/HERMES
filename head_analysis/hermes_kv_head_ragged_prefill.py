"""
Experimental physical per-KV-head ragged prefill/chunk path for Qwen2.5-VL.

This module extends the decode-only HeadRaggedCache prototype to video/text
prefill. Once installed, HERMES keeps the main KV cache in a flat ragged layout:

    flat_k / flat_v / head_lens / cu_klen

For q_len > 1, every KV head attends to its own historical ragged segment plus
the current dense chunk K/V. Compression prunes each KV head independently and
does not form a dense union cache.
"""

import re
import time
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table
from head_analysis.hermes_kv_head_ragged import (
    HeadRaggedCache,
    _ragged_language_decode_step,
    flash_attn_varlen_func,
)


def _is_ragged_cache(cache):
    return isinstance(cache, HeadRaggedCache)


def _position_ids_2d(position_ids):
    if position_ids.dim() == 3:
        return position_ids[:, 0, :]
    return position_ids


def _attention_shape(model, attn=None):
    config = getattr(model.language_model, "config", None)
    num_query_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if attn is not None:
        num_query_heads = getattr(attn.config, "num_attention_heads", num_query_heads)
        num_kv_heads = getattr(attn.config, "num_key_value_heads", num_kv_heads)
    num_query_heads = int(num_query_heads or 28)
    num_kv_heads = int(num_kv_heads or 4)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )
    return num_query_heads, num_kv_heads, num_query_heads // num_kv_heads


def _build_total_kv(layer_cache, key_states, value_states, num_kv_heads):
    k_parts = []
    v_parts = []
    total_lens = []
    q_len = key_states.shape[2]

    for kv_head in range(num_kv_heads):
        hist_k, hist_v = layer_cache.get_segment(kv_head)
        curr_k = key_states[0, kv_head]
        curr_v = value_states[0, kv_head]
        k_parts.append(torch.cat([hist_k, curr_k], dim=0))
        v_parts.append(torch.cat([hist_v, curr_v], dim=0))
        total_lens.append(hist_k.shape[0] + q_len)

    flat_k = torch.cat(k_parts, dim=0).view(-1, 1, key_states.shape[-1]).contiguous()
    flat_v = torch.cat(v_parts, dim=0).view(-1, 1, value_states.shape[-1]).contiguous()
    total_lens = torch.tensor(total_lens, device=key_states.device, dtype=torch.int32)
    cu_k = torch.cat([
        torch.zeros(1, device=key_states.device, dtype=torch.int32),
        torch.cumsum(total_lens, dim=0, dtype=torch.int32),
    ]).to(dtype=torch.int32)
    return flat_k, flat_v, total_lens, cu_k


def _ragged_attention_flash_chunk(query_states, key_states, value_states,
                                  layer_cache, num_kv_heads, group_size):
    if flash_attn_varlen_func is None:
        return None
    if query_states.device.type != "cuda":
        return None
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        return None

    _, _, q_len, head_dim = query_states.shape
    q_flat = (
        query_states[0]
        .view(num_kv_heads, group_size, q_len, head_dim)
        .permute(0, 2, 1, 3)
        .reshape(num_kv_heads * q_len, group_size, head_dim)
        .contiguous()
    )
    k_flat, v_flat, total_lens, cu_k = _build_total_kv(
        layer_cache, key_states, value_states, num_kv_heads
    )
    cu_q = torch.arange(
        0,
        (num_kv_heads + 1) * q_len,
        step=q_len,
        device=query_states.device,
        dtype=torch.int32,
    )

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        cu_k,
        max_seqlen_q=int(q_len),
        max_seqlen_k=int(total_lens.max().item()),
        causal=True,
    )
    out = (
        out.view(num_kv_heads, q_len, group_size, head_dim)
        .permute(0, 2, 1, 3)
        .reshape(num_kv_heads * group_size, q_len, head_dim)
        .transpose(0, 1)
        .reshape(1, q_len, num_kv_heads * group_size * head_dim)
    )
    return out


def _ragged_attention_manual_chunk(query_states, key_states, value_states,
                                   layer_cache, num_kv_heads, group_size,
                                   scaling, collect_scores=False):
    _, _, q_len, head_dim = query_states.shape
    out_groups = []
    score_vectors = []

    for kv_head in range(num_kv_heads):
        hist_k, hist_v = layer_cache.get_segment(kv_head)
        curr_k = key_states[0, kv_head]
        curr_v = value_states[0, kv_head]
        total_k = torch.cat([hist_k, curr_k], dim=0)
        total_v = torch.cat([hist_v, curr_v], dim=0)
        hist_len = hist_k.shape[0]
        total_len = total_k.shape[0]

        q_group = query_states[0, kv_head * group_size:(kv_head + 1) * group_size]
        scores = torch.einsum("gqd,kd->gqk", q_group.float(), total_k.float()) * scaling

        key_idx = torch.arange(total_len, device=query_states.device)
        query_idx = torch.arange(q_len, device=query_states.device)
        causal = key_idx.view(1, -1) <= (hist_len + query_idx).view(-1, 1)
        scores = scores.masked_fill(~causal.view(1, q_len, total_len), -torch.finfo(scores.dtype).max)

        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        if collect_scores:
            score_vectors.append(probs[:, :, :hist_len].mean(dim=(0, 1)).detach())
        probs = probs.to(q_group.dtype)
        out_groups.append(torch.einsum("gqk,kd->gqd", probs, total_v))

    out = (
        torch.cat(out_groups, dim=0)
        .transpose(0, 1)
        .reshape(1, q_len, num_kv_heads * group_size * head_dim)
    )
    if collect_scores:
        return out, score_vectors
    return out, None


def _ragged_chunk_self_attn_forward(model, layer_idx, attn, hidden_states,
                                    ragged_cache, position_embeddings,
                                    position_ids, append=True,
                                    collect_scores=False):
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1:
        raise ValueError("Ragged prefill currently supports batch_size=1 only")

    num_query_heads, num_kv_heads, group_size = _attention_shape(model, attn)
    head_dim = attn.head_dim

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_query_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    target_dtype = attn.q_proj.weight.dtype
    if query_states.dtype != target_dtype:
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    cos, sin = position_embeddings
    mrope_section = getattr(model, "_mrope_section", attn.rope_scaling["mrope_section"])
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        mrope_section,
    )

    layer_cache = ragged_cache.layers[layer_idx]
    if collect_scores:
        scaling = getattr(attn, "scaling", head_dim ** -0.5)
        attn_output, layer_scores = _ragged_attention_manual_chunk(
            query_states,
            key_states,
            value_states,
            layer_cache,
            num_kv_heads,
            group_size,
            scaling,
            collect_scores=True,
        )
    else:
        attn_output = _ragged_attention_flash_chunk(
            query_states, key_states, value_states, layer_cache, num_kv_heads, group_size
        )
        if attn_output is None:
            scaling = getattr(attn, "scaling", head_dim ** -0.5)
            attn_output, _ = _ragged_attention_manual_chunk(
                query_states,
                key_states,
                value_states,
                layer_cache,
                num_kv_heads,
                group_size,
                scaling,
                collect_scores=False,
            )
        layer_scores = None

    if append:
        ragged_cache.append_chunk(
            layer_idx,
            key_states,
            value_states,
            pos=_position_ids_2d(position_ids),
        )

    return attn.o_proj(attn_output.contiguous()), layer_scores


def _ragged_language_chunk_forward(model, inputs_embeds, ragged_cache,
                                   position_ids_by_layer, append=True,
                                   collect_scores=False):
    hidden_states = inputs_embeds
    all_scores = []

    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)
        position_ids = position_ids_by_layer[layer_idx]
        position_embeddings = model.language_model.rotary_emb(hidden_states_norm, position_ids)
        attn_output, layer_scores = _ragged_chunk_self_attn_forward(
            model,
            layer_idx,
            decoder_layer.self_attn,
            hidden_states_norm,
            ragged_cache,
            position_embeddings,
            position_ids,
            append=append,
            collect_scores=collect_scores,
        )
        if collect_scores:
            all_scores.append(layer_scores)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = model.language_model.norm(hidden_states)
    return hidden_states, all_scores


def _build_text_position_ids(model, offsets, q_len, batch=1):
    return [
        model._build_position_ids_3d_for_text(offsets[layer_idx], q_len, batch)
        for layer_idx in range(model.num_layers)
    ]


def _ragged_query_scores(model, input_ids, offsets):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    position_ids_by_layer = _build_text_position_ids(
        model, offsets, inputs_embeds.shape[1], inputs_embeds.shape[0]
    )
    _, scores = _ragged_language_chunk_forward(
        model,
        inputs_embeds,
        model.kv_cache,
        position_ids_by_layer,
        append=False,
        collect_scores=True,
    )
    return scores


def _ragged_budget_table(model, num_keep):
    config = getattr(model, "_kv_head_budget_config", {})
    num_query_heads, num_kv_heads, _ = _attention_shape(model)
    num_layers = int(getattr(model, "num_layers", len(model.kv_cache.layers)))
    scores = getattr(model, "_kv_head_budget_scores", None)
    if scores is None:
        scores = np.zeros((num_layers, num_query_heads), dtype=np.float64)

    return build_kv_head_budget_table(
        scores,
        num_keep,
        num_layers=num_layers,
        num_query_heads=int(config.get("num_query_heads", num_query_heads)),
        num_kv_heads=int(config.get("num_kv_heads", num_kv_heads)),
        strength=float(config.get("strength", 0.75)),
        min_ratio=float(config.get("min_ratio", 0.75)),
        max_ratio=float(config.get("max_ratio", 1.25)),
        scheme=config.get("budget_scheme", "relative"),
        sparsemm_ratio=float(config.get("sparsemm_ratio", 0.1)),
        sparsemm_window_size=int(config.get("sparsemm_window_size", 32)),
    )


def _snapshot_head_lens_as_ints(ragged_cache):
    return [
        [int(x) for x in layer_lens.detach().cpu().tolist()]
        for layer_lens in ragged_cache.snapshot_head_lens()
    ]


def _select_ragged_keep_indices(model, local_scores, global_scores, mixed_scores, num_keep):
    device = model.device
    visual_start_idx = int(model.visual_start_idx)
    num_layers = len(model.kv_cache.layers)
    num_kv_heads = int(model.kv_cache.layers[0].head_lens.numel())
    budget_table = _ragged_budget_table(model, num_keep)
    config = getattr(model, "_kv_head_budget_config", {})
    scheme = str(config.get("budget_scheme", "relative") or "relative").lower()
    protected_recent_window = (
        max(0, int(config.get("sparsemm_window_size", 0)))
        if scheme.startswith("sparsemm")
        else 0
    )

    keep_indices = []
    visible_lengths = []

    for layer_idx in range(num_layers):
        if layer_idx < model.short_term_threshold:
            layer_scores = local_scores[layer_idx]
            recency_alpha = 1.0
            k_decay = 20.0
        elif layer_idx >= model.long_term_threshold:
            layer_scores = global_scores[layer_idx]
            recency_alpha = 0.0
            k_decay = 0.0
        else:
            layer_scores = mixed_scores[layer_idx]
            progress = (
                (layer_idx - model.short_term_threshold)
                / (model.long_term_threshold - model.short_term_threshold)
            )
            recency_alpha = 0.75 - 0.6 * progress
            k_decay = 20.0 - 12.0 * progress

        layer_keep = []
        for kv_head in range(num_kv_heads):
            head_len = int(model.kv_cache.layers[layer_idx].head_lens[kv_head].item())
            text_keep = min(visual_start_idx, head_len)
            score = layer_scores[kv_head].to(device=device)
            if score.numel() < head_len:
                pad = torch.zeros(head_len - score.numel(), device=device, dtype=score.dtype)
                score = torch.cat([score, pad], dim=0)
            score = score[:head_len]

            num_visual_tokens = max(0, head_len - text_keep)
            if num_visual_tokens <= 0:
                keep = torch.arange(head_len, device=device, dtype=torch.long)
                layer_keep.append(keep)
                visible_lengths.append(int(keep.numel()))
                continue

            visual_score = score[text_keep:]
            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            recency = torch.exp(-k_decay * time_distances)
            recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)
            attn_norm = (visual_score - visual_score.min()) / (
                visual_score.max() - visual_score.min() + 1e-6
            )
            combined = attn_norm * (1.0 - recency_alpha) + recency * recency_alpha

            budget = min(int(budget_table[layer_idx, kv_head]), num_visual_tokens)
            if budget <= 0:
                selected_visual = torch.empty(0, device=device, dtype=torch.long)
            else:
                recent_k = min(protected_recent_window, budget, num_visual_tokens)
                history_len = max(0, num_visual_tokens - recent_k)
                history_k = min(max(budget - recent_k, 0), history_len)

                pieces = []
                if history_k > 0:
                    pieces.append(torch.topk(combined[:history_len], history_k, sorted=False)[1])
                if recent_k > 0:
                    pieces.append(
                        torch.arange(
                            num_visual_tokens - recent_k,
                            num_visual_tokens,
                            device=device,
                            dtype=torch.long,
                        )
                    )
                selected_visual = torch.unique(torch.cat(pieces), sorted=True) + text_keep

            keep = torch.cat([
                torch.arange(text_keep, device=device, dtype=torch.long),
                selected_visual,
            ])
            keep = torch.unique(keep, sorted=True)
            layer_keep.append(keep)
            visible_lengths.append(int(keep.numel()))
        keep_indices.append(layer_keep)

    return keep_indices, budget_table, visible_lengths


def _sync_position_cache_from_ragged(model):
    if not _is_ragged_cache(model.kv_cache):
        return
    for layer_idx, layer in enumerate(model.kv_cache.layers):
        if layer.flat_pos is None:
            model._position_ids_cache[layer_idx] = None
        else:
            model._position_ids_cache[layer_idx] = layer.flat_pos


def _ragged_pseudo_forward(model, local_question=None, global_question=None):
    device = model.device
    if local_question is None:
        local_question = "What is happening in the video?"
    if global_question is None:
        global_question = "What is the main topic of the video?"

    offsets = model._get_next_global_offset_per_layer()

    local_input_ids = model.processor.tokenizer(local_question).input_ids
    local_input_ids = torch.as_tensor([local_input_ids], device=device, dtype=torch.long)
    local_scores = _ragged_query_scores(model, local_input_ids, offsets)

    global_input_ids = model.processor.tokenizer(global_question).input_ids
    global_input_ids = torch.as_tensor([global_input_ids], device=device, dtype=torch.long)
    global_scores = _ragged_query_scores(model, global_input_ids, offsets)

    mixed_question = local_question + "; " + global_question
    mixed_input_ids = model.processor.tokenizer(mixed_question).input_ids
    mixed_input_ids = torch.as_tensor([mixed_input_ids], device=device, dtype=torch.long)
    mixed_scores = _ragged_query_scores(model, mixed_input_ids, offsets)

    print(f"GPU memory usage: {model.get_gpu_memory_usage_gb()} GB")
    current_len = max(model.kv_cache.max_lens_per_layer())
    if current_len <= model.kv_size:
        return

    if getattr(model, "_rolekv_ragged_config", None):
        from head_analysis.rolekv_ragged_policy import select_rolekv_ragged_keep_indices

        budget_table = _ragged_budget_table(model, model.kv_size)
        keep_indices, visible_lengths = select_rolekv_ragged_keep_indices(
            model,
            local_scores,
            global_scores,
            mixed_scores,
            num_keep=model.kv_size,
            budget_table=budget_table,
        )
        selector_name = "RoleKV-ragged"
    else:
        keep_indices, budget_table, visible_lengths = _select_ragged_keep_indices(
            model,
            local_scores,
            global_scores,
            mixed_scores,
            num_keep=model.kv_size,
        )
        selector_name = "default-ragged"

    print(
        f"Applying ragged KV-Cache compression via {selector_name} "
        f"due to max head_len > {model.kv_size}"
    )
    model.kv_cache.prune_per_head(keep_indices)
    _sync_position_cache_from_ragged(model)
    rmin, rmax, rmean = model.kv_cache.stats()
    print(
        "[kv_head_ragged_prefill] prune stats: "
        f"kv_budget=[{budget_table.min()}, {budget_table.max()}], "
        f"visible=[{min(visible_lengths)}, {max(visible_lengths)}], "
        f"head_lens=[{rmin}, {rmax}], mean={rmean:.1f}"
    )
    role_stats = getattr(model, "_rolekv_ragged_last_stats", None)
    if role_stats:
        print(
            "[RoleKV-ragged] selection stats: "
            f"roles={role_stats.get('selected_kv_roles')}, "
            f"quota={role_stats.get('quota_stats')}, "
            f"mode={role_stats.get('mode')}, "
            f"quota_ratio={role_stats.get('quota_ratio')}"
        )


def apply_kv_head_ragged_prefill(model):
    """Install physical per-KV-head ragged cache for prefill/chunk/decode."""
    original_encode_init_prompt = model.encode_init_prompt
    original_encode_video_chunk = model.encode_video_chunk
    original_predict_and_compress = model.predict_and_compress
    original_question_answering = model.question_answering
    original_get_cache_seq_len_per_layer = model._get_cache_seq_len_per_layer
    original_get_next_global_offset_per_layer = model._get_next_global_offset_per_layer
    original_truncate_kv_cache = model._truncate_kv_cache

    @torch.inference_mode()
    def get_cache_seq_len_per_layer_ragged(self):
        if _is_ragged_cache(self.kv_cache):
            return self.kv_cache.max_lens_per_layer()
        return original_get_cache_seq_len_per_layer()

    def get_next_global_offset_per_layer_ragged(self):
        if _is_ragged_cache(self.kv_cache):
            offsets = []
            for layer_idx, layer in enumerate(self.kv_cache.layers):
                max_pos = None
                if layer.flat_pos is not None and layer.flat_pos.numel() > 0:
                    max_pos = layer.flat_pos.max().item()
                pos_cache = self._position_ids_cache[layer_idx]
                if pos_cache is not None and pos_cache.numel() > 0:
                    cache_max = pos_cache.max().item()
                    max_pos = cache_max if max_pos is None else max(max_pos, cache_max)
                offsets.append(0 if max_pos is None else int(max_pos) + 1)
            return offsets
        return original_get_next_global_offset_per_layer()

    def truncate_kv_cache_ragged(self, target_lengths):
        if _is_ragged_cache(self.kv_cache):
            self.kv_cache.truncate(target_lengths)
            _sync_position_cache_from_ragged(self)
            return
        return original_truncate_kv_cache(target_lengths)

    @torch.inference_mode()
    def encode_init_prompt_ragged(self):
        original_encode_init_prompt()
        self.kv_cache = HeadRaggedCache.from_dense_cache(self, self.kv_cache)
        _sync_position_cache_from_ragged(self)
        rmin, rmax, rmean = self.kv_cache.stats()
        print(
            "[kv_head_ragged_prefill] init cache converted to ragged: "
            f"min={rmin}, max={rmax}, mean={rmean:.1f}"
        )

    @torch.inference_mode()
    def encode_video_chunk_ragged(self, video_chunk):
        if not _is_ragged_cache(self.kv_cache):
            original_encode_video_chunk(video_chunk)
            return
        if video_chunk is None or (hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0):
            return

        from inference.qwenvl_hermes import get_qwen2_5_vl_position_ids

        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        video_input = self.processor(text=[""], videos=video_chunk, return_tensors="pt").to(
            self.device, self.dtype
        )
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        video_features = self.get_video_features(pixel_values_videos, video_grid_thw)[0].unsqueeze(0)

        offsets = self._get_next_global_offset_per_layer()
        q_len = video_features.shape[1]
        batch = video_features.shape[0]
        base_offset = offsets[0]
        grid_pos_ids = get_qwen2_5_vl_position_ids(
            video_grid_thw[0].tolist(),
            q_len,
            offset=base_offset,
            vision_config=self.config.vision_config,
            sample_fps=self.sample_fps,
        ).to(self.device)

        position_ids_by_layer = []
        for layer_idx in range(self.num_layers):
            layer_offset = offsets[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            position_ids_by_layer.append(self._build_position_ids_3d_for_vision(current_layer_pos, batch))

        pre_head_lens = _snapshot_head_lens_as_ints(self.kv_cache)
        _ragged_language_chunk_forward(
            self,
            video_features,
            self.kv_cache,
            position_ids_by_layer,
            append=True,
            collect_scores=False,
        )
        post_head_lens = _snapshot_head_lens_as_ints(self.kv_cache)
        self._rolekv_last_pre_lens_ragged = pre_head_lens
        self._rolekv_last_post_lens_ragged = post_head_lens
        _sync_position_cache_from_ragged(self)
        self.last_encoded_frames = video_chunk.shape[0]
        self.total_processed_frames += video_chunk.shape[0]
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict_and_compress_ragged(self):
        if not _is_ragged_cache(self.kv_cache):
            return original_predict_and_compress()
        if self.compress_mode == "streamingvlm":
            raise NotImplementedError("ragged prefill currently supports HERMES compression only")
        local_question, global_question = self.predict_next_question()
        _ragged_pseudo_forward(self, local_question, global_question)

    @torch.inference_mode()
    def question_answering_ragged_prefill(self, input_text, max_new_tokens=128,
                                          temperature=0, repetition_penalty=1.1,
                                          pseudo_forward=False):
        if pseudo_forward or not _is_ragged_cache(self.kv_cache):
            return original_question_answering(
                input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pseudo_forward=pseudo_forward,
            )

        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []
        start_time = time.perf_counter()

        prompt = input_text["prompt"]
        input_ids = self.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=device, dtype=torch.long)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]

        past_lens_prefill = self.kv_cache.snapshot_head_lens()
        offsets = self._get_next_global_offset_per_layer()
        position_ids_by_layer = _build_text_position_ids(self, offsets, q_len_prefill, batch)

        hidden_states, _ = _ragged_language_chunk_forward(
            self,
            inputs_embeds,
            self.kv_cache,
            position_ids_by_layer,
            append=True,
            collect_scores=False,
        )
        logits = self.lm_head(hidden_states)

        for step in range(max_new_tokens):
            last_token_logits = logits[0, -1, :]
            if repetition_penalty != 1.0 and len(output_ids) > 0:
                for token_id in set(output_ids):
                    if last_token_logits[token_id] < 0:
                        last_token_logits[token_id] *= repetition_penalty
                    else:
                        last_token_logits[token_id] /= repetition_penalty

            if temperature == 0.0:
                _, indices = torch.topk(last_token_logits, 1)
                token = int(indices[0])
            else:
                scaled_logits = last_token_logits / temperature
                scaled_logits = torch.nan_to_num(
                    scaled_logits,
                    nan=-float("inf"),
                    posinf=float("inf"),
                    neginf=-float("inf"),
                )
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    _, indices = torch.topk(last_token_logits, 1)
                    token = int(indices[0])

            output_ids.append(token)
            if step == 0:
                end_time = time.perf_counter()
                print(f"TTFT: {end_time - start_time} seconds")
            if token in stop_token_ids:
                break

            position_ids_step = []
            for layer_idx in range(self.num_layers):
                offset = offsets[layer_idx] + q_len_prefill + step
                position_ids_step.append(self._build_position_ids_3d_for_text(offset, 1, 1))

            hidden_states = _ragged_language_decode_step(
                self,
                torch.as_tensor([[token]], device=device),
                self.kv_cache,
                position_ids_step,
            )
            logits = self.lm_head(hidden_states)

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        current_question = input_text["question"]
        current_options = None
        formatted_question = input_text.get("formatted_question", None)
        if formatted_question:
            option_matches = re.findall(
                r"\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)",
                formatted_question,
                re.DOTALL,
            )
            if option_matches:
                current_options = [opt.strip() for opt in option_matches]
        self.conv_history.append((current_question, output, current_options))
        logger.info(f"Saved conversation to history. Total conversations: {len(self.conv_history)}")

        self.kv_cache.truncate(past_lens_prefill)
        _sync_position_cache_from_ragged(self)
        new_lens = self._get_cache_seq_len_per_layer()
        print(f"Answering Cache lengths: min={min(new_lens)}, max={max(new_lens)}")
        torch.cuda.empty_cache()
        return output

    model._get_cache_seq_len_per_layer = MethodType(get_cache_seq_len_per_layer_ragged, model)
    model._get_next_global_offset_per_layer = MethodType(get_next_global_offset_per_layer_ragged, model)
    model._truncate_kv_cache = MethodType(truncate_kv_cache_ragged, model)
    model.encode_init_prompt = MethodType(encode_init_prompt_ragged, model)
    model.encode_video_chunk = MethodType(encode_video_chunk_ragged, model)
    model.predict_and_compress = MethodType(predict_and_compress_ragged, model)
    model.question_answering = MethodType(question_answering_ragged_prefill, model)
    model._kv_head_ragged_prefill_enabled = True
    print("[kv_head_ragged_prefill] Installed physical per-KV-head ragged prefill/chunk/decode path.")
    return model
