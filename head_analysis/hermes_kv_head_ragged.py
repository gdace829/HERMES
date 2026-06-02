"""
Decode-only physical per-KV-head ragged cache for Qwen2.5-VL HERMES.

This is an experimental bridge toward HybridKV-style cache layout. It keeps
HERMES streaming/video-chunk encoding on the existing dense cache path, then
converts the answer-generation cache to:

    flat_k / flat_v / head_lens / cu_klen

for token-by-token decode. Each KV head can hold a different number of tokens.
"""

import re
import time
from types import MethodType

import torch
import torch.nn.functional as F
from logzero import logger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

try:
    from flash_attn import flash_attn_varlen_func
except Exception:  # pragma: no cover - runtime optional
    flash_attn_varlen_func = None


class HeadRaggedLayerCache:
    def __init__(self, flat_k, flat_v, head_lens, flat_pos=None):
        self.flat_k = flat_k.contiguous()
        self.flat_v = flat_v.contiguous()
        self.head_lens = head_lens.to(device=flat_k.device, dtype=torch.int32)
        self.flat_pos = None if flat_pos is None else flat_pos.to(device=flat_k.device).contiguous()
        self._refresh_cu_klen()

    @classmethod
    def from_dense_layer(cls, k_layer, v_layer, allowed_positions=None,
                         union_len=None, position_ids=None):
        if k_layer.shape[0] != 1:
            raise ValueError("HeadRaggedLayerCache currently supports batch_size=1 only")

        _, num_kv_heads, seq_len, _ = k_layer.shape
        device = k_layer.device
        if union_len is None:
            union_len = seq_len
        if position_ids is not None:
            position_ids = position_ids.to(device=device)
            if position_ids.dim() == 3:
                position_ids = position_ids[:, 0, :]
            if position_ids.shape[-1] != seq_len:
                position_ids = None

        flat_k_parts = []
        flat_v_parts = []
        flat_pos_parts = []
        head_lens = []

        for kv_head in range(num_kv_heads):
            if allowed_positions is None:
                idx = torch.arange(seq_len, device=device, dtype=torch.long)
            else:
                idx = allowed_positions[kv_head]
                if not isinstance(idx, torch.Tensor):
                    idx = torch.as_tensor(idx, device=device, dtype=torch.long)
                else:
                    idx = idx.to(device=device, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < seq_len)]

                # Tokens appended after the dense-union compression point are
                # shared by all heads: long-term summary, question prefill, etc.
                if union_len < seq_len:
                    tail = torch.arange(union_len, seq_len, device=device, dtype=torch.long)
                    idx = torch.cat([idx, tail])
                idx = torch.unique(idx, sorted=True)

            flat_k_parts.append(k_layer[0, kv_head, idx, :])
            flat_v_parts.append(v_layer[0, kv_head, idx, :])
            if position_ids is not None:
                flat_pos_parts.append(position_ids[:, idx])
            head_lens.append(int(idx.numel()))

        flat_pos = torch.cat(flat_pos_parts, dim=1) if flat_pos_parts else None
        return cls(
            torch.cat(flat_k_parts, dim=0),
            torch.cat(flat_v_parts, dim=0),
            torch.tensor(head_lens, device=device, dtype=torch.int32),
            flat_pos=flat_pos,
        )

    def _refresh_cu_klen(self):
        zero = torch.zeros(1, device=self.head_lens.device, dtype=torch.int32)
        self.cu_klen = torch.cat([
            zero,
            torch.cumsum(self.head_lens, dim=0, dtype=torch.int32),
        ]).to(dtype=torch.int32)
        self.max_seqlen_k = int(self.head_lens.max().item()) if self.head_lens.numel() else 0

    def _normalize_pos(self, pos, length):
        if pos is None:
            return None
        pos = pos.to(device=self.flat_k.device)
        if pos.dim() == 3:
            pos = pos[:, 0, :]
        elif pos.dim() == 1:
            pos = pos.view(3, 1)
        if pos.shape != (3, length):
            raise ValueError(f"Expected position shape {(3, length)}, got {tuple(pos.shape)}")
        return pos

    def append(self, key_states, value_states, pos=None):
        """Append one decoded token for every KV head."""
        if key_states.dim() == 4:
            key_states = key_states[0, :, 0, :]
            value_states = value_states[0, :, 0, :]
        elif key_states.dim() == 3:
            key_states = key_states[:, 0, :]
            value_states = value_states[:, 0, :]

        num_kv_heads = int(self.head_lens.numel())
        if key_states.shape[0] != num_kv_heads:
            raise ValueError(
                f"Expected {num_kv_heads} KV heads, got {key_states.shape[0]}"
            )

        new_k_parts = []
        new_v_parts = []
        pos = self._normalize_pos(pos, 1)
        if pos is None and self.flat_pos is not None:
            next_pos = self.flat_pos.max(dim=1).values + 1
            pos = next_pos.view(3, 1)
        new_pos_parts = [] if (self.flat_pos is not None or pos is not None) else None
        start = 0
        for kv_head in range(num_kv_heads):
            length = int(self.head_lens[kv_head].item())
            end = start + length
            new_k_parts.append(self.flat_k[start:end])
            new_k_parts.append(key_states[kv_head:kv_head + 1])
            new_v_parts.append(self.flat_v[start:end])
            new_v_parts.append(value_states[kv_head:kv_head + 1])
            if new_pos_parts is not None:
                if self.flat_pos is not None:
                    new_pos_parts.append(self.flat_pos[:, start:end])
                else:
                    new_pos_parts.append(torch.empty((3, length), device=self.flat_k.device, dtype=torch.float32))
                new_pos_parts.append(pos)
            start = end

        self.flat_k = torch.cat(new_k_parts, dim=0).contiguous()
        self.flat_v = torch.cat(new_v_parts, dim=0).contiguous()
        if new_pos_parts is not None:
            self.flat_pos = torch.cat(new_pos_parts, dim=1).contiguous()
        self.head_lens = self.head_lens + 1
        self._refresh_cu_klen()

    def append_chunk(self, key_states, value_states, pos=None):
        """Append q_len current tokens for every KV head."""
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError("append_chunk expects [batch, kv_heads, q_len, head_dim] tensors")
        if key_states.shape[0] != 1:
            raise ValueError("HeadRaggedLayerCache currently supports batch_size=1 only")

        _, num_kv_heads, q_len, _ = key_states.shape
        if num_kv_heads != int(self.head_lens.numel()):
            raise ValueError(
                f"Expected {int(self.head_lens.numel())} KV heads, got {num_kv_heads}"
            )

        pos = self._normalize_pos(pos, q_len)
        if pos is None and self.flat_pos is not None:
            start_pos = self.flat_pos.max(dim=1).values + 1
            delta = torch.arange(q_len, device=self.flat_k.device, dtype=start_pos.dtype)
            pos = start_pos.view(3, 1) + delta.view(1, -1)
        new_k_parts = []
        new_v_parts = []
        new_pos_parts = [] if (self.flat_pos is not None or pos is not None) else None
        start = 0
        for kv_head in range(num_kv_heads):
            length = int(self.head_lens[kv_head].item())
            end = start + length
            new_k_parts.append(self.flat_k[start:end])
            new_k_parts.append(key_states[0, kv_head])
            new_v_parts.append(self.flat_v[start:end])
            new_v_parts.append(value_states[0, kv_head])
            if new_pos_parts is not None:
                if self.flat_pos is not None:
                    new_pos_parts.append(self.flat_pos[:, start:end])
                else:
                    new_pos_parts.append(torch.empty((3, length), device=self.flat_k.device, dtype=torch.float32))
                new_pos_parts.append(pos)
            start = end

        self.flat_k = torch.cat(new_k_parts, dim=0).contiguous()
        self.flat_v = torch.cat(new_v_parts, dim=0).contiguous()
        if new_pos_parts is not None:
            self.flat_pos = torch.cat(new_pos_parts, dim=1).contiguous()
        self.head_lens = self.head_lens + int(q_len)
        self._refresh_cu_klen()

    def prune_per_head(self, keep_indices_per_head):
        """Keep physical indices independently inside every KV-head segment."""
        num_kv_heads = int(self.head_lens.numel())
        if len(keep_indices_per_head) != num_kv_heads:
            raise ValueError(
                f"Expected {num_kv_heads} keep-index lists, got {len(keep_indices_per_head)}"
            )

        new_k_parts = []
        new_v_parts = []
        new_pos_parts = [] if self.flat_pos is not None else None
        new_lens = []
        start = 0
        for kv_head in range(num_kv_heads):
            length = int(self.head_lens[kv_head].item())
            end = start + length
            idx = keep_indices_per_head[kv_head]
            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx, device=self.flat_k.device, dtype=torch.long)
            else:
                idx = idx.to(device=self.flat_k.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < length)]
            if idx.numel() == 0:
                idx = torch.tensor([0], device=self.flat_k.device, dtype=torch.long)
            idx = torch.unique(idx, sorted=True)
            abs_idx = start + idx
            new_k_parts.append(self.flat_k.index_select(0, abs_idx))
            new_v_parts.append(self.flat_v.index_select(0, abs_idx))
            if new_pos_parts is not None:
                new_pos_parts.append(self.flat_pos.index_select(1, abs_idx))
            new_lens.append(int(idx.numel()))
            start = end

        self.flat_k = torch.cat(new_k_parts, dim=0).contiguous()
        self.flat_v = torch.cat(new_v_parts, dim=0).contiguous()
        if new_pos_parts is not None:
            self.flat_pos = torch.cat(new_pos_parts, dim=1).contiguous()
        self.head_lens = torch.tensor(new_lens, device=self.flat_k.device, dtype=torch.int32)
        self._refresh_cu_klen()

    def truncate(self, target_lens):
        """Keep a prefix length for every KV head."""
        target_lens = target_lens.to(device=self.flat_k.device, dtype=torch.long)
        new_k_parts = []
        new_v_parts = []
        new_pos_parts = [] if self.flat_pos is not None else None
        new_lens = []
        start = 0
        for kv_head in range(int(self.head_lens.numel())):
            length = int(self.head_lens[kv_head].item())
            keep = max(0, min(int(target_lens[kv_head].item()), length))
            end = start + keep
            new_k_parts.append(self.flat_k[start:end])
            new_v_parts.append(self.flat_v[start:end])
            if new_pos_parts is not None:
                new_pos_parts.append(self.flat_pos[:, start:end])
            new_lens.append(keep)
            start += length

        self.flat_k = torch.cat(new_k_parts, dim=0).contiguous()
        self.flat_v = torch.cat(new_v_parts, dim=0).contiguous()
        if new_pos_parts is not None:
            self.flat_pos = torch.cat(new_pos_parts, dim=1).contiguous()
        self.head_lens = torch.tensor(new_lens, device=self.flat_k.device, dtype=torch.int32)
        self._refresh_cu_klen()

    def get_segment(self, kv_head):
        start = int(self.cu_klen[kv_head].item())
        end = int(self.cu_klen[kv_head + 1].item())
        return self.flat_k[start:end], self.flat_v[start:end]

    def get_position_segment(self, kv_head):
        if self.flat_pos is None:
            return None
        start = int(self.cu_klen[kv_head].item())
        end = int(self.cu_klen[kv_head + 1].item())
        return self.flat_pos[:, start:end]


class HeadRaggedCache:
    def __init__(self, layers):
        self.layers = layers

    @classmethod
    def from_dense_cache(cls, model, past_key_values):
        layers = []
        cache_positions = getattr(model, "_kv_head_budget_cache_positions", {})
        union_lens = getattr(model, "_kv_head_budget_union_lens", {})
        position_caches = getattr(model, "_position_ids_cache", None)

        for layer_idx in range(len(past_key_values)):
            k_layer, v_layer = past_key_values[layer_idx]
            position_ids = None
            if position_caches is not None and layer_idx < len(position_caches):
                position_ids = position_caches[layer_idx]
            layers.append(
                HeadRaggedLayerCache.from_dense_layer(
                    k_layer,
                    v_layer,
                    allowed_positions=cache_positions.get(layer_idx),
                    union_len=union_lens.get(layer_idx, k_layer.shape[2]),
                    position_ids=position_ids,
                )
            )
        return cls(layers)

    def __len__(self):
        return len(self.layers)

    def append(self, layer_idx, key_states, value_states, pos=None):
        self.layers[layer_idx].append(key_states, value_states, pos=pos)

    def append_chunk(self, layer_idx, key_states, value_states, pos=None):
        self.layers[layer_idx].append_chunk(key_states, value_states, pos=pos)

    def prune_per_head(self, keep_indices):
        for layer_idx, layer_keep in enumerate(keep_indices):
            self.layers[layer_idx].prune_per_head(layer_keep)

    def snapshot_head_lens(self):
        return [layer.head_lens.clone() for layer in self.layers]

    def truncate(self, target_lens):
        for layer_idx, layer_lens in enumerate(target_lens):
            self.layers[layer_idx].truncate(layer_lens)

    def stats(self):
        lens = torch.cat([layer.head_lens for layer in self.layers])
        return int(lens.min().item()), int(lens.max().item()), float(lens.float().mean().item())

    def max_lens_per_layer(self):
        return [int(layer.head_lens.max().item()) for layer in self.layers]


def _ragged_attention_manual(query_states, layer_cache, num_kv_heads, group_size, scaling):
    # query_states: [1, q_heads, 1, head_dim]
    q = query_states[0, :, 0, :]
    head_outputs = []
    for kv_head in range(num_kv_heads):
        q_group = q[kv_head * group_size:(kv_head + 1) * group_size]
        k_seg, v_seg = layer_cache.get_segment(kv_head)
        scores = torch.matmul(q_group.float(), k_seg.float().transpose(0, 1)) * scaling
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_group.dtype)
        head_outputs.append(torch.matmul(probs, v_seg))
    return torch.cat(head_outputs, dim=0).view(1, 1, -1)


def _ragged_attention_flash(query_states, layer_cache, num_kv_heads, group_size):
    if flash_attn_varlen_func is None:
        return None
    if query_states.device.type != "cuda":
        return None
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        return None

    _, _, _, head_dim = query_states.shape
    q_flat = query_states[0, :, 0, :].view(num_kv_heads, group_size, head_dim).contiguous()
    k_flat = layer_cache.flat_k.view(-1, 1, head_dim).contiguous()
    v_flat = layer_cache.flat_v.view(-1, 1, head_dim).contiguous()
    cu_q = torch.arange(
        0,
        num_kv_heads + 1,
        device=query_states.device,
        dtype=torch.int32,
    )

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        layer_cache.cu_klen,
        max_seqlen_q=1,
        max_seqlen_k=layer_cache.max_seqlen_k,
        causal=True,
    )
    return out.reshape(1, 1, num_kv_heads * group_size * head_dim)


def _ragged_self_attn_forward(model, layer_idx, attn, hidden_states,
                              ragged_cache, position_embeddings):
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1 or q_len != 1:
        raise ValueError("Ragged decode forward only supports batch=1 and q_len=1")

    num_query_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    group_size = num_query_heads // num_kv_heads
    head_dim = attn.head_dim

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_query_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    mrope_section = getattr(model, "_mrope_section", attn.rope_scaling["mrope_section"])
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        mrope_section,
    )

    ragged_cache.append(layer_idx, key_states, value_states)
    layer_cache = ragged_cache.layers[layer_idx]

    attn_output = _ragged_attention_flash(query_states, layer_cache, num_kv_heads, group_size)
    if attn_output is None:
        scaling = getattr(attn, "scaling", head_dim ** -0.5)
        attn_output = _ragged_attention_manual(
            query_states,
            layer_cache,
            num_kv_heads,
            group_size,
            scaling,
        )

    return attn.o_proj(attn_output.contiguous())


def _ragged_language_decode_step(model, input_ids, ragged_cache, position_ids_by_layer):
    hidden_states = model.get_input_embeddings()(input_ids)

    for layer_idx, decoder_layer in enumerate(model.language_model.layers):
        residual = hidden_states
        hidden_states_norm = decoder_layer.input_layernorm(hidden_states)
        position_embeddings = model.language_model.rotary_emb(
            hidden_states_norm,
            position_ids_by_layer[layer_idx],
        )
        attn_output = _ragged_self_attn_forward(
            model,
            layer_idx,
            decoder_layer.self_attn,
            hidden_states_norm,
            ragged_cache,
            position_embeddings,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return model.language_model.norm(hidden_states)


def apply_kv_head_ragged_decode(model):
    """Patch question_answering to use HeadRaggedCache for decode tokens."""
    original_question_answering = model.question_answering

    @torch.inference_mode()
    def question_answering_ragged(self, input_text, max_new_tokens=128,
                                  temperature=0, repetition_penalty=1.1,
                                  pseudo_forward=False):
        if pseudo_forward:
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
        input_ids = torch.as_tensor([input_ids], device=device)

        self._ensure_dynamic_cache()
        past_lens_prefill = self._get_cache_seq_len_per_layer()
        global_offset_prefill = self._get_next_global_offset_per_layer()

        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(
                global_offset_prefill[layer_idx],
                q_len_prefill,
                batch,
            )
            self._layer_position_ids[layer_idx] = position_ids_3d

        position_ids_3d = self._build_position_ids_3d_for_text(
            global_offset_prefill[0],
            q_len_prefill,
            batch,
        )

        out = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=self.kv_cache,
            position_ids=position_ids_3d,
        )
        past_key_values = out.past_key_values
        logits = self.lm_head(out.last_hidden_state)

        for layer_idx in range(self.num_layers):
            offset = global_offset_prefill[layer_idx]
            self._append_position_ids_layer(layer_idx, [offset, offset, offset], q_len_prefill)
        self._layer_position_ids.clear()

        ragged_cache = HeadRaggedCache.from_dense_cache(self, past_key_values)
        rmin, rmax, rmean = ragged_cache.stats()
        print(f"[kv_head_ragged] decode cache head_lens: min={rmin}, max={rmax}, mean={rmean:.1f}")

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

            curr_global_offset = self._get_next_global_offset_per_layer()
            position_ids_by_layer = []
            for layer_idx in range(self.num_layers):
                position_ids_by_layer.append(
                    self._build_position_ids_3d_for_text(curr_global_offset[layer_idx], 1, 1)
                )

            hidden_states = _ragged_language_decode_step(
                self,
                torch.as_tensor([[token]], device=device),
                ragged_cache,
                position_ids_by_layer,
            )
            logits = self.lm_head(hidden_states)

            for layer_idx in range(self.num_layers):
                offset = curr_global_offset[layer_idx]
                self._append_position_ids_layer(layer_idx, [offset, offset, offset], 1)

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

        self._truncate_kv_cache(past_lens_prefill)
        for layer_idx in range(self.num_layers):
            if (
                self._position_ids_cache[layer_idx] is not None
                and self._position_ids_cache[layer_idx].shape[1] > past_lens_prefill[layer_idx]
            ):
                self._position_ids_cache[layer_idx] = self._position_ids_cache[layer_idx][
                    :, :past_lens_prefill[layer_idx]
                ].contiguous()

        new_lens = self._get_cache_seq_len_per_layer()
        print(f"Answering Cache lengths: min={min(new_lens)}, max={max(new_lens)}")
        torch.cuda.empty_cache()
        return output

    model.question_answering = MethodType(question_answering_ragged, model)
    model._kv_head_ragged_decode_enabled = True
    print("[kv_head_ragged] Installed decode-only physical per-KV-head ragged cache.")
    return model
