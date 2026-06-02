"""
Measure true eager pseudo-query attention allocation to previous visual cache
vs the latest encoded video chunk.

Unlike obs_prev_current_chunk_attention.py, this observer does not use the
HERMES manual attention scoring path for the measurement. Video encoding and
normal HERMES compression stay on the model's default attention implementation
(typically SDPA), while only the short local/global pseudo-query forward is
temporarily switched to eager with output_attentions=True.
"""

import argparse
import csv
import json
import math
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.abstract_hermes import Abstract_Hermes
from inference.qwenvl_hermes import QwenVL_Hermes
from inference.reindex_3d import _get_mrope_section
from video_qa.base import BaseVQA


EPS = 1e-12


class EagerObservedQwenVL(QwenVL_Hermes):
    @torch.inference_mode()
    def compute_eager_attentions(self, input_ids):
        device = self.device
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = input_ids.shape[1]
        batch = input_ids.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_text(
                global_offset_per_layer[layer_idx], q_len, batch
            )

        default_position_ids = self._build_position_ids_3d_for_text(
            global_offset_per_layer[0], q_len, batch
        )

        old_model_impl = getattr(self.config, "_attn_implementation", None)
        old_lm_impl = getattr(self.language_model.config, "_attn_implementation", None)
        try:
            if old_model_impl is not None:
                self.config._attn_implementation = "eager"
            if old_lm_impl is not None:
                self.language_model.config._attn_implementation = "eager"
            out = self.language_model(
                input_ids=input_ids,
                use_cache=False,
                past_key_values=self.kv_cache,
                output_attentions=True,
                return_dict=True,
                position_ids=default_position_ids,
            )
        finally:
            if old_model_impl is not None:
                self.config._attn_implementation = old_model_impl
            if old_lm_impl is not None:
                self.language_model.config._attn_implementation = old_lm_impl
            self._layer_position_ids.clear()

        if out.attentions is None or any(att is None for att in out.attentions):
            raise RuntimeError("Eager output_attentions returned None.")
        return out.attentions


def build_observed_model(args, model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    device = f"cuda:{args.device}" if args.device >= 0 else "cuda"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )

    model = EagerObservedQwenVL.__new__(EagerObservedQwenVL)
    model.__dict__ = raw_model.__dict__.copy()

    Abstract_Hermes.__init__(model, processor, init_prompt_ids.tolist(), args.kv_size)
    model.streaming = True
    model.sample_fps = args.sample_fps
    model.compress_mode = args.compress_mode

    num_layers = raw_model.model.config.num_hidden_layers
    model.num_layers = num_layers
    model._position_ids_cache = [None for _ in range(num_layers)]
    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(model.num_layers * model.short_term_ratio)
    model.long_term_threshold = int(model.num_layers * (1 - model.long_term_ratio))
    model.total_processed_frames = 0
    model._mrope_section = _get_mrope_section(raw_model.model)
    model._layer_position_ids = {}
    model._hook_handles = []
    model._register_forward_hooks()
    model.eval()
    return model, processor


def observe_prev_current(model, attn_local, attn_global, pre_lens, post_lens, meta):
    rows = []
    visual_start = int(model.visual_start_idx)
    pos_cache = model._position_ids_cache

    for layer_idx, (al, ag) in enumerate(zip(attn_local, attn_global)):
        if al.dim() < 4 or ag.dim() < 4:
            continue

        if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
            cached_kv_len = int(pos_cache[layer_idx].shape[1])
        else:
            cached_kv_len = int(al.shape[3])

        pre_len = min(int(pre_lens[layer_idx]), cached_kv_len)
        post_len = min(int(post_lens[layer_idx]), cached_kv_len)

        prev_start = min(visual_start, cached_kv_len)
        prev_end = min(max(pre_len, visual_start), cached_kv_len)
        curr_start = min(max(pre_len, visual_start), cached_kv_len)
        curr_end = min(max(post_len, curr_start), cached_kv_len)

        prev_tokens = prev_end - prev_start
        curr_tokens = curr_end - curr_start
        if prev_tokens <= 0 or curr_tokens <= 0:
            continue

        local_head = al[0].mean(dim=1)
        global_head = ag[0].mean(dim=1)

        local_prev = local_head[:, prev_start:prev_end].sum(dim=1).float()
        local_curr = local_head[:, curr_start:curr_end].sum(dim=1).float()
        global_prev = global_head[:, prev_start:prev_end].sum(dim=1).float()
        global_curr = global_head[:, curr_start:curr_end].sum(dim=1).float()

        local_total = local_prev + local_curr
        global_total = global_prev + global_curr

        local_curr_share = local_curr / (local_total + EPS)
        global_curr_share = global_curr / (global_total + EPS)
        local_prev_share = local_prev / (local_total + EPS)
        global_prev_share = global_prev / (global_total + EPS)

        local_per_token_ratio = (local_curr / curr_tokens) / (local_prev / prev_tokens + EPS)
        global_per_token_ratio = (global_curr / curr_tokens) / (global_prev / prev_tokens + EPS)
        local_mass_ratio = local_curr / (local_prev + EPS)
        global_mass_ratio = global_curr / (global_prev + EPS)
        current_token_fraction = curr_tokens / (prev_tokens + curr_tokens)

        for head_idx in range(local_head.shape[0]):
            row = {
                "layer": layer_idx,
                "head": head_idx,
                "pre_len": pre_len,
                "post_len": post_len,
                "cached_kv_len": cached_kv_len,
                "visual_start": visual_start,
                "prev_visual_tokens": prev_tokens,
                "current_chunk_tokens": curr_tokens,
                "local_prev_mass": float(local_prev[head_idx].item()),
                "local_current_mass": float(local_curr[head_idx].item()),
                "local_prev_share": float(local_prev_share[head_idx].item()),
                "local_current_share": float(local_curr_share[head_idx].item()),
                "local_current_share_minus_token_fraction": float(
                    local_curr_share[head_idx].item() - current_token_fraction
                ),
                "local_current_to_prev_mass_ratio": float(local_mass_ratio[head_idx].item()),
                "local_current_to_prev_per_token_ratio": float(local_per_token_ratio[head_idx].item()),
                "global_prev_mass": float(global_prev[head_idx].item()),
                "global_current_mass": float(global_curr[head_idx].item()),
                "global_prev_share": float(global_prev_share[head_idx].item()),
                "global_current_share": float(global_curr_share[head_idx].item()),
                "global_current_share_minus_token_fraction": float(
                    global_curr_share[head_idx].item() - current_token_fraction
                ),
                "global_current_to_prev_mass_ratio": float(global_mass_ratio[head_idx].item()),
                "global_current_to_prev_per_token_ratio": float(global_per_token_ratio[head_idx].item()),
                "current_token_fraction": float(current_token_fraction),
                "global_minus_local_current_share": float(
                    (global_curr_share[head_idx] - local_curr_share[head_idx]).item()
                ),
            }
            row.update(meta)
            rows.append(row)
    return rows


def finite_mean(df, name):
    vals = df[name].replace([float("inf"), -float("inf")], float("nan")).dropna()
    return float(vals.mean()) if len(vals) else None


def finite_median(df, name):
    vals = df[name].replace([float("inf"), -float("inf")], float("nan")).dropna()
    return float(vals.median()) if len(vals) else None


def aggregate_and_save(rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw_path = os.path.join(save_dir, "raw_prev_current_attention.csv")
    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)

    if df.empty:
        summary = {"num_rows": 0, "raw_csv": raw_path}
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    layer_path = os.path.join(save_dir, "per_layer_summary.csv")
    layer_summary = (
        df.groupby("layer")[
            [
                "local_current_share",
                "global_current_share",
                "global_minus_local_current_share",
                "local_current_to_prev_per_token_ratio",
                "global_current_to_prev_per_token_ratio",
                "prev_visual_tokens",
                "current_chunk_tokens",
            ]
        ]
        .mean()
        .reset_index()
    )
    layer_summary.to_csv(layer_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "attention_source": "eager_output_attentions_for_pseudo_query_only",
        "num_rows": int(len(df)),
        "num_unique_observations": int(df[["video_idx", "chunk_idx"]].drop_duplicates().shape[0])
        if {"video_idx", "chunk_idx"}.issubset(df.columns)
        else None,
        "num_layers": int(df["layer"].max()) + 1,
        "num_heads": int(df["head"].max()) + 1,
        "raw_csv": raw_path,
        "per_layer_summary_csv": layer_path,
        "mean_local_current_share": finite_mean(df, "local_current_share"),
        "mean_global_current_share": finite_mean(df, "global_current_share"),
        "mean_current_token_fraction": finite_mean(df, "current_token_fraction"),
        "mean_global_minus_local_current_share": finite_mean(df, "global_minus_local_current_share"),
        "median_local_current_to_prev_per_token_ratio": finite_median(
            df, "local_current_to_prev_per_token_ratio"
        ),
        "median_global_current_to_prev_per_token_ratio": finite_median(
            df, "global_current_to_prev_per_token_ratio"
        ),
        "mean_prev_visual_tokens": finite_mean(df, "prev_visual_tokens"),
        "mean_current_chunk_tokens": finite_mean(df, "current_chunk_tokens"),
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)
    print(f"Model loaded. default_attn_impl={getattr(model.language_model.config, '_attn_implementation', None)}")

    with open(anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/eager_prev_current_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    rows = []
    obs_count = 0
    for video_idx, video_sample in enumerate(tqdm(anno, desc="Videos")):
        video_path = video_sample["video_path"]
        if video_path.endswith(".npy"):
            video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            video_fps = video_sample.get("fps", None)
            video = analyzer.load_video_frames(video_path, video_fps, clip=video_sample.get("clip", None))
            video_tensor = torch.from_numpy(video)
        else:
            video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
            video_tensor = torch.from_numpy(video)

        model.clear_cache()
        model.encode_init_prompt()
        current_frame_idx = 0
        chunk_idx = 0

        conversations = video_sample.get("conversations", [])
        if args.max_questions is not None:
            conversations = conversations[: args.max_questions]

        for question_idx, sample in enumerate(conversations):
            if "end_time" in sample:
                end_frame_idx = math.ceil(sample["end_time"] * args.sample_fps)
            else:
                end_frame_idx = len(video_tensor)

            while current_frame_idx < end_frame_idx:
                if args.max_observations and obs_count >= args.max_observations:
                    summary = aggregate_and_save(rows, args.save_dir)
                    print(json.dumps(summary, indent=2))
                    return summary

                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                if next_encode_end <= current_frame_idx:
                    break

                video_chunk = video_tensor[current_frame_idx:next_encode_end]
                pre_lens = model._get_cache_seq_len_per_layer()
                model.encode_video_chunk(video_chunk)
                post_lens = model._get_cache_seq_len_per_layer()

                local_q, global_q = model.predict_next_question()
                local_ids = processor.tokenizer(local_q).input_ids
                local_ids = torch.as_tensor([local_ids], device=model.device, dtype=torch.int)
                attn_local = model.compute_eager_attentions(local_ids)

                global_ids = processor.tokenizer(global_q).input_ids
                global_ids = torch.as_tensor([global_ids], device=model.device, dtype=torch.int)
                attn_global = model.compute_eager_attentions(global_ids)

                meta = {
                    "video_idx": video_idx,
                    "chunk_idx": chunk_idx,
                    "question_idx": question_idx,
                    "frame_start": current_frame_idx,
                    "frame_end": next_encode_end,
                    "task": sample.get("task", "Unknown"),
                }
                rows.extend(observe_prev_current(model, attn_local, attn_global, pre_lens, post_lens, meta))

                if model.compress_mode == "streamingvlm":
                    model._sliding_window_compress()
                else:
                    model.pseudo_forward(local_q, global_q)

                obs_count += 1
                current_frame_idx = next_encode_end
                chunk_idx += 1
                torch.cuda.empty_cache()

    summary = aggregate_and_save(rows, args.save_dir)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--num_videos", type=int, default=1)
    parser.add_argument("--max_questions", type=int, default=4)
    parser.add_argument("--max_observations", type=int, default=6)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/observations/obs_prev_current_chunk_attention_eager_n1_o6",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
