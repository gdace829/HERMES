"""
Sanity-check HERMES manual pseudo-query attention scores against eager
``output_attentions=True`` attention.

The existing prev/current observation uses HERMES' manual attention scoring
path. This script reruns a small number of chunk observations with
attn_implementation="eager", computes both manual and eager pseudo-query
attention on the same KV cache, and reports head-level correlations.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.abstract_hermes import Abstract_Hermes
from inference.qwenvl_hermes import QwenVL_Hermes
from inference.reindex_3d import _get_mrope_section, contiguous_kv
from video_qa.base import BaseVQA

try:
    from head_analysis.generate_prev_current_profile_artifacts import save_scatter
except Exception:
    save_scatter = None


EPS = 1e-12


class EagerSanityQwenVL(QwenVL_Hermes):
    @torch.inference_mode()
    def compute_eager_attentions(self, input_ids):
        """Run full eager forward attention for pseudo-query tokens."""
        device = self.device
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = input_ids.shape[1]
        batch = input_ids.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(
                global_offset_per_layer[layer_idx], q_len, batch
            )
            self._layer_position_ids[layer_idx] = position_ids_3d

        default_position_ids = self._build_position_ids_3d_for_text(
            global_offset_per_layer[0], q_len, batch
        )
        # Keep video encoding and normal compression on the model's default
        # attention implementation. Only this short pseudo-query forward is
        # switched to eager so output_attentions is the full model attention
        # without making long video-token forwards materialize huge maps.
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
            raise RuntimeError(
                "Eager output_attentions returned None. Check that the model was "
                "temporarily switched to attn_implementation='eager'."
            )
        return out.attentions


def build_eager_model(args, model_path):
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

    model = EagerSanityQwenVL.__new__(EagerSanityQwenVL)
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


def current_share_rows(model, manual_attn, eager_attn, query_name, pre_lens, post_lens, meta):
    rows = []
    visual_start = int(model.visual_start_idx)
    pos_cache = model._position_ids_cache

    for layer_idx, (am, ae) in enumerate(zip(manual_attn, eager_attn)):
        if am.dim() < 4 or ae.dim() < 4:
            continue
        if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
            cached_kv_len = int(pos_cache[layer_idx].shape[1])
        else:
            cached_kv_len = int(am.shape[3])

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

        manual_head = am[0].mean(dim=1)
        eager_head = ae[0].mean(dim=1)
        if manual_head.shape[0] != eager_head.shape[0]:
            raise RuntimeError(
                f"Head mismatch at layer {layer_idx}: manual={manual_head.shape}, eager={eager_head.shape}"
            )

        manual_prev = manual_head[:, prev_start:prev_end].sum(dim=1).float()
        manual_curr = manual_head[:, curr_start:curr_end].sum(dim=1).float()
        eager_prev = eager_head[:, prev_start:prev_end].sum(dim=1).float()
        eager_curr = eager_head[:, curr_start:curr_end].sum(dim=1).float()

        manual_share = manual_curr / (manual_prev + manual_curr + EPS)
        eager_share = eager_curr / (eager_prev + eager_curr + EPS)
        manual_ratio = (manual_curr / curr_tokens) / (manual_prev / prev_tokens + EPS)
        eager_ratio = (eager_curr / curr_tokens) / (eager_prev / prev_tokens + EPS)

        for head_idx in range(manual_head.shape[0]):
            row = {
                "query": query_name,
                "layer": layer_idx,
                "head": head_idx,
                "pre_len": pre_len,
                "post_len": post_len,
                "cached_kv_len": cached_kv_len,
                "visual_start": visual_start,
                "prev_visual_tokens": prev_tokens,
                "current_chunk_tokens": curr_tokens,
                "manual_prev_mass": float(manual_prev[head_idx].item()),
                "manual_current_mass": float(manual_curr[head_idx].item()),
                "manual_current_share": float(manual_share[head_idx].item()),
                "manual_current_to_prev_per_token_ratio": float(manual_ratio[head_idx].item()),
                "eager_prev_mass": float(eager_prev[head_idx].item()),
                "eager_current_mass": float(eager_curr[head_idx].item()),
                "eager_current_share": float(eager_share[head_idx].item()),
                "eager_current_to_prev_per_token_ratio": float(eager_ratio[head_idx].item()),
                "abs_current_share_diff": float(abs(manual_share[head_idx] - eager_share[head_idx]).item()),
            }
            row.update(meta)
            rows.append(row)
    return rows


def corr(x, y):
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2 or xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def aggregate_and_save(rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw_path = os.path.join(save_dir, "manual_vs_eager_raw.csv")
    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)

    if df.empty:
        summary = {"num_rows": 0, "raw_csv": raw_path}
        with open(os.path.join(save_dir, "manual_vs_eager_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    head = (
        df.groupby(["query", "layer", "head"])
        .agg(
            manual_current_share=("manual_current_share", "mean"),
            eager_current_share=("eager_current_share", "mean"),
            manual_per_token_ratio=("manual_current_to_prev_per_token_ratio", "mean"),
            eager_per_token_ratio=("eager_current_to_prev_per_token_ratio", "mean"),
            abs_current_share_diff=("abs_current_share_diff", "mean"),
            num_observations=("manual_current_share", "size"),
        )
        .reset_index()
    )
    head_path = os.path.join(save_dir, "manual_vs_eager_head_scores.csv")
    head.to_csv(head_path, index=False, quoting=csv.QUOTE_MINIMAL)

    wide = head.pivot_table(
        index=["layer", "head"],
        columns="query",
        values=["manual_current_share", "eager_current_share"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    if {"manual_current_share_local", "manual_current_share_global", "eager_current_share_local", "eager_current_share_global"}.issubset(wide.columns):
        wide["manual_s_current_share"] = 0.5 * (
            wide["manual_current_share_local"] + wide["manual_current_share_global"]
        )
        wide["eager_s_current_share"] = 0.5 * (
            wide["eager_current_share_local"] + wide["eager_current_share_global"]
        )
    wide_path = os.path.join(save_dir, "manual_vs_eager_head_scores_wide.csv")
    wide.to_csv(wide_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "num_rows": int(len(df)),
        "num_chunk_observations": int(df[["video_idx", "chunk_idx"]].drop_duplicates().shape[0])
        if {"video_idx", "chunk_idx"}.issubset(df.columns)
        else None,
        "raw_csv": raw_path,
        "head_scores_csv": head_path,
        "head_scores_wide_csv": wide_path,
        "raw_current_share_corr": corr(df["manual_current_share"], df["eager_current_share"]),
        "raw_current_share_mae": float(df["abs_current_share_diff"].mean()),
        "head_current_share_corr_all_queries": corr(head["manual_current_share"], head["eager_current_share"]),
        "head_current_share_mae_all_queries": float(
            np.mean(np.abs(head["manual_current_share"] - head["eager_current_share"]))
        ),
        "per_query": {},
    }
    for query, part in head.groupby("query"):
        summary["per_query"][query] = {
            "head_current_share_corr": corr(part["manual_current_share"], part["eager_current_share"]),
            "head_current_share_mae": float(
                np.mean(np.abs(part["manual_current_share"] - part["eager_current_share"]))
            ),
            "manual_mean_current_share": float(part["manual_current_share"].mean()),
            "eager_mean_current_share": float(part["eager_current_share"].mean()),
        }
    if "manual_s_current_share" in wide.columns:
        summary["s_current_share_corr"] = corr(wide["manual_s_current_share"], wide["eager_s_current_share"])
        summary["s_current_share_mae"] = float(
            np.mean(np.abs(wide["manual_s_current_share"] - wide["eager_s_current_share"]))
        )
        if save_scatter is not None:
            scatter_path = os.path.join(save_dir, "manual_vs_eager_s_current_share_scatter.png")
            save_scatter(
                wide["manual_s_current_share"].tolist(),
                wide["eager_s_current_share"].tolist(),
                scatter_path,
                "Manual vs eager query-robust current-share",
                "manual s_h",
                "eager s_h",
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
            )
            summary["s_current_share_scatter"] = scatter_path

    summary_path = os.path.join(save_dir, "manual_vs_eager_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"
    print(f"Loading eager model on cuda:{args.device}: {model_path}")
    model, processor = build_eager_model(args, model_path)
    print(f"Model loaded. attn_impl={getattr(model.language_model.config, '_attn_implementation', None)}")

    with open(anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/eager_attention_sanity_tmp",
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
                meta = {
                    "video_idx": video_idx,
                    "chunk_idx": chunk_idx,
                    "question_idx": question_idx,
                    "frame_start": current_frame_idx,
                    "frame_end": next_encode_end,
                    "task": sample.get("task", "Unknown"),
                }

                for query_name, query_text in (("local", local_q), ("global", global_q)):
                    input_ids = processor.tokenizer(query_text).input_ids
                    input_ids = torch.as_tensor([input_ids], device=model.device, dtype=torch.int)
                    manual = model._compute_attention_scores_manually(input_ids, model.kv_cache)
                    eager = model.compute_eager_attentions(input_ids)
                    rows.extend(
                        current_share_rows(
                            model, manual, eager, query_name, pre_lens, post_lens, meta
                        )
                    )
                    torch.cuda.empty_cache()

                # Keep the streaming state close to normal HERMES after measuring.
                if model.compress_mode == "streamingvlm":
                    model._sliding_window_compress()
                else:
                    model.pseudo_forward(local_q, global_q)

                obs_count += 1
                current_frame_idx = next_encode_end
                chunk_idx += 1

        if args.max_observations and obs_count >= args.max_observations:
            break

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
    parser.add_argument("--max_questions", type=int, default=1)
    parser.add_argument("--max_observations", type=int, default=4)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/observations/eager_attention_sanity_n1_o4",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
