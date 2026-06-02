"""
Measure pseudo-query attention allocation to previous visual cache vs the latest
encoded video chunk.

This is a standalone observer. It does not modify inference code.

For each streaming chunk:
  1. record per-layer KV lengths before encode_video_chunk;
  2. encode the latest chunk;
  3. record per-layer KV lengths after encode_video_chunk;
  4. before the normal compression step, compute local/global pseudo-query
     attention and split cached visual keys into:
       previous visual cache: [visual_start_idx, pre_len)
       latest chunk:          [pre_len, post_len)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import QwenVL_Hermes
from video_qa.base import BaseVQA


class PrevCurrentObservedQwenVL(QwenVL_Hermes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_obs = []
        self._chunk_pre_lens = None
        self._chunk_post_lens = None
        self._chunk_meta = {}

    def set_current_chunk_bounds(self, pre_lens, post_lens, **meta):
        self._chunk_pre_lens = list(pre_lens)
        self._chunk_post_lens = list(post_lens)
        self._chunk_meta = dict(meta)

    def predict_and_compress(self):
        local_q, global_q = self.predict_next_question()

        local_ids = self.processor.tokenizer(local_q).input_ids
        local_ids = torch.as_tensor([local_ids], device=self.device, dtype=torch.int)
        attn_local = self._compute_attention_scores_manually(local_ids, self.kv_cache)

        global_ids = self.processor.tokenizer(global_q).input_ids
        global_ids = torch.as_tensor([global_ids], device=self.device, dtype=torch.int)
        attn_global = self._compute_attention_scores_manually(global_ids, self.kv_cache)

        self._observe_prev_current(attn_local, attn_global)

        if self.compress_mode == "streamingvlm":
            self._sliding_window_compress()
        else:
            self.pseudo_forward(local_q, global_q)

    @torch.inference_mode()
    def _observe_prev_current(self, attn_local, attn_global):
        if self._chunk_pre_lens is None or self._chunk_post_lens is None:
            return

        visual_start = self.visual_start_idx
        pos_cache = self._position_ids_cache
        eps = 1e-12

        for layer_idx, (al, ag) in enumerate(zip(attn_local, attn_global)):
            if al.dim() < 4 or ag.dim() < 4:
                continue

            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cached_kv_len = pos_cache[layer_idx].shape[1]
            else:
                cached_kv_len = al.shape[3]

            if layer_idx >= len(self._chunk_pre_lens) or layer_idx >= len(self._chunk_post_lens):
                continue

            pre_len = min(int(self._chunk_pre_lens[layer_idx]), cached_kv_len)
            post_len = min(int(self._chunk_post_lens[layer_idx]), cached_kv_len)

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

            local_curr_share = local_curr / (local_total + eps)
            global_curr_share = global_curr / (global_total + eps)
            local_prev_share = local_prev / (local_total + eps)
            global_prev_share = global_prev / (global_total + eps)

            local_per_token_ratio = (local_curr / curr_tokens) / (local_prev / prev_tokens + eps)
            global_per_token_ratio = (global_curr / curr_tokens) / (global_prev / prev_tokens + eps)
            local_mass_ratio = local_curr / (local_prev + eps)
            global_mass_ratio = global_curr / (global_prev + eps)
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
                row.update(self._chunk_meta)
                self.chunk_obs.append(row)


def build_observed_model(args, model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from inference.abstract_hermes import Abstract_Hermes
    from inference.reindex_3d import _get_mrope_section

    device = f"cuda:{args.device}" if args.device >= 0 else "cuda"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float16
    )

    model = PrevCurrentObservedQwenVL.__new__(PrevCurrentObservedQwenVL)
    model.__dict__ = raw_model.__dict__.copy()

    Abstract_Hermes.__init__(model, processor, init_prompt_ids.tolist(), args.kv_size)
    model.chunk_obs = []
    model._chunk_pre_lens = None
    model._chunk_post_lens = None
    model._chunk_meta = {}
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


def aggregate_and_save(rows, save_dir, num_layers):
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    raw_path = os.path.join(save_dir, "raw_prev_current_attention.csv")
    df.to_csv(raw_path, index=False)

    if df.empty:
        summary = {"num_rows": 0, "raw_csv": raw_path}
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    num_heads = int(df["head"].max()) + 1
    fields = [
        "local_current_share",
        "global_current_share",
        "local_current_share_minus_token_fraction",
        "global_current_share_minus_token_fraction",
        "global_minus_local_current_share",
        "local_current_to_prev_mass_ratio",
        "global_current_to_prev_mass_ratio",
        "local_current_to_prev_per_token_ratio",
        "global_current_to_prev_per_token_ratio",
    ]

    arrays = {}
    count = np.zeros((num_layers, num_heads), dtype=np.float64)
    for field in fields:
        arr_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
        for row in df[["layer", "head", field]].itertuples(index=False):
            l, h, val = int(row.layer), int(row.head), float(getattr(row, field))
            arr_sum[l, h] += val
            if field == fields[0]:
                count[l, h] += 1
        arrays[field] = np.divide(arr_sum, count, out=np.zeros_like(arr_sum), where=count > 0)

    npz_path = os.path.join(save_dir, "head_prev_current_attention.npz")
    np.savez(npz_path, agg_count=count, num_layers=num_layers, num_heads=num_heads, **arrays)

    layer_summary = (
        df.groupby("layer")[fields + ["prev_visual_tokens", "current_chunk_tokens"]]
        .mean()
        .reset_index()
    )
    layer_path = os.path.join(save_dir, "per_layer_summary.csv")
    layer_summary.to_csv(layer_path, index=False)

    def finite_mean(name):
        vals = df[name].replace([np.inf, -np.inf], np.nan).dropna()
        return float(vals.mean()) if len(vals) else None

    def finite_median(name):
        vals = df[name].replace([np.inf, -np.inf], np.nan).dropna()
        return float(vals.median()) if len(vals) else None

    summary = {
        "num_rows": int(len(df)),
        "num_unique_observations": int(df[["video_idx", "chunk_idx"]].drop_duplicates().shape[0])
        if {"video_idx", "chunk_idx"}.issubset(df.columns)
        else None,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "raw_csv": raw_path,
        "npz": npz_path,
        "per_layer_summary_csv": layer_path,
        "mean_local_current_share": finite_mean("local_current_share"),
        "mean_global_current_share": finite_mean("global_current_share"),
        "mean_current_token_fraction": finite_mean("current_token_fraction")
        if "current_token_fraction" in df.columns
        else None,
        "mean_local_current_share_minus_token_fraction": finite_mean("local_current_share_minus_token_fraction")
        if "local_current_share_minus_token_fraction" in df.columns
        else None,
        "mean_global_current_share_minus_token_fraction": finite_mean("global_current_share_minus_token_fraction")
        if "global_current_share_minus_token_fraction" in df.columns
        else None,
        "mean_global_minus_local_current_share": finite_mean("global_minus_local_current_share"),
        "median_local_current_to_prev_mass_ratio": finite_median("local_current_to_prev_mass_ratio"),
        "median_global_current_to_prev_mass_ratio": finite_median("global_current_to_prev_mass_ratio"),
        "median_local_current_to_prev_per_token_ratio": finite_median("local_current_to_prev_per_token_ratio"),
        "median_global_current_to_prev_per_token_ratio": finite_median("global_current_to_prev_per_token_ratio"),
        "mean_prev_visual_tokens": finite_mean("prev_visual_tokens"),
        "mean_current_chunk_tokens": finite_mean("current_chunk_tokens"),
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_specs = [
            ("local_current_share", "Local query: attention share to latest chunk", "local_current_share_heatmap.png"),
            ("global_current_share", "Global query: attention share to latest chunk", "global_current_share_heatmap.png"),
            (
                "global_minus_local_current_share",
                "Global - local latest-chunk share",
                "global_minus_local_current_share_heatmap.png",
            ),
        ]
        for field, title, filename in plot_specs:
            plt.figure(figsize=(12, 7))
            im = plt.imshow(arrays[field], aspect="auto", cmap="coolwarm")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Head")
            plt.ylabel("Layer")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, filename), dpi=200)
            plt.close()
    except Exception as exc:
        summary["plot_error"] = str(exc)

    return summary


def run_analysis(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"

    print(f"Loading model: {model_path}")
    model, processor = build_observed_model(args, model_path)
    print(f"Model loaded. kv_size={args.kv_size}, compress_mode={args.compress_mode}")

    with open(anno_path) as f:
        anno = json.load(f)
    if args.num_videos:
        anno = anno[: args.num_videos]

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/head_obs_prev_current_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

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

        for question_idx, sample in enumerate(video_sample["conversations"]):
            if "end_time" in sample:
                end_frame_idx = math.ceil(sample["end_time"] * args.sample_fps)
            else:
                end_frame_idx = len(video_tensor)

            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                if next_encode_end <= current_frame_idx:
                    break

                video_chunk = video_tensor[current_frame_idx:next_encode_end]
                pre_lens = model._get_cache_seq_len_per_layer()
                model.encode_video_chunk(video_chunk)
                post_lens = model._get_cache_seq_len_per_layer()
                model.set_current_chunk_bounds(
                    pre_lens,
                    post_lens,
                    video_idx=video_idx,
                    chunk_idx=chunk_idx,
                    question_idx=question_idx,
                    frame_start=current_frame_idx,
                    frame_end=next_encode_end,
                    task=sample.get("task", "Unknown"),
                )
                current_frame_idx = next_encode_end
                chunk_idx += 1
                model.predict_and_compress()

            if "choices" in sample:
                choices = sample["choices"]
                answer = sample.get("answer")
                if answer is None:
                    answer = choices[0]
                correct_choice = analyzer.choice_letters[choices.index(answer)]
                analyzer.video_close_qa(sample["question"], choices, correct_choice)

    summary = aggregate_and_save(model.chunk_obs, args.save_dir, model.num_layers)
    print("\n=== Previous cache vs latest chunk attention ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if model.chunk_obs:
        import pandas as pd

        df = pd.DataFrame(model.chunk_obs)
        head_mean = (
            df.groupby(["layer", "head"])[
                [
                    "local_current_share",
                    "global_current_share",
                    "global_minus_local_current_share",
                    "global_current_to_prev_per_token_ratio",
                ]
            ]
            .mean()
            .reset_index()
        )
        print("\nTop heads most biased to latest chunk under local query:")
        for row in head_mean.sort_values("local_current_share", ascending=False).head(10).itertuples(index=False):
            print(
                f"  L{int(row.layer):2d} H{int(row.head):2d}: "
                f"local_current={row.local_current_share:.3f}, "
                f"global_current={row.global_current_share:.3f}, "
                f"g-l={row.global_minus_local_current_share:+.3f}"
            )
        print("\nTop heads most biased to previous cache under global query:")
        for row in head_mean.sort_values("global_current_share", ascending=True).head(10).itertuples(index=False):
            print(
                f"  L{int(row.layer):2d} H{int(row.head):2d}: "
                f"local_current={row.local_current_share:.3f}, "
                f"global_current={row.global_current_share:.3f}, "
                f"g-l={row.global_minus_local_current_share:+.3f}, "
                f"global_per_token_ratio={row.global_current_to_prev_per_token_ratio:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/observations/obs_prev_current_chunk_attention")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=4)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    run_analysis(parser.parse_args())
