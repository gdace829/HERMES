"""
Head Analysis: 对比"记忆依赖任务"与"近期依赖任务"上的 attention 模式差异。

使用方法:
    python head_analysis/run_analysis.py \
        --model qwen2.5_vl_7b \
        --kv_size 100000 \
        --compress_mode streamingvlm \
        --sample_fps 0.5

Probe 任务:
    Probe A (记忆依赖): Causal Reasoning + Counting
    Probe B (近期依赖): Clips Summarize + Prospective Reasoning

划分方式:
    按 visual token 数量切窗口 — 最后 N 个 visual token 算"近期"。
    不依赖视频长度和采样率，比例可控。
"""

import os, sys, json, re, math, time, argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA


PROBE_A = {"Causal Reasoning", "Counting"}             # 记忆依赖
PROBE_B = {"Clips Summarize", "Prospective Reasoning"}  # 近期依赖
RECENT_TOKEN_COUNT = 4096  # 最后 N 个 visual token 算"近期"


class HeadAnalysisVQA(BaseVQA):
    """在正常推理后捕获 attention，不改动原有推理逻辑"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_stats = []

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, encode_chunk_size=16):
        video_path = video_sample['video_path']

        if video_path.endswith('.npy'):
            video = self.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            video_fps = video_sample.get('fps', None)
            if video_fps is None:
                raise ValueError(f"video_fps required for image-based video: {video_path}")
            video = self.load_video_frames(video_path, video_fps, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        else:
            video = self.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        current_frame_idx = 0

        for sample in tqdm(video_sample['conversations'], desc=f"Q", leave=False):
            question = sample['question']
            answer = sample['answer']
            task = sample.get('task', 'Unknown')

            if 'end_time' in sample:
                end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)
            else:
                end_frame_idx = len(video_tensor)

            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                if next_encode_end > current_frame_idx:
                    video_chunk = video_tensor[current_frame_idx:next_encode_end]
                    self.qa_model.encode_video_chunk(video_chunk)
                    current_frame_idx = next_encode_end
                    self.qa_model.predict_and_compress()

            if 'choices' in sample:
                choices = sample['choices']
                if answer is None:
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]

                mc_input = self.format_mcqa_prompt(question, choices) if task in (PROBE_A | PROBE_B) else None

                qa_results = self.video_close_qa(question, choices, correct_choice)

                if mc_input is not None:
                    self._capture_attention(mc_input, task)

    @torch.inference_mode()
    def _capture_attention(self, input_text, task):
        """按 visual token 数量划分：最后 RECENT_TOKEN_COUNT 个 visual token 算近期。
        每头算 recent attention / total visual attention 的比例。
        """
        prompt = input_text['prompt']
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        attn_weights = self.qa_model._compute_attention_scores_manually(
            input_ids, self.qa_model.kv_cache)

        visual_start = self.qa_model.visual_start_idx
        pos_cache = self.qa_model._position_ids_cache

        for layer_idx, layer_attn in enumerate(attn_weights):
            if layer_attn.dim() < 4:
                continue

            kv_len = layer_attn.shape[3]
            if kv_len <= visual_start:
                continue

            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cached_kv_len = pos_cache[layer_idx].shape[1]
            else:
                cached_kv_len = kv_len

            n_visual = cached_kv_len - visual_start
            if n_visual <= RECENT_TOKEN_COUNT:
                continue

            # 最后 N 个 visual token 算近期
            recent_mask = torch.zeros(cached_kv_len, dtype=torch.bool, device=self.qa_model.device)
            recent_mask[visual_start + n_visual - RECENT_TOKEN_COUNT:] = True

            attn_per_head = layer_attn[0].mean(dim=1)  # [heads, kv_len]

            for head_idx in range(attn_per_head.shape[0]):
                head_attn = attn_per_head[head_idx]
                visual_attn = head_attn[visual_start:cached_kv_len]

                total = visual_attn.sum().item()
                if total == 0:
                    continue

                recent_ratio = visual_attn[recent_mask[visual_start:]].sum().item() / total

                self.head_stats.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'task': task,
                    'recent_ratio': recent_ratio,
                    'n_visual': n_visual,
                })


def run_analysis(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"

    device = f"cuda:{args.device}" if args.device >= 0 else "cuda"
    print(f"Loading model: {model_path} on {device}")
    model, processor = load_model(
        model_path,
        kv_size=args.kv_size,
        streaming=True,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
        device=device,
    )

    print(f"Loading annotation: {anno_path}")
    with open(anno_path) as f:
        anno = json.load(f)

    n_probe_a = sum(1 for v in anno for c in v['conversations'] if c['task'] in PROBE_A)
    n_probe_b = sum(1 for v in anno for c in v['conversations'] if c['task'] in PROBE_B)
    print(f"Probe A (记忆依赖): {n_probe_a} questions ({', '.join(sorted(PROBE_A))})")
    print(f"Probe B (近期依赖): {n_probe_b} questions ({', '.join(sorted(PROBE_B))})")

    if args.debug:
        videos_a = [v for v in anno if any(c['task'] in PROBE_A for c in v['conversations'])]
        videos_b = [v for v in anno if any(c['task'] in PROBE_B for c in v['conversations'])]
        selected = (videos_a[:2] + videos_b[:2])
        n_a = sum(1 for v in selected for c in v['conversations'] if c['task'] in PROBE_A)
        n_b = sum(1 for v in selected for c in v['conversations'] if c['task'] in PROBE_B)
        print(f"DEBUG: {len(selected)} videos (Probe A: {n_a} Qs, Probe B: {n_b} Qs)")
    else:
        selected = anno
        print(f"Processing all {len(anno)} videos...")

    save_dir = args.save_dir or "/tmp/head_analysis"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = HeadAnalysisVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )

    analyzer.analyze(debug=False)

    stats = analyzer.head_stats
    print(f"\nTotal head stats collected: {len(stats)}")

    if not stats:
        print("WARNING: No attention data collected.")
        return

    num_layers = model.num_layers
    max_head = max(s['head'] for s in stats)
    num_heads = max_head + 1

    agg = {
        'A': {'sum': np.zeros((num_layers, num_heads)), 'count': np.zeros((num_layers, num_heads))},
        'B': {'sum': np.zeros((num_layers, num_heads)), 'count': np.zeros((num_layers, num_heads))},
    }

    for s in stats:
        group = 'A' if s['task'] in PROBE_A else 'B'
        l, h = s['layer'], s['head']
        agg[group]['sum'][l, h] += s['recent_ratio']
        agg[group]['count'][l, h] += 1

    for g in ['A', 'B']:
        mask = agg[g]['count'] > 0
        agg[g]['avg'] = np.where(mask, agg[g]['sum'] / agg[g]['count'], 0)

    recent_A = agg['A']['avg']  # 记忆任务上的 recent ratio（低 = 记忆头）
    recent_B = agg['B']['avg']  # 近期任务上的 recent ratio（高 = 近期头）

    np.savez(os.path.join(save_dir, "head_scores.npz"),
             recent_A=recent_A, recent_B=recent_B,
             num_layers=num_layers, num_heads=num_heads)
    print(f"Saved scores to {save_dir}/head_scores.npz")

    print("\n=== Top Memory Heads (低 recent ratio @ CR+CT) ===")
    flat_mem = [(l, h, float(recent_A[l, h]), float(recent_B[l, h]),
                  float(recent_A[l, h]) - float(recent_B[l, h]))
                 for l in range(num_layers) for h in range(num_heads)
                 if agg['A']['count'][l, h] > 0]
    flat_mem.sort(key=lambda x: x[2])
    for l, h, ra, rb, d in flat_mem[:10]:
        print(f"  L{l:2d} H{h:2d}: rec(A)={ra:.3f}  rec(B)={rb:.3f}  Δ={d:+.3f}")

    print("\n=== Top Recent Heads (高 recent ratio @ CS+PR) ===")
    flat_rec = [(l, h, float(recent_A[l, h]), float(recent_B[l, h]),
                  float(recent_B[l, h]) - float(recent_A[l, h]))
                 for l in range(num_layers) for h in range(num_heads)
                 if agg['B']['count'][l, h] > 0]
    flat_rec.sort(key=lambda x: -x[3])
    for l, h, ra, rb, d in flat_rec[:10]:
        print(f"  L{l:2d} H{h:2d}: rec(A)={ra:.3f}  rec(B)={rb:.3f}  Δ={d:+.3f}")

    return recent_A, recent_B


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_analysis(args)
