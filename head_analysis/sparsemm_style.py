"""
Head Analysis — SparseMM 风格 token 命中

用问题的 end_time 定义"答案时间窗口"，统计每个头在答题时对该窗口 token 的注意力命中率。

指标:
  - answer_hit: 该头对答案窗口内 visual token 的注意力 / 该头对全部 visual token 的注意力
  - 按 CR+CT (记忆) 和 CS+PR (近期) 分别统计，比较差异

用法:
    python head_analysis/sparsemm_style.py \
        --model qwen2.5_vl_7b --kv_size 100000 \
        --compress_mode streamingvlm --sample_fps 0.5 \
        --num_videos 30 --device 0
"""

import os, sys, json, math, argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA

PROBE_A = {"Causal Reasoning", "Counting"}
PROBE_B = {"Clips Summarize", "Prospective Reasoning"}
ANSWER_WINDOW_SEC = 10  # 答案窗口大小（秒）


class HitAnalysisVQA(BaseVQA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_stats = []

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, encode_chunk_size=16):
        video_path = video_sample['video_path']

        if video_path.endswith('.npy'):
            video = self.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            vfps = video_sample.get('fps', None)
            if vfps is None:
                raise ValueError(f"video_fps required: {video_path}")
            video = self.load_video_frames(video_path, vfps, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        else:
            video = self.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        current_frame_idx = 0

        for sample in tqdm(video_sample['conversations'], desc="Q", leave=False):
            question = sample['question']
            answer = sample['answer']
            task = sample.get('task', 'Unknown')

            if 'end_time' in sample:
                end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)
                end_time = sample['end_time']
            else:
                end_frame_idx = len(video_tensor)
                end_time = 999

            while current_frame_idx < end_frame_idx:
                next_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                if next_end > current_frame_idx:
                    self.qa_model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                    current_frame_idx = next_end
                    self.qa_model.predict_and_compress()

            if 'choices' not in sample:
                continue

            choices = sample['choices']
            if answer is None:
                answer = choices[0]
            correct_choice = self.choice_letters[choices.index(answer)]

            mc_input = self.format_mcqa_prompt(question, choices)
            self.video_close_qa(question, choices, correct_choice)

            if task in (PROBE_A | PROBE_B):
                self._record_hit(mc_input, task, end_time)

    @torch.inference_mode()
    def _record_hit(self, input_text, task, end_time):
        """计算每个头对"答案时间窗口"内 visual token 的注意力命中率"""
        prompt = input_text['prompt']
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        attn = self.qa_model._compute_attention_scores_manually(input_ids, self.qa_model.kv_cache)

        visual_start = self.qa_model.visual_start_idx
        pos_cache = self.qa_model._position_ids_cache

        # end_time → M-RoPE t 坐标
        tps = self.sample_fps / 2.0           # t 单位 / 秒
        answer_t = end_time * tps             # 问题时刻的 t 坐标
        window_t = ANSWER_WINDOW_SEC * tps     # 窗口大小 (t 单位)
        answer_t_start = answer_t - window_t

        for layer_idx, layer_attn in enumerate(attn):
            if layer_attn.dim() < 4:
                continue

            kv_len = layer_attn.shape[3]
            if kv_len <= visual_start:
                continue

            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                t_pos = pos_cache[layer_idx][0]
                cached_len = t_pos.shape[0]
            else:
                cached_len = kv_len
                t_pos = torch.arange(cached_len, device=self.qa_model.device)

            n_visual = cached_len - visual_start
            if n_visual <= 0:
                continue

            visual_t = t_pos[visual_start:cached_len]
            answer_mask = (visual_t >= answer_t_start) & (visual_t <= answer_t)
            if answer_mask.sum() == 0:
                continue

            # [heads, n_visual] — mean over query positions
            attn_vis = layer_attn[0, :, :, visual_start:cached_len].mean(dim=1)

            for head_idx in range(attn_vis.shape[0]):
                h = attn_vis[head_idx]
                total = h.sum().item()
                hit = h[answer_mask].sum().item()
                if total == 0:
                    continue

                self.hit_stats.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'task': task,
                    'answer_hit': hit / total,
                    'end_time': end_time,
                })


def run(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"

    device = f"cuda:{args.device}"
    print(f"Loading model: {model_path} on {device}")
    model, processor = load_model(
        model_path, kv_size=args.kv_size, streaming=True,
        sample_fps=args.sample_fps, compress_mode=args.compress_mode, device=device,
    )

    with open(anno_path) as f:
        anno = json.load(f)

    n_a = sum(1 for v in anno for c in v['conversations'] if c['task'] in PROBE_A)
    n_b = sum(1 for v in anno for c in v['conversations'] if c['task'] in PROBE_B)
    print(f"Probe A (记忆): {n_a} Qs, Probe B (近期): {n_b} Qs")

    if args.num_videos:
        # 选包含 probe 任务的视频
        selected = [v for v in anno[:args.num_videos * 3]
                     if any(c['task'] in (PROBE_A | PROBE_B) for c in v['conversations'])]
        selected = selected[:args.num_videos]
    else:
        selected = [v for v in anno if any(c['task'] in (PROBE_A | PROBE_B) for c in v['conversations'])]

    n_sel_a = sum(1 for v in selected for c in v['conversations'] if c['task'] in PROBE_A)
    n_sel_b = sum(1 for v in selected for c in v['conversations'] if c['task'] in PROBE_B)
    print(f"Selected {len(selected)} videos: A={n_sel_a} Qs, B={n_sel_b} Qs")

    save_dir = args.save_dir or "results/head_analysis/hit"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = HitAnalysisVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )
    analyzer.analyze(debug=False)

    stats = analyzer.hit_stats
    print(f"\nTotal hits collected: {len(stats)}")

    if not stats:
        print("WARNING: No hits collected.")
        return

    num_layers = model.num_layers
    max_head = max(s['head'] for s in stats)
    num_heads = max_head + 1

    # 聚合
    agg = {
        'A': {'sum': np.zeros((num_layers, num_heads)), 'cnt': np.zeros((num_layers, num_heads))},
        'B': {'sum': np.zeros((num_layers, num_heads)), 'cnt': np.zeros((num_layers, num_heads))},
    }

    for s in stats:
        g = 'A' if s['task'] in PROBE_A else 'B'
        l, h = s['layer'], s['head']
        agg[g]['sum'][l, h] += s['answer_hit']
        agg[g]['cnt'][l, h] += 1

    hit_A = np.where(agg['A']['cnt'] > 0, agg['A']['sum'] / agg['A']['cnt'], 0)
    hit_B = np.where(agg['B']['cnt'] > 0, agg['B']['sum'] / agg['B']['cnt'], 0)

    retrieval = (hit_A + hit_B) / 2
    mem_spec = hit_A - hit_B

    np.savez(os.path.join(save_dir, "hit_scores.npz"),
             hit_A=hit_A, hit_B=hit_B,
             retrieval=retrieval, mem_spec=mem_spec,
             num_layers=num_layers, num_heads=num_heads)
    print(f"Saved to {save_dir}/hit_scores.npz")

    # 输出
    print("\n=== Top Retrieval Heads (高 answer_hit) ===")
    flat = [(l, h, float(retrieval[l, h]), float(mem_spec[l, h]))
            for l in range(num_layers) for h in range(num_heads)
            if agg['A']['cnt'][l, h] + agg['B']['cnt'][l, h] > 2]
    flat.sort(key=lambda x: -x[2])
    for l, h, r, m in flat[:10]:
        print(f"  L{l:2d} H{h:2d}: retrieval={r:.3f}  mem_spec={m:+.3f}")

    print("\n=== Top Memory-Specialized (高 mem_spec) ===")
    flat.sort(key=lambda x: -x[3])
    for l, h, r, m in flat[:10]:
        print(f"  L{l:2d} H{h:2d}: retrieval={r:.3f}  mem_spec={m:+.3f}")

    print("\n=== Per-layer retrieval ===")
    for l in range(num_layers):
        mask = (agg['A']['cnt'][l] + agg['B']['cnt'][l]) > 2
        if mask.any():
            print(f"  L{l:2d}: retrieval={retrieval[l][mask].mean():.3f}  "
                  f"mem_spec={mem_spec[l][mask].mean():+.3f}")

    return hit_A, hit_B, retrieval, mem_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=100000)
    parser.add_argument("--compress_mode", type=str, default="streamingvlm")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/hit")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=30)
    args = parser.parse_args()

    run(args)
