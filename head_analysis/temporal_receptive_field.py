"""
Head 时间感受野分析 — 编码阶段，不需要伪查询，不需要任务标签

每次 encode_video_chunk 之后，新 chunk 的 Q attend 整个 KV cache。
对每个头，测它对新 token vs 旧 token 的注意力分布：
  - recent_ratio = 对最近编码的 chunk 的注意力 / 对所有 visual token 的注意力
  - 高 → 近期偏好的头
  - 低 → 看旧信息的头
"""

import os, sys, json, math, argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA


class ReceptiveFieldVQA(BaseVQA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_stats = []        # 每次 chunk 编码后的头级统计
        self._chunk_vstart = []      # 每次 chunk 编码前的 visual token 边界

    @torch.inference_mode()
    def analyze_a_video(self, video_sample, encode_chunk_size=16):
        video_path = video_sample['video_path']
        if video_path.endswith('.npy'):
            video = self.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            vfps = video_sample.get('fps', None)
            if vfps is None: raise ValueError(f"video_fps required: {video_path}")
            video = self.load_video_frames(video_path, vfps, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        else:
            video = self.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self._chunk_vstart = [0]  # 第一个 chunk 起始在 visual_start

        current_frame_idx = 0

        for sample in tqdm(video_sample['conversations'], desc="Q", leave=False):
            question = sample['question']
            answer = sample.get('answer')

            if 'end_time' in sample:
                end_fidx = math.ceil(sample['end_time'] * self.sample_fps)
            else:
                end_fidx = len(video_tensor)

            while current_frame_idx < end_fidx:
                next_end = min(current_frame_idx + encode_chunk_size, end_fidx)
                if next_end > current_frame_idx:
                    self.qa_model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                    current_frame_idx = next_end

                    # --- 编码后立即测时间感受野 ---
                    self._measure_receptive_field()

                    self._chunk_vstart.append(
                        self.qa_model.kv_cache[0][0].shape[2] - self.qa_model.visual_start_idx)

                    # 不压缩（保留所有 token 确保感受野干净）
                    # self.qa_model.predict_and_compress()

            if 'choices' not in sample:
                continue

            choices = sample['choices']
            if answer is None:
                answer = choices[0]
            correct_choice = self.choice_letters[choices.index(answer)]
            self.video_close_qa(question, choices, correct_choice)

    @torch.inference_mode()
    def _measure_receptive_field(self):
        """新 chunk 刚编码完，用它的 embedding 当 query，测每头对旧 vs 新 token 的注意力"""

        # 拿新 chunk 的最后几个 token 当 query（最近的帧最有代表性）
        kv_cache = self.qa_model.kv_cache
        visual_start = self.qa_model.visual_start_idx

        # 当前 KV cache 总长度
        total_len = kv_cache[0][0].shape[2]
        n_visual = total_len - visual_start
        if n_visual < 200:  # 太早，视觉 token 不够
            return

        # 当前 chunk 的大小（最近一次 encode 增加的 token 数）
        if len(self._chunk_vstart) > 0:
            prev_visual = self._chunk_vstart[-1]
        else:
            prev_visual = 0

        chunk_visual_size = n_visual - prev_visual
        if chunk_visual_size <= 0:
            return

        # 取新 chunk 最后几个 token 当 query
        query_start = max(visual_start, total_len - min(chunk_visual_size, 50))
        query_end = total_len

        # 对每层每头独立计算
        pos_cache = self.qa_model._position_ids_cache

        for layer_idx in range(self.qa_model.num_layers):
            # 手动算该层的 attention
            k_layer, v_layer = kv_cache[layer_idx]
            seq_len = k_layer.shape[2]

            # Query: 新 chunk 最后几个 token（取它们的 key 当 query 算 self-attention）
            # 简化：取该层新 token 对所有旧 token 的平均 attention
            # 用 full attention 计算（单层）

            # 获取该层的 attention（通过 language_model 单层 forward）
            # 直接调用 _compute_attention_scores_manually 太重，
            # 用简化版：拿新 token 的位置，构造 query，算 attention

            if layer_idx >= len(pos_cache) or pos_cache[layer_idx] is None:
                continue

            n_q = query_end - query_start
            if n_q <= 0:
                continue

            # 取最后 n_q 个 token 当 query
            k = k_layer[0, :, :, :]  # [kv_heads, seq_len, head_dim]
            v = v_layer[0, :, :, :]

            # 取 query 部分的 key 当 query（self-attention 风格）
            q = k[:, query_start:query_end, :]  # [kv_heads, n_q, head_dim]

            # 简化：取 query 的最后 1 个 token，对全序列算点积
            q_last = q[:, -1:, :]  # [kv_heads, 1, head_dim]

            # attention = softmax(q × k^T / sqrt(d))
            d = k.shape[-1]
            scores = torch.matmul(q_last.float(), k.float().transpose(-2, -1)) / (d ** 0.5)
            # 因果 mask 不需要（encode 已做完）
            attn = torch.softmax(scores, dim=-1)  # [kv_heads, 1, seq_len]

            # attn 聚合到原来的 head 数 (GQA: 4 KV heads → 28 heads)
            # 简化：取 KV head 的 attention 直接按头统计
            attn = attn.squeeze(1)  # [kv_heads, seq_len]

            for kv_head in range(attn.shape[0]):
                head_attn = attn[kv_head]  # [seq_len]

                # 只取 visual token 部分
                vs_attn = head_attn[visual_start:seq_len]
                total_vs = vs_attn.sum().item()
                if total_vs == 0:
                    continue

                # 新 chunk 的 token = 最后 chunk_visual_size 个
                recent_ratio = vs_attn[-chunk_visual_size:].sum().item() / total_vs

                self.field_stats.append({
                    'layer': layer_idx,
                    'kv_head': kv_head,
                    'chunk_idx': len(self._chunk_vstart),
                    'recent_ratio': recent_ratio,
                    'n_visual': n_visual,
                    'chunk_size': chunk_visual_size,
                })


def run(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"

    device = f"cuda:{args.device}"
    print(f"Loading model: {model_path} on {device}")
    model, processor = load_model(
        model_path, kv_size=100000, streaming=True,
        sample_fps=args.sample_fps, compress_mode='streamingvlm', device=device,
    )
    # 关压缩
    model.predict_and_compress = lambda: None

    with open(anno_path) as f:
        anno = json.load(f)

    if args.num_videos:
        selected = anno[:args.num_videos]
    else:
        selected = anno[:30]

    print(f"Processing {len(selected)} videos...")

    save_dir = args.save_dir or "results/head_analysis/receptive_field"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = ReceptiveFieldVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )
    analyzer.analyze(debug=False)

    stats = analyzer.field_stats
    print(f"\nField measurements: {len(stats)}")

    if not stats:
        print("WARNING: No data.")
        return

    num_layers = model.num_layers
    max_kv_head = max(s['kv_head'] for s in stats)
    n_kv_heads = max_kv_head + 1  # 4 for Qwen2.5-VL-7B

    # 聚合：每层每 KV 头平均 recent_ratio
    agg_sum = np.zeros((num_layers, n_kv_heads))
    agg_cnt = np.zeros((num_layers, n_kv_heads))

    for s in stats:
        l, h = s['layer'], s['kv_head']
        agg_sum[l, h] += s['recent_ratio']
        agg_cnt[l, h] += 1

    recent_ratio = np.where(agg_cnt > 0, agg_sum / agg_cnt, 0.5)

    np.savez(os.path.join(save_dir, "receptive_field.npz"),
             recent_ratio=recent_ratio, count=agg_cnt,
             num_layers=num_layers, n_kv_heads=n_kv_heads)
    print(f"Saved to {save_dir}/receptive_field.npz")

    # 输出
    print(f"\n=== Per-Layer Per-KV-Head Recent Ratio ===")
    print(f"{'Layer':<6}", end="")
    for h in range(n_kv_heads):
        print(f" {'KV'+str(h):>10}", end="")
    print(f" {'LayerAvg':>10}")
    print("-" * (6 + 11 * (n_kv_heads + 1)))

    for l in range(num_layers):
        if agg_cnt[l].sum() == 0:
            continue
        print(f"L{l:<4}", end="")
        layer_vals = []
        for h in range(n_kv_heads):
            if agg_cnt[l, h] > 0:
                v = recent_ratio[l, h]
                layer_vals.append(v)
                print(f" {v:10.4f}", end="")
            else:
                print(f" {'-':>10}", end="")
        print(f" {np.mean(layer_vals):10.4f}" if layer_vals else "")

    # Top KV heads
    print(f"\n=== Most Recent-biased KV heads (高 recent_ratio) ===")
    flat = [(l, h, float(recent_ratio[l,h]), int(agg_cnt[l,h]))
            for l in range(num_layers) for h in range(n_kv_heads) if agg_cnt[l,h] > 10]
    flat.sort(key=lambda x: -x[2])
    for l, h, r, c in flat[:10]:
        print(f"  L{l:2d} KV{h}: recent_ratio={r:.3f}  (n={c})")

    print(f"\n=== Most History-biased KV heads (低 recent_ratio) ===")
    flat.sort(key=lambda x: x[2])
    for l, h, r, c in flat[:10]:
        print(f"  L{l:2d} KV{h}: recent_ratio={r:.3f}  (n={c})")

    return recent_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/receptive_field")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=20)
    args = parser.parse_args()
    run(args)
