"""
Counting (CT) 题的 token 命中分析

方法: 取答对的 CT 题，用模型自身的 attention 定位"答案帧"，
      然后统计每个头是否关注了这些帧。

不需要任何人工标注。模型答对 → 它看的地方就是答案所在。
"""

import os, sys, json, math, argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA


class CountingHitVQA(BaseVQA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_hits = []       # 每个正确 CT 题，每个 head 的命中率
        self.correct_ct_count = 0

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
            task = sample.get('task', 'Unknown')
            if task != 'Counting':
                # 非 CT 题：照常跑，但不记录
                if 'end_time' in sample:
                    end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)
                else:
                    end_frame_idx = len(video_tensor)
                while current_frame_idx < end_frame_idx:
                    next_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                    if next_end > current_frame_idx:
                        self.qa_model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                        current_frame_idx = next_end
                        self.qa_model.predict_and_compress()
                if 'choices' in sample:
                    choices = sample['choices']
                    answer = sample.get('answer')
                    if answer is None:
                        answer = choices[0]
                    correct_choice = self.choice_letters[choices.index(answer)]
                    self.video_close_qa(sample['question'], choices, correct_choice)
                continue

            # ---- CT 题：捕获 attention ----
            end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)

            while current_frame_idx < end_frame_idx:
                next_end = min(current_frame_idx + encode_chunk_size, end_frame_idx)
                if next_end > current_frame_idx:
                    self.qa_model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                    current_frame_idx = next_end
                    self.qa_model.predict_and_compress()

            if 'choices' not in sample:
                continue

            choices = sample['choices']
            answer = sample.get('answer')
            if answer is None:
                answer = choices[0]
            correct_choice = self.choice_letters[choices.index(answer)]

            mc_input = self.format_mcqa_prompt(sample['question'], choices)
            qa_result = self.video_close_qa(sample['question'], choices, correct_choice)

            # 只记录答对的 CT 题
            if qa_result.get('acc', 0) != 1.0:
                continue

            self.correct_ct_count += 1
            self._record_token_hits(mc_input, sample.get('end_time', 0))

    @torch.inference_mode()
    def _record_token_hits(self, input_text, end_time):
        """答对的 CT 题：用 attention 定位答案帧，测每个头的命中率"""
        prompt = input_text['prompt']
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        attn_weights = self.qa_model._compute_attention_scores_manually(
            input_ids, self.qa_model.kv_cache)

        visual_start = self.qa_model.visual_start_idx
        pos_cache = self.qa_model._position_ids_cache

        # --- Step 1: 聚合所有层的 attention 得到"ensemble 答案定位" ---
        ensemble_attn = None  # [n_visual] 聚合后的重要性

        for layer_idx, layer_attn in enumerate(attn_weights):
            if layer_attn.dim() < 4:
                continue
            kv_len = layer_attn.shape[3]
            if kv_len <= visual_start:
                continue

            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cached_len = pos_cache[layer_idx].shape[1]
            else:
                cached_len = kv_len

            n_visual = cached_len - visual_start
            if n_visual <= 0:
                continue

            # [heads, n_visual]
            lv = layer_attn[0, :, :, visual_start:cached_len].mean(dim=(0, 1))
            if ensemble_attn is None:
                ensemble_attn = lv.clone()
            else:
                ensemble_attn += lv

        if ensemble_attn is None or ensemble_attn.sum() == 0:
            return

        ensemble_attn = ensemble_attn / ensemble_attn.sum()

        # --- Step 2: Top-20% attention = "答案帧" ---
        k = max(1, int(len(ensemble_attn) * 0.2))
        _, top_idx = torch.topk(ensemble_attn, k)
        answer_mask = torch.zeros(len(ensemble_attn), dtype=torch.bool, device=ensemble_attn.device)
        answer_mask[top_idx] = True

        # --- Step 3: 每个头的命中率 ---
        for layer_idx, layer_attn in enumerate(attn_weights):
            if layer_attn.dim() < 4:
                continue
            kv_len = layer_attn.shape[3]
            if kv_len <= visual_start:
                continue
            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cached_len = pos_cache[layer_idx].shape[1]
            else:
                cached_len = kv_len
            if cached_len - visual_start != len(answer_mask):
                continue  # 长度不匹配，跳过

            # [heads, n_visual]
            hv = layer_attn[0, :, :, visual_start:cached_len].mean(dim=1)
            for head_idx in range(hv.shape[0]):
                h = hv[head_idx]
                total = h.sum().item()
                if total == 0:
                    continue
                hit = h[answer_mask].sum().item() / total  # 命中率
                self.token_hits.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'hit_rate': hit,
                    'answer_mask_size': k,
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

    n_ct = sum(1 for v in anno for c in v['conversations'] if c['task'] == 'Counting')
    print(f"CT questions total: {n_ct}")

    if args.num_videos:
        selected = [v for v in anno[:args.num_videos*2]
                     if any(c['task'] == 'Counting' for c in v['conversations'])]
        selected = selected[:max(1, args.num_videos)]
    else:
        selected = [v for v in anno if any(c['task'] == 'Counting' for c in v['conversations'])]

    n_sel = sum(1 for v in selected for c in v['conversations'] if c['task'] == 'Counting')
    print(f"Selected {len(selected)} videos, {n_sel} CT questions")

    save_dir = args.save_dir or "results/head_analysis/hit_ct"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = CountingHitVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )
    analyzer.analyze(debug=False)

    hits = analyzer.token_hits
    n_correct = analyzer.correct_ct_count
    print(f"\nCorrect CT answers: {n_correct}, Token hits: {len(hits)}")

    if not hits:
        print("WARNING: No hits (no correct CT answers collected).")
        return

    num_layers = model.num_layers
    max_head = max(h['head'] for h in hits)
    num_heads = max_head + 1

    # 存 raw — SparseMM 风格：每头一个列表，每个样本一条记录
    raw_dict = defaultdict(list)
    for h in hits:
        raw_dict[f"{h['layer']}-{h['head']}"].append(h['hit_rate'])

    import json
    raw_path = os.path.join(save_dir, "ct_hit_raw.json")
    with open(raw_path, 'w') as f:
        json.dump(raw_dict, f)
    print(f"Raw hit data saved to {raw_path} ({len(raw_dict)} heads, {len(hits)} records)")

    # 聚合统计
    agg_sum = np.zeros((num_layers, num_heads))
    agg_cnt = np.zeros((num_layers, num_heads))

    for h in hits:
        l, he = h['layer'], h['head']
        agg_sum[l, he] += h['hit_rate']
        agg_cnt[l, he] += 1

    hit_rate = np.where(agg_cnt > 0, agg_sum / agg_cnt, 0)

    np.savez(os.path.join(save_dir, "ct_hit_scores.npz"),
             hit_rate=hit_rate, count=agg_cnt,
             num_layers=num_layers, num_heads=num_heads)
    print(f"Saved to {save_dir}/ct_hit_scores.npz")

    print("\n=== Top CT-Hit Heads (高命中率) ===")
    flat = [(l, h, float(hit_rate[l, h]), int(agg_cnt[l, h]))
            for l in range(num_layers) for h in range(num_heads) if agg_cnt[l, h] > 0]
    flat.sort(key=lambda x: -x[2])
    for l, h, r, c in flat[:10]:
        print(f"  L{l:2d} H{h:2d}: hit_rate={r:.3f}  (n={c})")

    print("\n=== Per-layer average hit_rate ===")
    for l in range(num_layers):
        mask = agg_cnt[l] > 0
        if mask.any():
            print(f"  L{l:2d}: hit_rate={hit_rate[l][mask].mean():.3f}  n={int(agg_cnt[l][mask].sum())}")

    return hit_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/hit_ct")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=20)
    args = parser.parse_args()

    run(args)
