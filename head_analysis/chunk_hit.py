"""
Chunk-level hit analysis — SparseMM 风格

CT 题答对时: 答案数字 = 应有 N 个事件 chunk
取每个头对每个 chunk 的注意力 → 头部 top-N chunk = 该头的"命中 chunk"
命中 chunk 和 ensemble top-N 做 overlap = 该头的命中率

纯外挂，不改原代码。输出 SparseMM 风格的 raw JSON。
"""

import os, sys, json, math, argparse, re
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA


class ChunkHitVQA(BaseVQA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_records = []       # 每个正确 CT 题一条记录
        self._chunk_vstart = []     # 每个 chunk 编码前的 visual_start
        # 关掉压缩：CT 题需要完整 chunk 信息，挂 no-op 到模型上
        self.qa_model.predict_and_compress = lambda: None

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
        self._chunk_vstart = [self.qa_model.visual_start_idx]  # 第 0 个 chunk 的起始
        current_frame_idx = 0

        for sample in tqdm(video_sample['conversations'], desc="Q", leave=False):
            task = sample.get('task', '')
            question = sample['question']

            if 'end_time' in sample:
                end_fidx = math.ceil(sample['end_time'] * self.sample_fps)
                end_time = sample['end_time']
            else:
                end_fidx = len(video_tensor)
                end_time = 999

            # 记录这个 question 开始时的 chunk 数
            chunk_start = len(self._chunk_vstart)

            while current_frame_idx < end_fidx:
                next_end = min(current_frame_idx + encode_chunk_size, end_fidx)
                if next_end > current_frame_idx:
                    self.qa_model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                    current_frame_idx = next_end
                    # 记录当前 visual token 总数
                    self._chunk_vstart.append(self.qa_model.kv_cache[0][0].shape[2])
                    self.qa_model.predict_and_compress()

            chunk_end = len(self._chunk_vstart)  # 这个 question 覆盖的 chunk 范围

            if 'choices' not in sample:
                continue

            choices = sample['choices']
            answer = sample.get('answer')
            if answer is None: answer = choices[0]
            correct_choice = self.choice_letters[choices.index(answer)]

            mc_input = self.format_mcqa_prompt(question, choices)
            qa_result = self.video_close_qa(question, choices, correct_choice)
            is_correct = qa_result.get('acc', 0) == 1.0

            if task not in ('Counting', 'Causal Reasoning'):
                continue

            # 提取答案数字
            answer_num = self._parse_number(answer)

            # 跳过 answer=0 的 CT 题（没有事件，top-N 无意义）
            if task == 'Counting' and answer_num is not None and answer_num == 0:
                continue

            # 分析每个 chunk 的注意力
            self._analyze_chunk_hits(mc_input, chunk_start, chunk_end, task, answer_num, is_correct)

    def _parse_number(self, text):
        m = re.search(r'\b(\d+)\b', text)
        if m: return int(m.group(1))
        for w, v in {'one':1,'two':2,'three':3,'four':4,'five':5,
                     'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
                     'zero':0,'no':0,'none':0}.items():
            if w in text.lower(): return v
        return None

    @torch.inference_mode()
    def _analyze_chunk_hits(self, input_text, chunk_start, chunk_end, task, answer_num, is_correct):
        """取 question attention，按 chunk 聚合，找到事件 chunk"""
        prompt = input_text['prompt']
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        attn_weights = self.qa_model._compute_attention_scores_manually(
            input_ids, self.qa_model.kv_cache)

        pos_cache = self.qa_model._position_ids_cache

            # --- 计算 chunk 边界 ---
        vs = self.qa_model.visual_start_idx
        cached_len = self.qa_model.kv_cache[0][0].shape[2]  # 第一层的 kv_len
        total_visual = cached_len - vs

        chunk_vstart = [max(0, s - vs) for s in self._chunk_vstart[chunk_start:chunk_end]]
        chunk_vend = chunk_vstart[1:] + [total_visual]
        n_chunks = len(chunk_vstart)

        if n_chunks < 2:
            return

        # --- 每层每头对每个 chunk 的 attention ---
        # 存储为 list of [heads, n_chunks]
        layer_head_chunk = []

        for layer_idx, layer_attn in enumerate(attn_weights):
            if layer_attn.dim() < 4:
                layer_head_chunk.append(None)
                continue
            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cached_len = pos_cache[layer_idx].shape[1]
            else:
                cached_len = layer_attn.shape[3]

            vs = self.qa_model.visual_start_idx
            n_vis = cached_len - vs
            if n_vis <= 0:
                layer_head_chunk.append(None)
                continue

            lv = layer_attn[0, :, :, vs:cached_len].mean(dim=1)  # [heads, n_visual]

            # 按 chunk 聚合
            lhc = torch.zeros((lv.shape[0], n_chunks), device=lv.device)
            for ci in range(n_chunks):
                s, e = chunk_vstart[ci], min(chunk_vend[ci], n_vis)
                if s < e:
                    lhc[:, ci] = lv[:, s:e].sum(dim=1)

            # 归一化
            lhc = lhc / (lhc.sum(dim=1, keepdim=True) + 1e-8)
            layer_head_chunk.append(lhc)

        # --- ensemble: 所有层所有头平均 → 用于选事件 chunk ---
        all_sum = None
        all_cnt = 0
        for lhc in layer_head_chunk:
            if lhc is not None:
                if all_sum is None: all_sum = lhc.sum(dim=0).clone()
                else: all_sum += lhc.sum(dim=0)
                all_cnt += lhc.shape[0]
        if all_sum is None: return

        ensemble_chunk = (all_sum / all_cnt).cpu().numpy()

        # 事件 chunk: top-N
        N = answer_num if answer_num is not None else max(2, n_chunks // 5)
        N = max(1, min(N, n_chunks))
        event_chunks = set(np.argsort(-ensemble_chunk)[:N].tolist())

        # --- 每个 (layer, head) 的 top-K 命中率 ---
        # SparseMM 风格：头自己选 top-K chunk → 看几个命中 event chunk
        K = max(1, N)  # top-K = 答案数字（预期的命中数）
        for layer_idx, lhc in enumerate(layer_head_chunk):
            if lhc is None: continue
            lhc_np = lhc.cpu().numpy()
            for head_idx in range(lhc_np.shape[0]):
                head_chunk_attn = lhc_np[head_idx]
                # 该头自己的 top-K chunk
                topk = np.argpartition(-head_chunk_attn, min(K, n_chunks)-1)[:K]
                # 命中：top-K 中有多少个是事件 chunk
                hits = len(set(topk.tolist()) & event_chunks)
                hit_rate = hits / K

                self.hit_records.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'task': task,
                    'hit_rate': float(hit_rate),
                    'n_hits': hits,
                    'K': K,
                    'answer_num': answer_num,
                    'n_chunks': n_chunks,
                    'is_correct': is_correct,
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

    targets = {'Counting', 'Causal Reasoning'}
    n_target = sum(1 for v in anno for c in v['conversations'] if c['task'] in targets)
    print(f"Target questions (CT+CR): {n_target}")

    if args.num_videos:
        selected = [v for v in anno if any(c['task'] in targets for c in v['conversations'])]
        selected = selected[:args.num_videos]
    else:
        selected = [v for v in anno if any(c['task'] in targets for c in v['conversations'])]

    n_sel = sum(1 for v in selected for c in v['conversations'] if c['task'] in targets)
    print(f"Selected {len(selected)} videos, {n_sel} target Qs")

    save_dir = args.save_dir or "results/head_analysis/chunk_hit"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = ChunkHitVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )
    analyzer.analyze(debug=False)

    records = analyzer.hit_records
    correct_records = [r for r in records if r['is_correct']]
    print(f"\nTotal records: {len(records)}, Correct: {len(correct_records)}")

    if not correct_records:
        print("WARNING: No correct answers.")
        return

    # --- SparseMM 风格 raw JSON ---
    num_layers = model.num_layers
    max_head = max(r['head'] for r in correct_records)
    num_heads = max_head + 1

    raw_dict = defaultdict(list)
    for r in correct_records:
        key = f"{r['layer']}-{r['head']}"
        raw_dict[key].append(r['hit_rate'])

    raw_path = os.path.join(save_dir, "chunk_hit_raw.json")
    with open(raw_path, 'w') as f:
        json.dump(raw_dict, f)
    print(f"Raw hit data saved to {raw_path} ({len(raw_dict)} heads)")

    # --- 聚合统计 ---
    agg_sum = np.zeros((num_layers, num_heads))
    agg_cnt = np.zeros((num_layers, num_heads))

    for r in correct_records:
        l, h = r['layer'], r['head']
        agg_sum[l, h] += r['hit_rate']
        agg_cnt[l, h] += 1

    hit_rate = np.where(agg_cnt > 0, agg_sum / agg_cnt, 0)

    np.savez(os.path.join(save_dir, "chunk_hit_scores.npz"),
             hit_rate=hit_rate, count=agg_cnt,
             num_layers=num_layers, num_heads=num_heads)
    print(f"Saved to {save_dir}/chunk_hit_scores.npz")

    # 输出
    print("\n=== Top Chunk-Hit Heads (高命中率) ===")
    flat = [(l, h, float(hit_rate[l,h]), int(agg_cnt[l,h]))
            for l in range(num_layers) for h in range(num_heads) if agg_cnt[l,h] >= 3]
    flat.sort(key=lambda x: -x[2])
    for l, h, r, c in flat[:15]:
        print(f"  L{l:2d} H{h:2d}: hit_rate={r:.3f}  (n={c})")

    print("\n=== Per-layer chunk hit rate ===")
    for l in range(num_layers):
        mask = agg_cnt[l] >= 3
        if mask.any():
            print(f"  L{l:2d}: {hit_rate[l][mask].mean():.3f}  (n={int(agg_cnt[l][mask].sum())})")

    return hit_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/chunk_hit")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=20)
    args = parser.parse_args()
    run(args)
