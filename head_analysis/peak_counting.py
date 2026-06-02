"""
Counting 题的峰值检测：答案数字 = "应该有几次事件"

不用标注。对每个头，从 attention 时间曲线中检测峰值数，
与正确答案比：峰值数量越接近答案 = 该头计数能力越强。

用法:
    python head_analysis/peak_counting.py \
        --model qwen2.5_vl_7b --kv_size 6000 --compress_mode hermes \
        --num_videos 20 --device 0
"""

import os, sys, json, math, argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA


class PeakCountingVQA(BaseVQA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_stats = []

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
        current_frame_idx = 0

        for sample in tqdm(video_sample['conversations'], desc="Q", leave=False):
            task = sample.get('task', '')

            if 'end_time' in sample:
                end_frame_idx = math.ceil(sample['end_time'] * self.sample_fps)
            else:
                end_frame_idx = len(video_tensor)
            end_time = sample.get('end_time', 0)

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
            if answer is None: answer = choices[0]
            correct_choice = self.choice_letters[choices.index(answer)]

            mc_input = self.format_mcqa_prompt(sample['question'], choices)
            qa_result = self.video_close_qa(sample['question'], choices, correct_choice)

            if task not in ('Counting', 'Causal Reasoning', 'Clips Summarize',
                            'Prospective Reasoning'):
                continue

            # 提取答案数字（CT 题答案通常是数字）
            correct_num = self._parse_number(answer)
            self._record_peaks(mc_input, task, end_time, correct_num)

    def _parse_number(self, answer_text):
        """从答案文本提取数字，失败返回 None"""
        import re
        # 先找阿拉伯数字
        m = re.search(r'\b(\d+)\b', answer_text)
        if m: return int(m.group(1))
        # 再找英文数字
        words = {'one':1,'two':2,'three':3,'four':4,'five':5,
                 'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
                 'zero':0, 'no':0, 'none':0}
        for w, v in words.items():
            if w in answer_text.lower():
                return v
        return None

    @torch.inference_mode()
    def _record_peaks(self, input_text, task, end_time, answer_number):
        """提取每个头的 attention 时间曲线，检测峰值数"""
        prompt = input_text['prompt']
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        attn = self.qa_model._compute_attention_scores_manually(
            input_ids, self.qa_model.kv_cache)

        visual_start = self.qa_model.visual_start_idx
        pos_cache = self.qa_model._position_ids_cache

        for layer_idx, layer_attn in enumerate(attn):
            if layer_attn.dim() < 4: continue
            kv_len = layer_attn.shape[3]
            if kv_len <= visual_start: continue

            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                t_pos = pos_cache[layer_idx][0]
                cached_len = t_pos.shape[0]
            else:
                cached_len = kv_len
                t_pos = torch.arange(cached_len, device=self.qa_model.device)

            n_vis = cached_len - visual_start
            if n_vis < 100: continue  # 太短没意义

            # [heads, n_visual]
            hv = layer_attn[0, :, :, visual_start:cached_len].mean(dim=1)
            visual_t = t_pos[visual_start:cached_len].cpu().numpy()

            for head_idx in range(hv.shape[0]):
                h = hv[head_idx].cpu().numpy()
                # 归一化
                h = h / (h.sum() + 1e-8)

                # 检测峰值 (prominence=0.3*std 过滤噪声)
                prominence = max(0.3 * h.std(), h.mean() * 0.3)
                peaks, props = find_peaks(h, prominence=prominence, distance=5)
                n_peaks = len(peaks)

                # 峰值时间分布
                if len(peaks) > 0:
                    peak_times = visual_t[peaks]
                else:
                    peak_times = np.array([])

                self.peak_stats.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'task': task,
                    'n_peaks': n_peaks,
                    'n_visual': n_vis,
                    'answer_number': answer_number,
                    'peak_times': peak_times.tolist(),
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

    # 选包含 CT+CR 的视频
    targets = {'Counting', 'Causal Reasoning'}
    if args.num_videos:
        selected = [v for v in anno if any(c['task'] in targets for c in v['conversations'])]
        selected = selected[:args.num_videos]
    else:
        selected = [v for v in anno if any(c['task'] in targets for c in v['conversations'])]

    n_ct = sum(1 for v in selected for c in v['conversations'] if c['task'] in targets)
    print(f"{len(selected)} videos, {n_ct} target questions")

    save_dir = args.save_dir or "results/head_analysis/peaks"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = PeakCountingVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )
    analyzer.analyze(debug=False)

    stats = analyzer.peak_stats
    print(f"\nPeak stats collected: {len(stats)}")

    # 选有数字标注的
    ct_stats = [s for s in stats if s['answer_number'] is not None]
    print(f"With answer number: {len(ct_stats)}")

    if not ct_stats:
        print("WARNING: No answer-number data. Only non-counting answers collected.")
        return

    # 聚合: 峰值误差 = |n_peaks - answer_number|
    num_layers, num_heads = model.num_layers, max(s['head'] for s in stats) + 1
    agg_err = defaultdict(list)
    agg_npeak = defaultdict(list)

    for s in ct_stats:
        key = (s['layer'], s['head'])
        err = abs(s['n_peaks'] - s['answer_number'])
        agg_err[key].append(err)
        agg_npeak[key].append(s['n_peaks'])

    # 用峰值误差 (越小=计数越准) 和峰值数均值
    err_mat = np.full((num_layers, num_heads), np.nan)
    peak_mat = np.full((num_layers, num_heads), np.nan)
    count_mat = np.zeros((num_layers, num_heads), dtype=int)

    for (l, h), errs in agg_err.items():
        err_mat[l, h] = np.mean(errs)
        peak_mat[l, h] = np.mean(agg_npeak[(l, h)])
        count_mat[l, h] = len(errs)

    np.savez(os.path.join(save_dir, "peak_scores.npz"),
             peak_error=err_mat, n_peaks=peak_mat, count=count_mat,
             num_layers=num_layers, num_heads=num_heads)
    print(f"Saved to {save_dir}/peak_scores.npz")

    # 找最低误差的头 (计数最准)
    print("\n=== Top Counting Heads (最低峰值误差) ===")
    flat = [(l, h, err_mat[l,h], peak_mat[l,h], count_mat[l,h])
            for l in range(num_layers) for h in range(num_heads)
            if count_mat[l,h] >= 3]
    flat.sort(key=lambda x: x[2])
    for l, h, e, p, c in flat[:10]:
        print(f"  L{l:2d} H{h:2d}: err={e:.2f}  avg_peaks={p:.1f}  (n={c})")

    print("\n=== Per-layer avg peak error ===")
    for l in range(num_layers):
        m = count_mat[l] >= 3
        if m.any():
            print(f"  L{l:2d}: err={np.nanmean(err_mat[l][m]):.2f}")

    return err_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/peaks")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=20)
    args = parser.parse_args()
    run(args)
