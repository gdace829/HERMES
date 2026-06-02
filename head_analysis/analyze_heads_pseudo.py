"""
Head Analysis via Pseudo-Forward Attention:
对比每个头对 local query 和 global query 的注意力分布。

local query  = "Describe the current scene..." → 短期
global query = "Summarize the video narrative..." → 长期

每个头算: global_attn / (local_attn + global_attn)
  > 0.5 → 长期偏好的头
  < 0.5 → 短期偏好的头

完全外挂，不动一行原代码。
使用方法:
    python head_analysis/analyze_heads_pseudo.py \
        --model qwen2.5_vl_7b --kv_size 100000 \
        --compress_mode streamingvlm --sample_fps 0.5 \
        --num_videos 20 --device 0
"""

import os, sys, json, math, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import QwenVL_Hermes, load_model
from video_qa.base import BaseVQA


class ObservedQwenVL(QwenVL_Hermes):
    """只重写 prune_kv_cache_by_attention 来记录 attention weights"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_obs = []  # 每条记录的 per-head 统计

    def predict_and_compress(self):
        """重写父类方法：在压缩前用 pseudo query 算 attention 做观测（不影响原压缩逻辑）"""
        # 先做观测——用当前的 KV cache 计算 local/global attention
        local_q, global_q = self.predict_next_question()

        local_ids = self.processor.tokenizer(local_q).input_ids
        local_ids = torch.as_tensor([local_ids], device=self.device, dtype=torch.int)
        attn_local = self._compute_attention_scores_manually(local_ids, self.kv_cache)

        global_ids = self.processor.tokenizer(global_q).input_ids
        global_ids = torch.as_tensor([global_ids], device=self.device, dtype=torch.int)
        attn_global = self._compute_attention_scores_manually(global_ids, self.kv_cache)

        self._observe_heads(attn_local, attn_global)

        # 正常压缩
        if self.compress_mode == "streamingvlm":
            self._sliding_window_compress()
        else:
            self.pseudo_forward(local_q, global_q)

    @torch.inference_mode()
    def _observe_heads(self, attn_local, attn_global):
        """只取 cached KV 部分的 visual token，用 M-RoPE t 坐标分早期/近期"""

        visual_start = self.visual_start_idx
        pos_cache = self._position_ids_cache

        for layer_idx in range(len(attn_local)):
            al = attn_local[layer_idx]
            ag = attn_global[layer_idx]
            if al.dim() < 4 or ag.dim() < 4:
                continue

            # 只取 cached KV 范围（排除当前 query token 占的位置）
            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cached_kv_len = pos_cache[layer_idx].shape[1]
                t_pos = pos_cache[layer_idx][0]
            else:
                cached_kv_len = al.shape[3]
                t_pos = torch.arange(cached_kv_len, device=self.device)

            if cached_kv_len <= visual_start:
                continue

            n_visual = cached_kv_len - visual_start

            # 只取 cached 范围内的 visual token 注意力 [heads, n_visual]
            al_vis = al[0, :, :, visual_start:cached_kv_len].mean(dim=1)
            ag_vis = ag[0, :, :, visual_start:cached_kv_len].mean(dim=1)

            # 按 t 坐标中位数切分早期/近期
            visual_t = t_pos[visual_start:cached_kv_len]
            t_mid = (visual_t.min() + visual_t.max()) / 2
            early_mask = visual_t <= t_mid

            if early_mask.sum() == 0 or early_mask.sum() == n_visual:
                continue

            for head_idx in range(al_vis.shape[0]):
                al_head = al_vis[head_idx]
                ag_head = ag_vis[head_idx]

                al_total = al_head.sum().item()
                ag_total = ag_head.sum().item()
                if al_total == 0 or ag_total == 0:
                    continue

                le = al_head[early_mask].sum().item() / al_total
                ge = ag_head[early_mask].sum().item() / ag_total

                self.head_obs.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'local_early_ratio': le,
                    'global_early_ratio': ge,
                    'shift': ge - le,
                    'n_visual': n_visual,
                })


def run_analysis(args):
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"

    # 加载基础模型
    print(f"Loading base model: {model_path}")
    base_model = QwenVL_Hermes.__new__(QwenVL_Hermes)
    # ...
    # We need the load_model helper but with our Observed class

    # 直接用 ObservedQwenVL
    print("Building ObservedQwenVL ...")
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from inference.abstract_hermes import Abstract_Hermes
    from inference.reindex_3d import _get_mrope_section

    device = f"cuda:{args.device}"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float16)

    # Build ObservedQwenVL from raw_model
    model = ObservedQwenVL.__new__(ObservedQwenVL)
    model.__dict__ = raw_model.__dict__.copy()

    Abstract_Hermes.__init__(model, processor, init_prompt_ids.tolist(), args.kv_size)
    model.head_obs = []  # ObservedQwenVL 的属性，__dict__ copy 后需要手动设
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

    print(f"Model loaded. n_init={init_prompt_ids.shape[1]}, kv_size={args.kv_size}")

    # 加载数据
    with open(anno_path) as f:
        anno = json.load(f)

    if args.num_videos:
        anno = anno[:args.num_videos]

    print(f"Processing {len(anno)} videos ...")

    # 走一遍 streaming 推理
    from video_qa.hermes_vqa import HermesVQA

    os.makedirs(args.save_dir, exist_ok=True)

    # HermesVQA uses its own analyze_a_video which calls predict_and_compress -> pseudo_forward
    # We need to use the overridden method but through the normal VQA flow
    # HermesVQA stores model as self.qa_model – we pass our observed model there

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno, save_dir="/tmp/head_obs_tmp",
        sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )

    for video_sample in tqdm(anno):
        video_path = video_sample['video_path']

        if video_path.endswith('.npy'):
            video = analyzer.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            video_fps = video_sample.get('fps', None)
            video = analyzer.load_video_frames(video_path, video_fps, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        else:
            video = analyzer.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)

        model.clear_cache()
        model.encode_init_prompt()
        current_frame_idx = 0

        for sample in video_sample['conversations']:
            if 'end_time' in sample:
                end_frame_idx = math.ceil(sample['end_time'] * args.sample_fps)
            else:
                end_frame_idx = len(video_tensor)

            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + 16, end_frame_idx)
                if next_encode_end > current_frame_idx:
                    video_chunk = video_tensor[current_frame_idx:next_encode_end]
                    model.encode_video_chunk(video_chunk)
                    current_frame_idx = next_encode_end
                    # 这里调 predict_and_compress → pseudo_forward → prune_kv_cache_by_attention
                    # 我们的 _observe_heads 会在这之前被调用
                    model.predict_and_compress()

            if 'choices' in sample:
                choices = sample['choices']
                answer = sample.get('answer')
                if answer is None:
                    answer = choices[0]
                correct_choice = analyzer.choice_letters[choices.index(answer)]
                analyzer.video_close_qa(sample['question'], choices, correct_choice)

    # ---- 汇总结果 ----
    stats = model.head_obs
    print(f"\nTotal observations: {len(stats)}")

    if not stats:
        print("WARNING: No observations collected.")
        return

    num_layers = model.num_layers
    max_head = max(s['head'] for s in stats)
    num_heads = max_head + 1

    # 聚合：每层每头平均 shift（global_early - local_early）
    agg_sum = np.zeros((num_layers, num_heads))
    agg_cnt = np.zeros((num_layers, num_heads))
    agg_le = np.zeros((num_layers, num_heads))
    agg_ge = np.zeros((num_layers, num_heads))

    for s in stats:
        l, h = s['layer'], s['head']
        agg_sum[l, h] += s['shift']
        agg_le[l, h] += s['local_early_ratio']
        agg_ge[l, h] += s['global_early_ratio']
        agg_cnt[l, h] += 1

    shift_avg = np.where(agg_cnt > 0, agg_sum / agg_cnt, 0)
    le_avg = np.where(agg_cnt > 0, agg_le / agg_cnt, 0)
    ge_avg = np.where(agg_cnt > 0, agg_ge / agg_cnt, 0)

    out_path = os.path.join(args.save_dir, "head_pseudo.npz")
    np.savez(out_path,
             shift=shift_avg, local_early=le_avg, global_early=ge_avg,
             num_layers=num_layers, num_heads=num_heads, agg_count=agg_cnt)
    print(f"Saved to {out_path}")

    # 额外存一份 raw（每条观测的原始值），方便后续处理
    import pandas as pd
    raw_path = os.path.join(args.save_dir, "raw_obs.csv")
    pd.DataFrame(stats).to_csv(raw_path, index=False)
    print(f"Raw observations saved to {raw_path} ({len(stats)} rows)")

    # 输出
    print("\n=== Top Long-Term Heads (高 shift: global 比 local 更看早期) ===")
    flat = [(l, h, float(shift_avg[l, h]), float(le_avg[l, h]), float(ge_avg[l, h]), int(agg_cnt[l, h]))
            for l in range(num_layers) for h in range(num_heads) if agg_cnt[l, h] > 0]
    flat.sort(key=lambda x: -x[2])
    for l, h, s, le, ge, c in flat[:10]:
        print(f"  L{l:2d} H{h:2d}: shift={s:+.4f}  local_early={le:.3f}  global_early={ge:.3f}  (n={c})")

    print("\n=== Top Short-Term Heads (低 shift: 两个 query 都只看近期) ===")
    flat.sort(key=lambda x: x[2])
    for l, h, s, le, ge, c in flat[:10]:
        print(f"  L{l:2d} H{h:2d}: shift={s:+.4f}  local_early={le:.3f}  global_early={ge:.3f}  (n={c})")

    # 逐层
    print("\n=== Per-layer avg shift ===")
    for l in range(num_layers):
        if agg_cnt[l].sum() > 0:
            avg = agg_sum[l].sum() / agg_cnt[l].sum()
            print(f"  L{l:2d}: shift={avg:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=100000)
    parser.add_argument("--compress_mode", type=str, default="streamingvlm")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/pseudo")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=20)
    args = parser.parse_args()

    run_analysis(args)
