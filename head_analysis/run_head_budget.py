"""
HERMES + Head-Level Budget 实验脚本

支持三种头分数来源:
  --scores sparsemm     : 用 SparseMM 的 qwen.json (视觉检索能力)
  --scores pseudo       : 用我们自己打的 head_pseudo.npz (时间偏好)
  --scores <path>       : 自定义路径

用法:
  python head_analysis/run_head_budget.py \
      --model qwen2.5_vl_7b --kv_size 6000 --compress_mode hermes \
      --scores sparsemm --num_videos 5 --device 0
"""

import os, sys, json, math, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import QwenVL_Hermes, load_model
from inference.abstract_hermes import Abstract_Hermes
from inference.reindex_3d import _get_mrope_section
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from head_analysis.hermes_head_budget import apply_head_budget, build_head_weights
from video_qa.base import BaseVQA


def get_scores(args):
    if args.scores == 'sparsemm':
        path = "/home/sjs/SparseMM/visual_head/head_score/qwen.json"
        print(f"[Scores] Loading SparseMM scores from {path}")
        from head_analysis.hermes_head_budget import load_sparsemm_scores
        return load_sparsemm_scores(path)
    elif args.scores == 'pseudo':
        path = "results/head_analysis/pseudo-qwen2.5_vl_7b-kv6000-hermes/head_pseudo.npz"
        print(f"[Scores] Loading pseudo scores from {path}")
        from head_analysis.hermes_head_budget import load_pseudo_scores
        return load_pseudo_scores(path)
    elif args.scores:
        path = args.scores
        print(f"[Scores] Loading from {path}")
        if path.endswith('.json'):
            from head_analysis.hermes_head_budget import load_sparsemm_scores
            return load_sparsemm_scores(path)
        elif path.endswith('.npz'):
            from head_analysis.hermes_head_budget import load_pseudo_scores
            return load_pseudo_scores(path)
    return None


def build_model_with_head_budget(args, scores):
    """加载 Qwen2.5-VL-7B 并挂上头级预算"""
    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    device = f"cuda:{args.device}"

    print(f"Loading model: {model_path}")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float16)

    model = QwenVL_Hermes.__new__(QwenVL_Hermes)
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

    # ---- 应用 head budget ----
    if scores is not None:
        model = apply_head_budget(model, scores=scores,
                                   num_layers=num_layers, num_heads=28)

    return model, processor


def run(args):
    anno_path = args.anno_path or "data/streamingbench/streamingbench_realtime.json"

    scores = get_scores(args)
    model, processor = build_model_with_head_budget(args, scores)

    with open(anno_path) as f:
        anno = json.load(f)

    if args.num_videos:
        anno = anno[:args.num_videos]

    save_dir = args.save_dir or f"results/head_budget/{args.scores or 'uniform'}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {len(anno)} videos, saving to {save_dir}")

    class TempVQA(BaseVQA):
        pass

    analyzer = TempVQA(
        anno=anno, save_dir="/tmp/hb_tmp", sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )

    records = []
    for video_sample in tqdm(anno):
        video_path = video_sample['video_path']
        if video_path.endswith('.npy'):
            video = analyzer.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        elif os.path.isdir(video_path):
            vfps = video_sample.get('fps', None) or 30
            video = analyzer.load_video_frames(video_path, vfps, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)
        else:
            video = analyzer.load_video(video_path, clip=video_sample.get('clip', None))
            video_tensor = torch.from_numpy(video)

        model.clear_cache()
        model.encode_init_prompt()
        current_frame_idx = 0

        for sample in video_sample['conversations']:
            if 'end_time' in sample:
                end_fidx = math.ceil(sample['end_time'] * args.sample_fps)
            else:
                end_fidx = len(video_tensor)

            while current_frame_idx < end_fidx:
                next_end = min(current_frame_idx + 16, end_fidx)
                if next_end > current_frame_idx:
                    model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                    current_frame_idx = next_end
                    model.predict_and_compress()

            if 'choices' not in sample:
                continue

            choices = sample['choices']
            answer = sample.get('answer')
            if answer is None: answer = choices[0]
            correct_choice = analyzer.choice_letters[choices.index(answer)]
            qa = analyzer.video_close_qa(sample['question'], choices, correct_choice)

            records.append({
                'video_id': video_sample['video_id'],
                'question': sample['question'],
                'task': sample.get('task', 'Unknown'),
                'answer': answer,
                'pred_choice': qa['pred_choice'],
                'correct_choice': correct_choice,
                'acc': qa['acc'],
            })

    # 统计
    import pandas as pd
    df = pd.DataFrame(records)
    overall = df['acc'].mean() * 100

    print(f"\n=== Results ({args.scores or 'uniform'}) ===")
    print(f"Overall: {overall:.2f}% ({len(df)} questions)")

    if 'task' in df.columns:
        for task in sorted(df['task'].unique()):
            sub = df[df['task'] == task]
            print(f"  {task}: {sub['acc'].mean()*100:.1f}% (n={len(sub)})")

    df.to_csv(os.path.join(save_dir, "results.csv"), index=False)
    print(f"Saved to {save_dir}/results.csv")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--scores", type=str, default=None,
                        help="sparsemm, pseudo, or path to json/npz")
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=10)
    args = parser.parse_args()
    run(args)
