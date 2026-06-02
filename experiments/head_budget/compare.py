"""
A/B Test: Uniform HERMES vs Head-Weighted HERMES

直接对比两种策略在相同视频上的表现
"""

import os, sys, json, math, argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from head_analysis.hermes_head_budget import (
    load_sparsemm_scores, load_pseudo_scores, apply_head_budget)
from inference.qwenvl_hermes import QwenVL_Hermes
from inference.abstract_hermes import Abstract_Hermes
from inference.reindex_3d import _get_mrope_section
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from video_qa.base import BaseVQA


def load_model(model_path, kv_size, sample_fps, compress_mode, device, scores=None):
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    sp = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_ids = processor.tokenizer(sp, return_tensors="pt").input_ids.to(device)

    raw = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float16)

    m = QwenVL_Hermes.__new__(QwenVL_Hermes)
    m.__dict__ = raw.__dict__.copy()
    Abstract_Hermes.__init__(m, processor, init_ids.tolist(), kv_size)
    m.streaming = True
    m.sample_fps = sample_fps
    m.compress_mode = compress_mode
    nl = raw.model.config.num_hidden_layers
    m.num_layers = nl
    m._position_ids_cache = [None] * nl
    m.short_term_ratio = 0.1
    m.long_term_ratio = 0.3
    m.short_term_threshold = int(nl * 0.1)
    m.long_term_threshold = int(nl * 0.7)
    m._mrope_section = _get_mrope_section(raw.model)
    m._layer_position_ids = {}
    m._hook_handles = []
    m._register_forward_hooks()
    m.eval()

    if scores is not None:
        m = apply_head_budget(m, scores=scores, num_layers=nl, num_heads=28)
    return m, processor


def evaluate(model, processor, anno, sample_fps, save_dir, label):
    class Tmp(BaseVQA):
        pass

    analyzer = Tmp(anno=anno, save_dir="/tmp/ev", sample_fps=sample_fps,
                   qa_model=model, qa_processor=processor, num_chunks=None, chunk_idx=None)
    records = []

    for vs in tqdm(anno, desc=label):
        vp = vs['video_path']
        if vp.endswith('.npy'):
            v = analyzer.load_video(vp, clip=vs.get('clip', None))
            vt = torch.from_numpy(v)
        elif os.path.isdir(vp):
            vf = vs.get('fps', None) or 30
            v = analyzer.load_video_frames(vp, vf, clip=vs.get('clip', None))
            vt = torch.from_numpy(v)
        else:
            v = analyzer.load_video(vp, clip=vs.get('clip', None))
            vt = torch.from_numpy(v)

        model.clear_cache()
        model.encode_init_prompt()
        cf = 0

        for s in vs['conversations']:
            ef = math.ceil(s['end_time'] * sample_fps) if 'end_time' in s else len(vt)
            while cf < ef:
                ne = min(cf + 16, ef)
                if ne > cf:
                    model.encode_video_chunk(vt[cf:ne])
                    cf = ne
                    model.predict_and_compress()
            if 'choices' not in s:
                continue
            ch = s['choices']
            ans = s.get('answer')
            if ans is None: ans = ch[0]
            cc = analyzer.choice_letters[ch.index(ans)]
            qa = analyzer.video_close_qa(s['question'], ch, cc)
            records.append({
                'video_id': vs['video_id'],
                'task': s.get('task', '?'),
                'acc': qa['acc'],
                'pred': qa['pred_choice'],
                'correct': cc,
            })

    df = pd.DataFrame(records)
    df.to_csv(f"{save_dir}/{label}.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--scores", default="pseudo",
                        help="sparsemm | pseudo | none")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    model_path = f"models/{'Qwen2.5-VL-7B-Instruct' if args.model == 'qwen2.5_vl_7b' else args.model}"
    anno_path = "data/streamingbench/streamingbench_realtime.json"
    with open(anno_path) as f:
        anno = json.load(f)
    anno = anno[:args.num_videos]

    device = f"cuda:{args.device}"
    save_dir = f"results/head_budget_ab/{args.scores}"
    os.makedirs(save_dir, exist_ok=True)

    # 加载分数
    scores = None
    if args.scores == 'sparsemm':
        scores = load_sparsemm_scores("/home/sjs/SparseMM/visual_head/head_score/qwen.json")
        print(f"[Scores] SparseMM: range [{scores.min():.4f}, {scores.max():.4f}]")
    elif args.scores == 'pseudo':
        scores = load_pseudo_scores("results/head_analysis/pseudo-qwen2.5_vl_7b-kv6000-hermes/head_pseudo.npz")
        print(f"[Scores] Pseudo: range [{scores.min():.4f}, {scores.max():.4f}]")

    # == Run A: Uniform (no scores) ==
    print("\n=== A: Uniform HERMES ===")
    model_u, proc_u = load_model(model_path, args.kv_size, args.sample_fps,
                                  "hermes", device, scores=None)
    df_u = evaluate(model_u, proc_u, anno, args.sample_fps, save_dir, "uniform")
    acc_u = df_u['acc'].mean() * 100
    print(f"Uniform: {acc_u:.2f}%")

    del model_u

    # == Run B: Head-Weighted ==
    print(f"\n=== B: Head-Weighted HERMES ({args.scores}) ===")
    model_w, proc_w = load_model(model_path, args.kv_size, args.sample_fps,
                                  "hermes", device, scores=scores)
    df_w = evaluate(model_w, proc_w, anno, args.sample_fps, save_dir, "weighted")
    acc_w = df_w['acc'].mean() * 100
    print(f"Weighted: {acc_w:.2f}%")

    # == Compare ==
    print(f"\n=== A/B Comparison ({len(anno)} videos) ===")
    print(f"Uniform:  {acc_u:.2f}%")
    print(f"Weighted: {acc_w:.2f}%")
    print(f"Delta:    {acc_w - acc_u:+.2f}%")

    if 'task' in df_u.columns:
        print(f"\n{'Task':<25} {'Uniform':>8} {'Weighted':>8} {'Delta':>8}")
        tasks = sorted(set(df_u['task'].unique()) | set(df_w['task'].unique()))
        for t in tasks:
            au = df_u[df_u['task'] == t]['acc'].mean() * 100 if t in df_u['task'].values else 0
            aw = df_w[df_w['task'] == t]['acc'].mean() * 100 if t in df_w['task'].values else 0
            print(f"{t:<25} {au:>7.1f}% {aw:>7.1f}% {aw-au:>+7.1f}%")

    with open(f"{save_dir}/summary.txt", 'w') as f:
        f.write(f"Uniform:  {acc_u:.2f}%\nWeighted: {acc_w:.2f}%\nDelta:    {acc_w - acc_u:+.2f}%\n")


if __name__ == "__main__":
    main()
