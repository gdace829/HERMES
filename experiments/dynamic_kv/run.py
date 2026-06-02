"""Per-Head Dynamic KV Experiment Runner"""
import os, sys, json, math, argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

from experiments.dynamic_kv.layer_budget import install
from head_analysis.hermes_head_budget import load_pseudo_scores, load_sparsemm_scores
from inference.qwenvl_hermes import QwenVL_Hermes
from inference.abstract_hermes import Abstract_Hermes
from inference.reindex_3d import _get_mrope_section
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from video_qa.base import BaseVQA


def load_model_with_per_head_kv(model_path, kv_size, sample_fps, device, scores=None):
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    sp = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_ids = processor.tokenizer(sp, return_tensors="pt").input_ids.to(device)

    raw = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.float16)

    m = QwenVL_Hermes.__new__(QwenVL_Hermes)
    m.__dict__ = raw.__dict__.copy()
    Abstract_Hermes.__init__(m, processor, init_ids.tolist(), kv_size)
    m.streaming, m.sample_fps, m.compress_mode = True, sample_fps, "hermes"
    nl = raw.model.config.num_hidden_layers
    m.num_layers = nl
    m._position_ids_cache = [None] * nl
    m.short_term_ratio, m.long_term_ratio = 0.1, 0.3
    m.short_term_threshold = int(nl * 0.1)
    m.long_term_threshold = int(nl * 0.7)
    m._mrope_section = _get_mrope_section(raw.model)
    m._layer_position_ids, m._hook_handles = {}, []
    m._register_forward_hooks()
    m.eval()

    install(m, head_scores=scores)
    return m, processor


def evaluate(model, processor, anno, sample_fps, name):
    class T(BaseVQA):
        pass
    a = T(anno=anno, save_dir="/tmp/e", sample_fps=sample_fps,
           qa_model=model, qa_processor=processor, num_chunks=None, chunk_idx=None)
    records = []
    for vs in tqdm(anno, desc=name):
        vp = vs['video_path']
        if vp.endswith('.npy'):
            vv = torch.from_numpy(a.load_video(vp, clip=vs.get('clip', None)))
        elif os.path.isdir(vp):
            vf = vs.get('fps', None) or 30
            vv = torch.from_numpy(a.load_video_frames(vp, vf, clip=vs.get('clip', None)))
        else:
            vv = torch.from_numpy(a.load_video(vp, clip=vs.get('clip', None)))
        model.clear_cache()
        model.encode_init_prompt()
        cf = 0
        for s in vs['conversations']:
            ef = math.ceil(s['end_time'] * sample_fps) if 'end_time' in s else len(vv)
            while cf < ef:
                ne = min(cf + 16, ef)
                if ne > cf:
                    model.encode_video_chunk(vv[cf:ne]); cf = ne
                    model.predict_and_compress()
            if 'choices' not in s: continue
            ch = s['choices']; ans = s.get('answer')
            if ans is None: ans = ch[0]
            cc = a.choice_letters[ch.index(ans)]
            qa = a.video_close_qa(s['question'], ch, cc)
            records.append({'task': s.get('task', '?'), 'acc': qa['acc']})
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kv_size", type=int, default=6000)
    p.add_argument("--scores", default="pseudo")
    p.add_argument("--num_videos", type=int, default=5)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    model_path = os.path.join(PROJ_ROOT, "models/Qwen2.5-VL-7B-Instruct")
    with open(os.path.join(PROJ_ROOT, "data/streamingbench/streamingbench_realtime.json")) as f:
        anno = json.load(f)[:args.num_videos]
    device = f"cuda:{args.device}"

    scores = None
    if args.scores == 'pseudo':
        scores = load_pseudo_scores(os.path.join(PROJ_ROOT,
            "results/head_analysis/pseudo-qwen2.5_vl_7b-kv6000-hermes/head_pseudo.npz"))
    elif args.scores == 'sparsemm':
        scores = load_sparsemm_scores("/home/sjs/SparseMM/visual_head/head_score/qwen.json")
    elif args.scores.endswith('.json'):
        scores = load_sparsemm_scores(args.scores)
    elif args.scores.endswith('.npz'):
        scores = load_pseudo_scores(args.scores)

    print(f"=== Per-Head KV ({args.scores}, {args.num_videos} vids) ===")
    model, processor = load_model_with_per_head_kv(model_path, args.kv_size, 0.5, device, scores)
    df = evaluate(model, processor, anno, 0.5, "per_head")
    acc = df['acc'].mean() * 100
    print(f"\nOverall: {acc:.2f}%")
    for t in sorted(df['task'].unique()):
        s = df[df['task'] == t]
        print(f"  {t}: {s['acc'].mean()*100:.1f}% (n={len(s)})")

    sd = os.path.join(PROJ_ROOT, f"results/dynamic_kv/{args.scores}")
    os.makedirs(sd, exist_ok=True)
    df.to_csv(f"{sd}/results.csv", index=False)


if __name__ == "__main__":
    main()
