"""
Chunk-level hit analysis with self-labeling

流程:
  1. 每个 chunk 编码后，向模型问 YES/NO: "Did [event] happen in this scene?"
  2. YES 的 chunk 标记为事件 chunk
  3. 删除问句和答案 token，恢复 KV cache 到编码后状态
  4. CT 最终题来时，每头 top-K attention chunks vs 标记的事件 chunks → hit rate

和 SparseMM 对齐: 标签(chunk YES/NO) 和 测试(CT题注意力) 是两次独立前向，不循环。
"""

import os, sys, json, math, argparse, re
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.qwenvl_hermes import load_model
from video_qa.base import BaseVQA


def extract_action(question):
    """从 CT 题提取核心动作"""
    q = question.lower().strip('?').strip('.')
    # 去掉前缀
    for p in ['how many times in total have', 'how many times have',
              'how many times did', 'how many times', 'how many']:
        if q.startswith(p):
            q = q[len(p):].strip()
            break
    # 去掉后缀
    for s in ['in total so far', 'in total', 'so far', 'during this',
              'in this video', 'right now', 'so far?', 'in this scene']:
        if s in q:
            q = q.split(s)[0].strip()
    q = q.strip('?').strip('.').strip()
    return f"Is this happening: {q}?" if len(q) > 5 else None


class ChunkLabelHitVQA(BaseVQA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_records = []
        self._chunk_vstart = []

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
        self._chunk_vstart = [self.qa_model.visual_start_idx]
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

            # 准备 chunk 标注
            is_ct = (task == 'Counting')
            chunk_question = extract_action(question) if is_ct else None
            chunk_labels = []  # 每题的 chunk 标签
            chunk_start = len(self._chunk_vstart)  # 这个 question 开始前的 chunk 索引

            while current_frame_idx < end_fidx:
                next_end = min(current_frame_idx + encode_chunk_size, end_fidx)
                if next_end > current_frame_idx:
                    self.qa_model.encode_video_chunk(video_tensor[current_frame_idx:next_end])
                    current_frame_idx = next_end

                    if is_ct and chunk_question:
                        label = self._label_chunk(chunk_question)
                        chunk_labels.append(label)

                    self._chunk_vstart.append(self.qa_model.kv_cache[0][0].shape[2])

            chunk_end = len(self._chunk_vstart) - 1

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

            answer_num = self._parse_number(answer)
            if task == 'Counting':
                if answer_num is None:
                    continue
                if answer_num == 0:
                    continue

            # --- 测 chunk attention 命中率 ---
            self._analyze_chunk_hits(
                mc_input, chunk_start, chunk_end, task, answer_num, is_correct,
                chunk_labels=chunk_labels if chunk_labels else None)

    @torch.inference_mode()
    def _label_chunk(self, yesno_question):
        """问模型: 这个 chunk 是否包含事件？记录 YES/NO 标签，删掉问句"""
        # 保存当前 KV cache 长度
        past_lens = self.qa_model._get_cache_seq_len_per_layer()

        prompt = f"\n{yesno_question} Answer Yes or No.<|im_end|><|im_start|>assistant\n"
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        # prefill
        global_offset = self.qa_model._get_next_global_offset_per_layer()
        self.qa_model._layer_position_ids.clear()
        for layer_idx in range(self.qa_model.num_layers):
            pos_3d = self.qa_model._build_position_ids_3d_for_text(
                global_offset[layer_idx], input_ids.shape[1], 1)
            self.qa_model._layer_position_ids[layer_idx] = pos_3d
        pos_3d = self.qa_model._build_position_ids_3d_for_text(global_offset[0], input_ids.shape[1], 1)

        out = self.qa_model.language_model(
            inputs_embeds=self.qa_model.get_input_embeddings()(input_ids),
            use_cache=True, past_key_values=self.qa_model.kv_cache,
            position_ids=pos_3d)
        self.qa_model._layer_position_ids.clear()
        logits = self.qa_model.lm_head(out.last_hidden_state)

        # decode 最多 5 token
        output_ids = []
        for _ in range(5):
            token = int(logits[0, -1, :].argmax())
            output_ids.append(token)
            if token in [self.qa_model.processor.tokenizer.eos_token_id]:
                break
            curr_offset = self.qa_model._get_next_global_offset_per_layer()
            self.qa_model._layer_position_ids.clear()
            for layer_idx in range(self.qa_model.num_layers):
                p = self.qa_model._build_position_ids_3d_for_text(curr_offset[layer_idx], 1, 1)
                self.qa_model._layer_position_ids[layer_idx] = p
            pos = self.qa_model._build_position_ids_3d_for_text(curr_offset[0], 1, 1)
            out = self.qa_model.language_model(
                input_ids=torch.as_tensor([[token]], device=self.qa_model.device),
                use_cache=True, past_key_values=out.past_key_values,
                position_ids=pos)
            logits = self.qa_model.lm_head(out.last_hidden_state)
        self.qa_model._layer_position_ids.clear()

        answer = self.qa_model.processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        label = 1 if 'yes' in answer else 0  # 1=事件chunk, 0=非事件
        # debug: 打印标签
        print(f"  [Chunk Label] Q: {yesno_question[:80]} → A: {answer} → label={label}")

        # 恢复 KV cache (删掉问句+答案)
        self.qa_model._truncate_kv_cache(past_lens)
        return label
        for layer_idx in range(self.qa_model.num_layers):
            if self.qa_model._position_ids_cache[layer_idx] is not None and \
               self.qa_model._position_ids_cache[layer_idx].shape[1] > past_lens[layer_idx]:
                self.qa_model._position_ids_cache[layer_idx] = \
                    self.qa_model._position_ids_cache[layer_idx][:, :past_lens[layer_idx]].contiguous()

    def _parse_number(self, text):
        m = re.search(r'\b(\d+)\b', text)
        if m: return int(m.group(1))
        for w, v in {'one':1,'two':2,'three':3,'four':4,'five':5,
                     'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
                     'zero':0,'no':0,'none':0}.items():
            if w in text.lower(): return v
        return None

    @torch.inference_mode()
    def _analyze_chunk_hits(self, input_text, chunk_start, chunk_end, task, answer_num, is_correct, chunk_labels=None):
        prompt = input_text['prompt']
        input_ids = self.qa_model.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=self.qa_model.device)

        attn_weights = self.qa_model._compute_attention_scores_manually(
            input_ids, self.qa_model.kv_cache)

        # chunk 边界：取当前 question 范围内的 vstart
        vs = self.qa_model.visual_start_idx
        cached_len = self.qa_model.kv_cache[0][0].shape[2]
        total_visual = cached_len - vs

        vstarts = self._chunk_vstart[chunk_start:chunk_end+1]  # 多取一个: N个chunk → N+1个边界
        cv_start = [max(0, s - vs) for s in vstarts[:-1]]       # 前N个 = 每个chunk起始
        cv_end   = [max(0, s - vs) for s in vstarts[1:]]        # 后N个 = 每个chunk结束
        n_chunks = len(cv_start)
        if n_chunks < 2: return

        # 确定事件 chunk
        if chunk_labels is not None:
            labels = np.array(chunk_labels[:n_chunks])
            event_chunks = set(np.where(labels == 1)[0].tolist())
            N = len(event_chunks)
            if N > 0:
                print(f"  [CT-Hit] answer={answer_num} | n_chunks={n_chunks} | "
                      f"EVENT chunks={sorted(event_chunks)} | N={N}")
        else:
            # fallback: ensemble top-N
            N = max(1, answer_num) if answer_num else max(2, n_chunks // 5)
            N = min(N, n_chunks)
            # 用第一层所有头平均找 top chunk
            l0 = attn_weights[0]
            if l0.dim() >= 4:
                ensemble = l0[0, :, :, vs:cached_len].mean(dim=(0,1))
                chunk_attn = np.array([ensemble[max(0, cv_start[i]):min(cv_end[i], total_visual)].sum().item()
                                       for i in range(n_chunks)])
                event_chunks = set(np.argsort(-chunk_attn)[:N].tolist())
            else:
                return

        # 每头 top-K 命中（SparseMM 风格）
        K = max(1, N) if N > 0 else 2
        pos_cache = self.qa_model._position_ids_cache

        for layer_idx, layer_attn in enumerate(attn_weights):
            if layer_attn.dim() < 4: continue
            if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
                cl = pos_cache[layer_idx].shape[1]
            else:
                cl = layer_attn.shape[3]
            nv = cl - vs
            if nv <= 0: continue

            lv = layer_attn[0, :, :, vs:cl].mean(dim=1)  # [heads, n_visual]

            # 每头每 chunk attention
            lhc = torch.zeros((lv.shape[0], n_chunks), device=lv.device)
            for ci in range(n_chunks):
                s, e = cv_start[ci], min(cv_end[ci], nv)
                if s < e: lhc[:, ci] = lv[:, s:e].sum(dim=1)
            lhc = lhc / (lhc.sum(dim=1, keepdim=True) + 1e-8)
            lhc_np = lhc.cpu().numpy()

            for head_idx in range(lhc_np.shape[0]):
                topk = np.argpartition(-lhc_np[head_idx], min(K, n_chunks)-1)[:K]
                hits = len(set(topk.tolist()) & event_chunks) if event_chunks else 0
                self.hit_records.append({
                    'layer': layer_idx, 'head': head_idx, 'task': task,
                    'hit_rate': hits / K if K > 0 else 0,
                    'n_hits': hits, 'K': K,
                    'answer_num': answer_num, 'n_chunks': n_chunks,
                    'is_correct': is_correct,
                    'has_labels': chunk_labels is not None,
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

    # 关压缩：需要完整 chunk 信息
    model.predict_and_compress = lambda: None

    with open(anno_path) as f:
        anno = json.load(f)

    targets = {'Counting', 'Causal Reasoning'}
    selected = [v for v in anno if any(c['task'] in targets for c in v['conversations'])]
    if args.num_videos:
        selected = selected[:args.num_videos]

    n_sel = sum(1 for v in selected for c in v['conversations'] if c['task'] in targets)
    print(f"Selected {len(selected)} videos, {n_sel} target Qs")

    save_dir = args.save_dir or "results/head_analysis/chunk_label_hit"
    os.makedirs(save_dir, exist_ok=True)

    analyzer = ChunkLabelHitVQA(
        anno=selected, save_dir=save_dir, sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )
    analyzer.analyze(debug=False)

    records = analyzer.hit_records
    labeled_records = [r for r in records if r.get('has_labels', False)]
    correct_labeled = [r for r in labeled_records if r['is_correct']]

    print(f"\nTotal records: {len(records)}, With labels: {len(labeled_records)}, Correct+Labeled: {len(correct_labeled)}")

    src = correct_labeled if correct_labeled else [r for r in records if r['is_correct']]
    if not src:
        src = records
    if not src:
        print("WARNING: No data.")
        return

    num_layers, num_heads = model.num_layers, max(s['head'] for s in src) + 1

    # Raw JSON (SparseMM 风格)
    raw_dict = defaultdict(list)
    for r in src:
        raw_dict[f"{r['layer']}-{r['head']}"].append(r['hit_rate'])
    raw_path = os.path.join(save_dir, "chunk_label_hit_raw.json")
    with open(raw_path, 'w') as f: json.dump(raw_dict, f)
    print(f"Raw saved: {raw_path} ({len(raw_dict)} heads)")

    # 聚合
    hs, hc = np.zeros((num_layers, num_heads)), np.zeros((num_layers, num_heads))
    for r in src:
        l, h = r['layer'], r['head']
        hs[l, h] += r['hit_rate']; hc[l, h] += 1
    hit_rate = np.where(hc > 0, hs / hc, 0)
    np.savez(os.path.join(save_dir, "chunk_label_hit_scores.npz"),
             hit_rate=hit_rate, count=hc, num_layers=num_layers, num_heads=num_heads)

    print("\n=== Top Chunk-Label Hit Heads ===")
    flat = [(l, h, float(hit_rate[l,h]), int(hc[l,h])) for l in range(num_layers) for h in range(num_heads) if hc[l,h] >= 2]
    flat.sort(key=lambda x: -x[2])
    for l, h, r, c in flat[:15]:
        print(f"  L{l:2d} H{h:2d}: hit_rate={r:.3f}  (n={c})")

    print("\n=== Per-layer ===")
    for l in range(num_layers):
        m = hc[l] >= 2
        if m.any():
            print(f"  L{l:2d}: {hit_rate[l][m].mean():.3f}")

    return hit_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--anno_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/head_analysis/chunk_label_hit")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=5)
    args = parser.parse_args()
    run(args)
