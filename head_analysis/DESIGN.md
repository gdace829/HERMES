# Head Analysis — SparseMM 风格 token 命中流程

## 核心思路

SparseMM (OCR): 画面中文字框 = "答案区域" → 统计 decode 时每个头是否 attend 到正确答案区域
本方案 (Video): 问题时间戳 = "答案窗口" → 统计每个头是否 attend 到答案相关的视频时间段

## 处理流程

### Phase 1: 构建"答案时间窗口"

输入: streamingbench_realtime.json
输出: 每个 QA pair 的 [answer_start_t, answer_end_t] (M-RoPE t 坐标)

方法:
  对于每个问题 (end_time = T)，答案相关信息最可能出现在:
    window = [T - 10s, T]   (默认 10 秒窗口)
  转换为 M-RoPE t 坐标:
    t_per_second = sample_fps / 2  (0.5fps → 0.25)
    answer_t_start = (T - 10) * t_per_second
    answer_t_end = T * t_per_second

按任务类型分组:
  Probe A (记忆): CR + CT  → 测长期检索能力
  Probe B (近期): CS + PR  → 测近期检索能力

### Phase 2: 捕获答题时的注意力

不修改原代码。在 HermesVQA.analyze_a_video 的 question_answering 调用后，
用 _compute_attention_scores_manually 重新计算 question → KV cache 的 attention。

具体:
  1. question_answering 正常执行 (prefill + decode)
  2. answer 生成后，_truncate_kv_cache 恢复 KV 到提问前状态
  3. 用 question 的 input_ids 调 _compute_attention_scores_manually
     得到每个头的注意力分布: [heads, kv_len]
  4. 对每个 head_idx:
       answer_hit = sum(attention_to_tokens_in_answer_window) / sum(attention_to_all_visual)
  5. 记录: (layer, head, answer_hit, task_type, question_position)

### Phase 3: 按头聚合统计

按 Probe A / Probe B 分别算每个头的平均 answer_hit:

  mem_score[layer, head] = mean(answer_hit over CR+CT questions)
  rec_score[layer, head] = mean(answer_hit over CS+PR questions)

衍生指标:
  - retrieval_ability = (mem_score + rec_score) / 2       ← 检索能力
  - memory_specialization = mem_score - rec_score          ← 长期偏向

### Phase 4: 头分类

按 retrieval_ability × memory_specialization 画 scatter:

              memory_specialization ↑
                                     |
    记忆检索头 (高检索+长期偏向)      |
                                     |
  ──────────────────────────────────→ retrieval_ability
                                     |
    近期检索头 (高检索+短期偏向)      |
                                     |

从 scatter 中选出:
  - 记忆检索头: retrieval_ability > median 且 memory_specialization > 0
  - 近期检索头: retrieval_ability > median 且 memory_specialization < 0
  - 无关头: retrieval_ability < median

### Phase 5: 预算分配 (可选, if 接入 HERMES)

在 prune_kv_cache_by_attention 中:
  - 记忆检索头: 用 global_attention 权重 × α (高)
  - 近期检索头: 用 local_attention 权重 × β (高)
  - 无关头: 用 mixed 权重 × γ (低)
  - 按 head 独立做 top-k

## 数据量估算

  StreamingBench: 498 videos × ~5 questions = ~2495 questions
  Probe A (CR+CT): 321 questions
  Probe B (CS+PR): 425 questions
  总可用: ~746 questions

  每个 question: 28 layers × 28 heads = 784 scalar stats
  总计: 746 × 784 ≈ 585K scalar values ≈ ~2MB

## 实施复杂度

  Phase 1: ★☆☆ (annotation 已有 end_time, 直接算)
  Phase 2: ★★☆ (复用 _compute_attention_scores_manually, 已跑通)
  Phase 3: ★☆☆ (numpy 聚合)
  Phase 4: ★☆☆ (matplotlib scatter)
  Phase 5: ★★★ (需改 qwenvl_hermes.py 的 prune_kv_cache_by_attention)

  总工作量: ~4-6 小时

## 对比现有方案

| | 当前伪查询方案 | 本方案 | SparseMM |
|------|------|------|------|
| 信号源 | pseudo query attention | 真问题 prefill attention | decode token attention |
| 命中定义 | attention 落在早期/近期 | attention 落在答案时间窗口 | attention 落在文字 bbox |
| 精度 | 时间: 早期/近期 二值 | 时间: ~秒级窗口 | 空间: 像素级 bbox |
| 样本量 | 每 chunk 一次 (几千次) | 每问题一次 (746 次) | 每图每 token (几千 × N) |
| 改动量 | 0 | 0 (复用现有) | 改整个 forward |
