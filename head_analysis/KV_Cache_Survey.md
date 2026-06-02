# KV Cache 压缩：底层框架与研究空白

## 1. 问题的本质

流视频理解中的 KV cache 问题可以形式化为：

$$\max_{\mathcal{I} \subseteq \mathcal{V}, |\mathcal{I}| \leq B} \ \text{Accuracy}(\mathcal{I})$$

即在固定预算 $B$ 下，从全体 visual token $\mathcal{V}$ 中选出一个子集 $\mathcal{I}$，使得下游 QA 准确率最大化。

所有 KV cache 压缩方法的差异仅在于：**在哪个维度上定义"重要性"，以及用什么样的探针（probe）来测量它**。

## 2. 已探索的维度

调研了 30+ 篇相关论文（流视频压缩 + 多模态视觉压缩），总结如下：

### 2.1 Token 维度

| 方法 | 重要性定义 | 探针 |
|------|---------|------|
| SnapKV (NeurIPS 2024) | 观察窗口的注意力模式 | 最后 N 个 token 的 attention |
| KeyDiff (2025) | Key 向量的区分性 | Key 之间的余弦相似度 |
| InfiniPot-V (2025) | 时间轴冗余 + 值范数重要性 | 时间冗余度 + L2 范数 |
| VisionZip (CVPR 2025) | 视觉 token 的信息量 | 注意力分数 |
| FastV (ECCV 2024) | 视觉 token 的冗余度 | 逐层剪枝 |

**共同假设**：不同 token 对最终输出的贡献不均匀。**所有 token 统一评分，不分头**。

### 2.2 Layer 维度

| 方法 | 重要性定义 | 策略 |
|------|---------|------|
| PyramidKV (2024) | 浅层需要更多 token，深层可压缩 | 金字塔式预算分配 |
| PyramidDrop (CVPR 2025) | token 冗余度随深度递增 | 逐层递增丢 token 比例 |
| HERMES (ACL 2026) | 浅层=短期记忆，深层=长期记忆 | 分层 α（recency/attention 权重） |

**共同假设**：不同层的功能不同，冗余度不同。**同层内所有头统一处理**。

### 2.3 Modality 维度

| 方法 | 发现 | 策略 |
|------|------|------|
| LOOK-M (2024) | 模型 prefill 时优先关注文本 | Text-prior：文本优先保留 |
| MadaKV (ACL 2025) | 不同头对文本/视觉有不同偏好 | 模态感知的逐头驱逐 |

**共同假设**：文本和视觉 token 的重要性不同。MadaKV 首次在模态维度上做了头级区分。

### 2.4 Time/Frame 维度

| 方法 | 发现 | 策略 |
|------|------|------|
| DyCoke (CVPR 2025) | 不同 decode 步关注不同帧 | 动态逐步裁剪 |
| StreamingTOM (CVPR 2026) | 相邻帧高度冗余 | Causal temporal reduction |
| StreamMem (2025) | 对话模板 token 可作为代理查询 | Query-agnostic 重要性评分 |
| DSCache (2025) | 位置编码溢出问题 | 累积历史 + 按需即时缓存 |

**共同假设**：时间维度上存在大量冗余。**所有头统一处理时间维度**。

### 2.5 Head 维度（最未被充分探索）

| 方法 | 发现 | 策略 |
|------|------|------|
| SparseMM (ICCV 2025) | 约 5% 头是视觉头（在画面中检索文字） | Per-head 非对称预算 |
| HybridKV (2026) | 头分静态/动态两类 | 分类 + 分层预算 + 混合压缩 |
| Pyramid Forcing (2026) | Anchor/Wave/Veil 三类头（视频生成） | Per-head 异构 cache 长度 |
| SnapKV (NeurIPS 2024) | 每头独立选择 KV 位置 | Per-head 独立聚类 |

**SparseMM 和 HybridKV 在头维度上的工作仅限于空间维度（OCR 检索能力）或单图场景。流视频场景下，头的时间偏好分化尚未被探索。**

## 3. 统一底层框架

所有 30+ 篇方法遵循相同的三段式：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ 观测 (Probe)  │ ──→ │ 分类 (Taxonomy) │ ──→ │ 异构策略 (Policy)   │
│              │     │              │     │                  │
│ 用什么信号     │     │ 分成几类      │     │ 不同类不同处理     │
│ 来测量重要性？  │     │ 每类特征？    │     │ 预算/打分/窗口？   │
└──────────────┘     └──────────────┘     └──────────────────┘
```

| 方法 | 观测 | 分类 | 异构策略 |
|------|------|------|---------|
| SparseMM | OCR decode attention 命中 | Visual/Non-Visual | 非对称 budget |
| HERMES | 伪查询 attention | 浅/中/深层 | 不同 α |
| Pyramid Forcing | 扩散 attention pattern | Anchor/Wave/Veil | 异构 cache 长度 |
| **本工作** | **伪查询探针 temporal shift Δ** | **长期/近期 记忆头** | **异构驱逐公式 + 异构 budget** |

## 4. 研究空白：Head × Time × Streaming

### 4.1 为什么是空白

- **Token 维度**已被充分探索（SnapKV、KeyDiff、InfiniPot-V 等）
- **Layer 维度**已被充分探索（PyramidKV、HERMES 等）
- **Modality 维度**已有代表工作（LOOK-M、MadaKV）
- **Head 维度**仅 SparseMM/HybridKV 在单图场景探索了**空间**维度
- **Head × Time** 交叉：**无人**
- **Head × Time × Streaming** 交叉：**无人**

### 4.2 本工作的核心命题

> MLLM 的注意力头在时间维度上存在稳定的功能分化（长期偏好 vs 短期偏好）。这种分化可以通过**零标注的伪查询探针**测量，并用于设计**异构的 per-head KV cache 驱逐策略**，在固定显存预算下提升流视频理解性能。

### 4.3 关键差异化

| 维度 | 已有工作 | 本工作 |
|------|---------|--------|
| 场景 | SparseMM: 单图 OCR；Pyramid Forcing: 视频生成 | **流视频 MLLM QA** |
| 探针 | SparseMM: OCR 标注；HERMES: 伪查询（只用于打分） | **伪查询探针（用于头分析+打分，零标注）** |
| 头分类 | Pyramid Forcing: attention pattern 聚类 | **伪查询 Δ 直接测量时间偏好** |
| 异构策略 | SparseMM: 只改 budget；Pyramid: 只改 cache 长度 | **改驱逐公式 + 改 budget** |
| 动态性 | 所有静态 | **可按问题类型/视频长度自适应** |

### 4.4 三阶段实现路线

**Phase 1（已完成）**：
- Per-head budget 分配（SparseMM 分数 + HERMES 打分融合）
- Per-head 变长 KV 存储 + flash_attn_varlen_func 注意力
- StreamingBench 全量测评

**Phase 2（进行中）**：
- 伪查询探针头分类（Anchor/ Wave/ Veil）
- 异构驱逐公式（三类头不同的 recency/attention 权重）
- 消融实验（budget-only vs budget+α vs 异构驱逐）

**Phase 3（规划中）**：
- 跨架构验证（LLaVA-OV-7B）
- 任务自适应动态 budget
- 效率指标全量测量（TTFT、峰值显存、吞吐量）

## 5. 参考文献

- Zhang et al., "HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding", ACL 2026
- Wang et al., "SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs", ICCV 2025
- Chen et al., "Pyramid Forcing: Head-Aware Pyramid KV Cache Policy for High-Quality Long Video Generation", 2026
- Zeng et al., "HybridKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference", 2026
- Yang et al., "StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding", 2025
- Chen et al., "StreamingTOM: Streaming Token Compression for Efficient Video Understanding", CVPR 2026
- Li et al., "MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference", ACL 2025
- Liu et al., "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024
- Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling", 2024
