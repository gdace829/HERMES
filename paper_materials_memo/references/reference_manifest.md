# Reference Manifest

Local references and writing exemplars for MemoSelect.

| Reference | Local Path / Source | Intended Use |
|---|---|---|
| Forcing-KV / 111.pdf | `/home/sjs/HERMES/111.pdf` | Writing logic exemplar: observation -> head profiling -> head-specific compression -> random/manual ablation. Do not copy method. |
| StreamingVLM | `/home/sjs/PaperList/流视频压缩/StreamingVLM_2510.09608.pdf` | Streaming video inference baseline and related work. |
| HERMES | project code and existing draft | Baseline and implementation context. |
| SparseMM visual head score | `/home/sjs/SparseMM/visual_head/head_score/qwen2.5-vl.json` | External visual-head baseline, not main method score. |

## Required Literature Work

PaperSpine research should still build a citation support bank for:

- streaming video VLM inference
- KV cache compression
- visual token pruning
- attention sink / streaming cache
- head specialization / head-aware compression
- boundary or recency effects if relevant
