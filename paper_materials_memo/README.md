# MemoSelect Materials Package

This folder is a curated material package for building the MemoSelect paper
with PaperSpine. It does not replace the raw experiment directories. Instead,
it records the paper-relevant claims, result summaries, figure candidates, and
reference anchors.

## Working Thesis

Memory attention is not the same as useful memory retrieval. In streaming video
KV cache compression, attention mass can be dominated by boundary or splice
points, while only some heads provide reliable signals for selecting
future-reusable internal memory. MemoSelect profiles such heads offline and uses
them to guide fixed-budget dense KV eviction.

## Main Local Sources

- Raw project: `/home/sjs/HERMES`
- Observation results: `results/observations/`
- StreamingBench results: `results/qwen2.5_vl_7b/streamingbench/`
- Paper draft page: Feishu MemoSelect child page
- Reference paper: `111.pdf` in repo root

## Current Status

- Existing evidence supports memory-biased readout and baseline comparison.
- Boundary-aware internal-memory selector utility is the next required
  experiment.
- MemoSelect method results are not yet available and must not be claimed as
  completed.
