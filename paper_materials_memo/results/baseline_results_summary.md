# Baseline Result Summary

## StreamingBench Full Results

Source files:

- HERMES: `results/qwen2.5_vl_7b/streamingbench/fps0.5-kv6000-hermes/results.csv`
- StreamingVLM: `results/qwen2.5_vl_7b/streamingbench/fps0.5-kv6000-streamingvlm/results.csv`
- Visual-head budget prototype:
  `results/qwen2.5_vl_7b/streamingbench/fps0.5-kv6000-hermes-kvheadbudget-qwen2.5-vl-sparsemm-union1-r0.1-w32-raggedprefill/results.csv`

| Method | Overall | Action | Attribute | Object | Spatial | Event | Text-Rich | Counting | Causal | Clips | Prospective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StreamingVLM | 76.95 | 70.54 | 88.12 | 79.84 | 72.76 | 76.10 | 83.49 | 41.97 | 72.66 | 88.33 | 82.41 |
| HERMES | 76.95 | 68.84 | 85.81 | 80.11 | 71.95 | 74.84 | 84.11 | 52.85 | 75.00 | 86.44 | 78.70 |
| Visual-head budget prototype | 77.35 | 72.52 | 87.13 | 79.02 | 70.33 | 77.99 | 85.67 | 50.26 | 71.09 | 85.49 | 82.41 |

## Interpretation

StreamingVLM and HERMES tie in overall score but differ by task. StreamingVLM
improves recent or short-term perception tasks but sharply hurts Counting.
HERMES better preserves history-heavy capabilities. The visual-head budget
prototype gives a weak positive overall signal but still hurts Counting,
Causal Reasoning, and Spatial Understanding. This motivates a more direct
future-reusable memory selection criterion.

## Safe Wording

The visual-head budget result suggests that head-level signals are useful, but
visual-head strength alone is not a reliable proxy for reusable memory
selection.
