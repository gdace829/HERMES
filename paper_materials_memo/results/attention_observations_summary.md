# Attention Observation Summary

## Full Eager Pseudo-Query Attention

Primary source:

`results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper/summary.json`

Key values:

- observations: 203
- layer-head observations: 159,152
- mean local current share: 0.236
- mean global current share: 0.242
- mean current token fraction: 0.741
- median local current-to-memory per-token ratio: 0.0506
- median global current-to-memory per-token ratio: 0.0533
- mean memory visual tokens: 5,953
- mean latest chunk tokens: 18,888

Interpretation:

The latest chunk contributes around 74% of visual tokens but receives only
around 24% of pseudo-query attention mass. Memory occupies around 26% of visual
tokens but receives around 76% of readout attention.

## KV-Head Aggregated Observation

Primary source:

`results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_head_profile_summary.json`

Key values:

- layer-KV-head groups: 112
- mean s_current_share: 0.237
- s_current_share quantiles:
  - min: 0.061
  - median: 0.230
  - 90%: 0.388
  - max: 0.478
- current token fraction: 0.741

Interpretation:

All existing KV-head groups have current attention share below the current
token fraction. However, this should not be overread as "all heads deeply use
memory"; high memory-region attention can be concentrated near the splice
boundary rather than distributed through internal memory.

## Needed Follow-Up

Add boundary-aware measurements:

- boundary attention ratio
- internal-memory utility heatmap
- scatter of boundary ratio vs internal-memory utility
- top/random/bottom internal-memory coverage
