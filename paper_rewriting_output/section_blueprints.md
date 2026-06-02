# Section Blueprints

## Title

Working title:

`MemoSelect: Future-Reusable Memory Selection for Streaming Video KV Cache Compression`

Alternative title:

`MemoSelect: Boundary-Aware Memory Selection for Streaming Video KV Cache Compression`

## Abstract Blueprint

1. State the streaming video KV cache bottleneck.
2. Explain why uniform or recent-window eviction is insufficient.
3. Introduce the key observation: memory attention is high but can be
   boundary-structured, so high memory attention is not necessarily useful
   memory retrieval.
4. Introduce boundary-aware future-reusable memory utility.
5. Introduce MemoSelect as dense fixed-budget eviction guided by reliable
   internal-memory selectors.
6. Report only completed results. Current completed results can mention
   observation and baseline limitations; final performance numbers require
   MemoSelect experiments.

## Introduction Blueprint

1. Streaming video inference requires repeated visual KV cache eviction.
2. Existing strategies rely on recency, average attention, or visual-head
   scores.
3. Existing strategies show complementary failure modes: StreamingVLM hurts
   Counting, visual-head budget gives weak overall gain but hurts memory-heavy
   tasks.
4. Observation: memory receives disproportionate pseudo-query readout.
5. Gap: memory attention may be concentrated at boundary/splice points and
   does not directly measure reusable internal memory.
6. Proposed answer: profile heads by future-reusable internal-memory selection
   utility and use reliable selectors for fixed-budget eviction.

## Observation Section Blueprint

1. Define memory/latest chunk decomposition.
2. Report memory-biased readout statistics.
3. Show head-level heterogeneity using existing heatmaps.
4. Add boundary concentration as the missing diagnostic.
5. Transition to internal-memory selector utility.

## Method Section Blueprint

1. Define boundary window and internal-memory candidate region.
2. Define compression-time selector score.
3. Define future readout coverage and random/oracle normalization.
4. Classify top/mixed/weak selectors.
5. Define online weighted voting under fixed dense KV budget.
6. Add current evidence floor and optional temporal coverage.
7. Clarify no future query is used online.

## Experiment Section Blueprint

1. Baseline comparison: StreamingVLM, HERMES, visual-head budget.
2. Profiling validity: top/random/bottom internal-memory coverage.
3. Method validation: MemoSelect-top vs random/bottom/boundary-only/HERMES.
4. Ablations: boundary exclusion, current floor, temporal coverage, beta/K
   sensitivity.
5. Failure analysis: Counting/Causal and boundary-dominated heads.
