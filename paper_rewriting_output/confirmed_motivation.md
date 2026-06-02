# Confirmed Working Motivation

Status: working motivation confirmed by project discussion, pending final
update after boundary-aware `u_h^int` profiling results.

## Controlling Motivation

Memory attention is not memory retrieval. In streaming video KV cache
compression, memory-region attention can be dominated by boundary or splice
points, while the actual compression objective is to retain internal visual
memory that future queries will reuse.

## MemoSelect Thesis

MemoSelect profiles heads by boundary-aware future-reusable internal-memory
utility and uses reliable memory-selector heads to guide dense fixed-budget KV
cache eviction for streaming video understanding.

## Why This Is Different From Region-Mass Head Profiling

Region-mass profiling asks where a head attends. MemoSelect asks whether the
memory tokens selected by a head are later reused by future queries. The method
therefore profiles selector reliability rather than coarse attention location.

## Current Evidence Boundary

Existing evidence supports:

- memory-biased pseudo-query readout;
- heterogeneous KV-head readout patterns;
- weak positive signal from visual-head budget allocation;
- failure of purely recent or visual-head budget strategies on memory-heavy
  tasks.

Evidence still needed:

- boundary-aware internal-memory utility `u_h^int`;
- top/random/bottom selector coverage;
- MemoSelect fixed-budget QA performance.
