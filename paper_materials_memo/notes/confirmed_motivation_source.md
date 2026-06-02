# Working Motivation Source

## One-Sentence Motivation

Streaming video KV cache compression should not simply preserve recent tokens
or high-attention memory tokens, because memory attention can be dominated by
boundary effects. The core need is to identify heads that select
future-reusable internal visual memory under a fixed KV budget.

## Controlling Thesis

MemoSelect is a boundary-aware memory-selection framework for streaming video
KV cache compression. It profiles heads by the future reusability of the
internal-memory tokens they select, then uses reliable memory-selector heads to
guide dense fixed-budget eviction.

## What This Paper Should Not Claim

- Do not claim strict semantic head roles.
- Do not claim that some heads are responsible for historical reasoning.
- Do not present SparseMM visual-head scores as the main novelty.
- Do not claim completed MemoSelect gains until the top/random/bottom selector
  experiments are actually run.
- Do not claim that all heads heavily rely on memory; some attention can be
  concentrated around the splice boundary.

## Core Writing Phrase

Memory attention is not memory retrieval.

## Paper Spine

1. Streaming video inference requires repeated visual KV cache eviction under a
   fixed budget.
2. Existing window, uniform, or average-attention strategies miss a key issue:
   not all high memory attention is useful memory retrieval.
3. Full eager pseudo-query measurements show that memory receives
   disproportionate readout attention, but whole-cache curves also show
   boundary and periodic structure.
4. Therefore, the paper should profile heads by internal-memory selection
   utility rather than by coarse memory/current attention mass.
5. MemoSelect uses boundary-aware future-reusable memory utility to identify
   memory-selector heads and uses them for fixed-budget dense eviction.
