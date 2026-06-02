# Claim Register

| Claim | Evidence ID | Strength | Allowed Wording | Avoid |
|---|---|---|---|---|
| C1: Streaming visual readout is memory-biased under full eager pseudo-query measurement. | E1 | Strong preliminary evidence | "The retained visual memory receives disproportionate pseudo-query readout compared with its token fraction." | "The model always relies on memory for all heads." |
| C2: KV-head groups show heterogeneous memory/current readout patterns. | E2 | Strong observation evidence | "KV-head groups differ in the strength of memory-biased readout." | "These are semantic memory heads" or "current heads dominate latest chunk." |
| C3: Memory attention is not equivalent to useful memory retrieval because boundary/splice-point attention can dominate memory-region mass. | E4 | Qualitative / diagnostic, needs quantified boundary ratio | "Whole-cache curves motivate boundary-aware profiling." | "Boundary attention fully explains memory attention" unless quantified. |
| C4: Visual-head budget allocation has a weak positive overall signal but is not sufficient for memory-heavy tasks. | E3 | Supported by full StreamingBench comparison | "Visual-head budget improves overall by 0.40 points but degrades Counting and Causal Reasoning." | "SparseMM proves our method works" or "visual-head score is our main novelty." |
| C5: Boundary-aware internal-memory utility can identify reliable memory-selector heads. | E6 | Pending | "We evaluate..." until experiment is complete. After completion, state result only with numbers. | Any completed-result claim before running the experiment. |
| C6: MemoSelect improves fixed-budget streaming video KV eviction. | E7 | Pending | "MemoSelect is designed to..." until QA results are complete. | Reporting gains or superiority without logs. |

## Current Paper Safety Rule

The current manuscript can be written as a method proposal plus supported
observations. It cannot yet be written as a completed experimental paper until
E6 and E7 are produced.
