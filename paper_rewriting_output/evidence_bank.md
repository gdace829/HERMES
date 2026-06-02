# Evidence Bank

Only claims supported by user-provided local materials are listed as evidence.
Pending experiments are explicitly marked as needed.

| Evidence ID | Source File | Claim Supported | Figure/Table Link | Verification Needed |
|---|---|---|---|---|
| E1 | `paper_materials_memo/results/attention_observations_summary.md`; raw: `results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper/summary.json` | Streaming visual readout is memory-biased: latest chunk has high token fraction but much lower pseudo-query attention share. | Candidate Table: attention statistics; Candidate Figure: current-share heatmap | Verify final numbers directly from JSON before paper submission. |
| E2 | `paper_materials_memo/results/attention_observations_summary.md`; raw: `results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_head_profile_summary.json` | KV-head groups differ in current/memory readout strength, but this is observation only and not sufficient for method scoring. | Candidate Figure F1/F2 | Ensure wording does not claim all heads deeply retrieve memory. |
| E3 | `paper_materials_memo/results/baseline_results_summary.md`; raw StreamingBench result CSVs | StreamingVLM, HERMES, and visual-head budget show complementary failure modes. Visual-head budget gives weak overall gain but hurts Counting/Causal/Spatial. | Candidate Table: baseline comparison | Recompute from CSV before final table. |
| E4 | `paper_materials_memo/figures/figure_manifest.md` | Whole-cache readout curves suggest boundary and periodic peaks, motivating boundary-aware profiling. | Candidate Figure F3/F4 | Exact figure paths and plotting settings need confirmation. |
| E5 | `paper_materials_memo/references/reference_manifest.md`; local `111.pdf` | Forcing-KV is a writing-logic exemplar for observation -> profiling -> compression -> ablation, but not a method to copy. | Related-work / writing pattern | Need formal citation metadata if used in paper. |
| E6 | Pending experiment | Boundary-aware internal-memory utility `u_h^int` separates reliable selectors from boundary-dominated heads. | Needed Figure F5/F6/F7 | Must run experiment before claiming. |
| E7 | Pending experiment | MemoSelect-top improves fixed-budget eviction over random/bottom/boundary-only selectors. | Needed main result table | Must run experiment before claiming. |
