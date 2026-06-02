# Figure Manifest

This file lists candidate figures. Images should be copied or linked into this
folder only after manual approval.

| Figure ID | Candidate Source | Intended Use | Status |
|---|---|---|---|
| F1 | `results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_s_current_share_heatmap.png` | Show current-share heterogeneity across KV-head groups. | Existing |
| F2 | `results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_b_log_per_token_ratio_heatmap.png` | Show memory/current density variation. Observation only, not method score. | Existing |
| F3 | whole-cache readout attention curve previously plotted | Show boundary and periodic peaks. | Need exact path confirmation |
| F4 | top/bottom SparseMM visual-head readout curves | Show visual-head groups differ in memory readout density. | Need exact path confirmation |
| F5 | boundary ratio heatmap | Diagnose boundary-attending heads. | Needed |
| F6 | internal-memory utility heatmap | Main profiling figure for MemoSelect. | Needed |
| F7 | top/random/bottom future coverage bar | Show memory-selector utility is meaningful. | Needed |

## Figure Policy

Do not use a figure as evidence unless its source path is known and the plotted
quantity is documented. Existing readout curves should be treated as
diagnostic until their exact generation script and settings are recorded.
