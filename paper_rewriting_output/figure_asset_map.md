# Figure Asset Map

| Figure ID | Source Image | Intended Caption | Target Location | LaTeX Label |
|---|---|---|---|---|
| F1 | `results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_s_current_share_heatmap.png` | Query-robust current-share heatmap over layer-KV-head groups. Shows heterogeneous readout allocation but not selector utility. | Observation section | `fig:kv-current-share` |
| F2 | `results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_b_log_per_token_ratio_heatmap.png` | Token-normalized memory/current density heatmap. Used as observation baseline rather than method score. | Observation section | `fig:kv-density-ratio` |
| F3 | TBD exact path | Whole-cache readout attention curve showing boundary and periodic structure. | Motivation / Observation section | `fig:whole-cache-readout` |
| F4 | TBD exact path | Top vs bottom visual-head readout curves. Used only as diagnostic that head readout patterns differ. | Appendix or motivation | `fig:visual-head-readout` |
| F5 | Needed | Boundary attention ratio heatmap. | Observation section after running boundary profiling | `fig:boundary-ratio` |
| F6 | Needed | Boundary-aware internal-memory utility heatmap. | Method/profiling section | `fig:internal-memory-utility` |
| F7 | Needed | Top/random/bottom selector future-readout coverage bar. | Main profiling validity result | `fig:selector-coverage` |

## Asset Rule

Do not copy images into `final_paper/figures/` until the figure path, plotting
script, and caption are verified.
