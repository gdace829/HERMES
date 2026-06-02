# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M1 | Build future-reusable memory utility u_h | Static profiling | StreamingBench calibration n16/o500 | coverage, u_h, top-bottom gap | MUST | TODO | Reuse eager profiling if future readout data is sufficient. |
| R002 | M1 | Validate selector separability | top vs random vs bottom u_h | Same as R001 | coverage bar, normalized utility | MUST | TODO | This is the first go/no-go gate. |
| R003 | M1 | Stability of u_h | video split / chunk split | Same as R001 | corr, top-head overlap | MUST | TODO | Needed for paper credibility. |
| R004 | M2 | Implement dense MemoSelect policy | code sanity | one video / 5 QA | no crash, fixed B=6000 | MUST | TODO | No ragged KV and no extra budget. |
| R005 | M2 | Isolate recency floor | current-floor only | StreamingBench smoke | acc, delta vs HERMES | MUST | TODO | W_c=32, counted in budget. |
| R006 | M2 | Selector smoke | MemoSelect-top | StreamingBench smoke | acc, per-task delta | MUST | TODO | beta in {0.1, 0.2, 0.5}. |
| R007 | M2 | Random control | MemoSelect-random | StreamingBench smoke | acc, delta vs top | MUST | TODO | Layer-matched random, seed 0 first. |
| R008 | M2 | Negative selector control | MemoSelect-bottom | StreamingBench smoke | acc, delta vs top | MUST | TODO | Should underperform top if u_h is meaningful. |
| R009 | M3 | n50 validation | HERMES / floor / top / random / bottom | StreamingBench n50 | overall and per-task acc | MUST | TODO | Run only after R002 and R006 are positive. |
| R010 | M3 | Random variance | MemoSelect-random seeds 0/1/2 | StreamingBench n50 | mean/std acc | MUST | TODO | Needed if top-random gap is small. |
| R011 | M4 | Temporal coverage ablation | MemoSelect-top vs full | StreamingBench n50 or smoke | acc, temporal coverage | NICE | TODO | Keep only if it helps. |
| R012 | M4 | Budget sensitivity | B=6000/4000/3000 | StreamingBench subset | acc vs budget | NICE | TODO | Appendix unless very strong. |
