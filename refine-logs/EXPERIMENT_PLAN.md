# Experiment Plan

**Problem**: Streaming video VLMs must evict visual KV cache under a fixed budget, but average attention or static per-head budget can dilute the heads that are actually good at selecting useful memory.

**Method Thesis**: MemoSelect identifies reliable memory-selector heads offline and uses them as weighted voters for fixed-budget dense KV eviction during streaming inference.

**Date**: 2026-05-26

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: Some KV-head groups are better memory selectors. | This is the observation that makes head-guided eviction meaningful. | Top-utility heads select memory tokens with higher future readout coverage than random and bottom heads, under the same top-K size. | B1 |
| C2: Memory-selector voting improves eviction under the same KV budget. | This turns the observation into a method contribution. | MemoSelect-top beats random/bottom/no-vote at the same 6K budget, ideally with gains on memory-heavy tasks. | B2, B3 |
| Anti-claim: The gain is just from preserving more recent tokens or using more cache. | Reviewers will reject the method if budget or recency explains the result. | All variants use the same total KV budget; current floor is isolated as its own baseline. | B2, B3 |

## Paper Storyline

Main paper must prove:

- Streaming visual KV cache has memory-biased readout and head-dependent memory-selection reliability.
- The useful profiling signal is not semantic head role labeling, but measurable memory selection utility.
- A dense fixed-budget eviction policy can exploit reliable memory selectors without ragged per-head cache or attention kernel changes.

Appendix can support:

- More videos for profiling stability.
- Budget sweep at 4K and 3K.
- Extra visualizations of whole-cache readout periodicity.

Experiments intentionally cut for now:

- Full semantic function proof for head roles.
- Physical ragged per-head KV as the main method.
- Large masking study by question type before the compression method works.

## Experiment Blocks

### Block 1: Static Memory-Selector Profiling

- Claim tested: C1.
- Why this block exists: It must show that memory-selector heads can be defined by an objective score, not by manual semantic interpretation.
- Dataset / split / task: StreamingBench calibration videos, starting with the existing n16/o500 eager profiling split.
- Compared systems: top u_h heads, random layer-matched heads, bottom u_h heads, oracle future top-K.
- Metrics: future readout coverage, normalized utility u_h, top-bottom gap, video-split and chunk-split stability.
- Setup details: Use compression-time pseudo-query attention to select K memory tokens per head; use future query or future pseudo-query readout as F_o; normalize against random and oracle.
- Success criterion: top u_h coverage is clearly above random and bottom; top-head overlap or split correlation is stable enough to reuse across videos.
- Failure interpretation: If top/random/bottom are close, b_h/readout differences are only descriptive and should not drive a method.
- Table / figure target: Section 3 observation table, u_h heatmap/histogram, top-random-bottom coverage bar.
- Priority: MUST-RUN.

### Block 2: Dense MemoSelect Smoke

- Claim tested: C2 and anti-claim.
- Why this block exists: It checks whether the profiling signal survives inside real StreamingBench QA, before spending GPU on full runs.
- Dataset / split / task: StreamingBench smoke, 5 to 10 questions per task first; then 20 per task if positive.
- Compared systems: HERMES baseline, current-floor only, MemoSelect-bottom, MemoSelect-random, MemoSelect-top.
- Metrics: overall accuracy, per-task accuracy, delta vs HERMES, delta vs current-floor.
- Setup details: Fixed dense KV budget B=6000. Current floor W_c=32 counts inside the budget. No ragged per-head cache.
- Success criterion: MemoSelect-top is better than random and bottom; if it ties HERMES but beats random/bottom, the profiling signal is directionally useful.
- Failure interpretation: If top/random/bottom are identical, the online scoring may be too weak or u_h profiling does not transfer.
- Table / figure target: First method validation table.
- Priority: MUST-RUN.

### Block 3: MemoSelect Full Ablation

- Claim tested: C2 and method component necessity.
- Why this block exists: It turns the smoke result into a paper-ready method table.
- Dataset / split / task: StreamingBench, at least n50 per target task; expand to full if the n50 trend is positive.
- Compared systems: HERMES, current-floor, MemoSelect-top, MemoSelect-random with 3 seeds, MemoSelect-bottom, MemoSelect-full with temporal coverage.
- Metrics: overall and per-task accuracy; memory-heavy tasks should be highlighted separately.
- Setup details: B=6000 primary; beta sweep {0.1, 0.2, 0.5}; tau fixed after smoke; W_c=32.
- Success criterion: MemoSelect-top or MemoSelect-full improves over current-floor and random, with no hidden budget increase.
- Failure interpretation: If temporal coverage helps but selector voting does not, pivot method toward temporal memory coverage.
- Table / figure target: Main results table and ablation table.
- Priority: MUST-RUN after B2 passes.

### Block 4: Periodic Readout / Temporal Coverage Diagnosis

- Claim tested: Supporting claim only.
- Why this block exists: It explains why pure global top-K may over-concentrate memory retention.
- Dataset / split / task: Existing whole-cache readout attention curves plus selected new examples.
- Compared systems: no coverage vs top1-per-bin coverage, K in {32, 64, 128}.
- Metrics: QA accuracy, retained-token temporal coverage, visualized readout curves.
- Setup details: Coverage anchors count inside B=6000.
- Success criterion: Coverage improves memory-heavy tasks or stabilizes performance under low budgets.
- Failure interpretation: If it does not help, keep periodicity as observation but remove coverage from the main method.
- Table / figure target: Appendix or a small ablation in main paper.
- Priority: NICE-TO-HAVE until selector voting works.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Freeze terminology and method target. | Use "memory selector", not previous/current role language. | Paper claim is one sentence: reliable heads select more reusable memory. | Done | None |
| M1 | Implement and run u_h profiling. | Static utility profiling on existing n16/o500 outputs if enough future readout exists; otherwise add future-readout collection. | top > random > bottom in coverage. | Low to medium | Need correct future readout definition. |
| M2 | Implement dense MemoSelect voting. | HERMES, floor-only, top/random/bottom on smoke. | top beats random/bottom or at least inverted/bottom. | Medium | Bonus scale beta may be too weak/strong. |
| M3 | n50 method validation. | Same variants on n50, random seeds 0/1/2. | Stable top/random gap. | Medium to high | StreamingBench variance. |
| M4 | Final polish. | Figures, stability table, budget sweep. | Main claim has both observation and compression evidence. | Medium | Too many sections; keep appendix lean. |

## Compute and Data Budget

- Total estimated GPU-hours before go/no-go: 8 to 24 GPU-hours, depending on whether future readout must be recollected.
- Total estimated GPU-hours for n50 validation: 1 to 3 days on one GPU if all variants are run serially.
- Data preparation needs: align compression observations with future readout distribution.
- Biggest bottleneck: implementing u_h cleanly without leaking future information into online inference.

## Risks and Mitigations

- Risk: b_kv top heads do not match useful memory selectors.
- Mitigation: Do not use b_kv as the method score; use u_h directly.

- Risk: current floor alone explains gains.
- Mitigation: Always include current-floor-only baseline with the same total B=6000.

- Risk: dense voting has no effect because HERMES score dominates.
- Mitigation: Sweep beta and normalize per-head scores over memory tokens before voting.

- Risk: top/random/bottom differ in profiling but not QA.
- Mitigation: Report profiling as observation, then pivot method toward temporal coverage or stronger memory-token score fusion.

## Final Checklist

- [ ] Static u_h profiling table is generated.
- [ ] u_h heatmap/histogram and top-random-bottom coverage bar are generated.
- [ ] Dense MemoSelect smoke passes with same B=6000.
- [ ] Current floor is isolated.
- [ ] Random selector uses layer-matched seeds.
- [ ] Main paper avoids semantic head-role claims.
- [ ] Ragged per-head KV is not the first-version main method.
