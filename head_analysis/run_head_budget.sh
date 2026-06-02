#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${4:-0}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen

SCORES="${1:-sparsemm}"  # sparsemm | pseudo | path
M="${2:-qwen2.5_vl_7b}"
KV="${3:-6000}"
NV="${5:-10}"

echo "=== HERMES + Head-Level Budget ==="
echo "Scores: $SCORES | Model: $M | KV: $KV | Videos: $NV"

python "$SCRIPT_DIR/run_head_budget.py" \
    --model "$M" --kv_size "$KV" --compress_mode hermes \
    --scores "$SCORES" --num_videos "$NV" --device 0 \
    --save_dir "results/head_budget/$(echo $SCORES | tr '/' '_')"
