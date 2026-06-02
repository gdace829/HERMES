#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${4:-0}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
M="${1:-qwen2.5_vl_7b}"; KV="${2:-6000}"; C="${3:-hermes}"; NV="${5:-5}"
echo "=== Chunk Label + Hit ==="
python "$SCRIPT_DIR/chunk_label_hit.py" \
    --model "$M" --kv_size "$KV" --compress_mode "$C" \
    --sample_fps 0.5 --num_videos "$NV" \
    --save_dir "results/head_analysis/chunk_label_hit-${M}"
