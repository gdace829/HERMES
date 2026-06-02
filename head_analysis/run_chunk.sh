#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${4:-0}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
MODEL="${1:-qwen2.5_vl_7b}"
KV="${2:-6000}"
COMPRESS="${3:-hermes}"
NV="${5:-20}"
echo "=== Chunk Hit Analysis ==="
echo "Model: $MODEL | KV: $KV | Compress: $COMPRESS | Videos: $NV"
python "$SCRIPT_DIR/chunk_hit.py" \
    --model "$MODEL" --kv_size "$KV" --compress_mode "$COMPRESS" \
    --sample_fps 0.5 --num_videos "$NV" \
    --save_dir "results/head_analysis/chunk_hit-${MODEL}-kv${KV}-${COMPRESS}"
