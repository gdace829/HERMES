#!/bin/bash
# Head Analysis via Pseudo-Forward: 全外挂，不动一行原代码
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${4:-0}"

source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen

MODEL="${1:-qwen2.5_vl_7b}"
KV="${2:-100000}"
COMPRESS="${3:-streamingvlm}"
N_VIDEOS="${5:-20}"

echo "=== Head Analysis via Pseudo-Forward ==="
echo "Model:    $MODEL  |  KV: $KV  |  Compress: $COMPRESS"
echo "GPU:      $CUDA_VISIBLE_DEVICES  |  Videos: $N_VIDEOS"
echo ""

python "$SCRIPT_DIR/analyze_heads_pseudo.py" \
    --model "$MODEL" \
    --kv_size "$KV" \
    --compress_mode "$COMPRESS" \
    --sample_fps 0.5 \
    --num_videos "$N_VIDEOS" \
    --save_dir "results/head_analysis/pseudo-${MODEL}-kv${KV}-${COMPRESS}"
