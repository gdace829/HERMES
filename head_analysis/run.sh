#!/bin/bash
# Head Analysis: 分析 attention head 在记忆依赖 vs 近期依赖任务上的差异
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen

MODEL="${1:-qwen2.5_vl_7b}"
KV_SIZE="${2:-6000}"
COMPRESS_MODE="${3:-hermes}"
GPU="${4:-0}"      # GPU 编号
DEBUG="${5:-}"     # 传 "debug" 则只跑 5 个视频

export CUDA_VISIBLE_DEVICES="$GPU"

DEBUG_FLAG=""
SAVE_SUFFIX="${MODEL}-kv${KV_SIZE}-${COMPRESS_MODE}"

if [ "$DEBUG" = "debug" ]; then
    DEBUG_FLAG="--debug"
    SAVE_SUFFIX="${SAVE_SUFFIX}-debug"
    echo "!!! DEBUG MODE: only 5 videos !!!"
fi

echo "=== Head Analysis ==="
echo "Model:    $MODEL"
echo "KV size:  $KV_SIZE"
echo "Compress: $COMPRESS_MODE"
echo ""

python "$SCRIPT_DIR/run_analysis.py" \
    --model "$MODEL" \
    --kv_size "$KV_SIZE" \
    --compress_mode "$COMPRESS_MODE" \
    --sample_fps 0.5 \
    --save_dir "results/head_analysis/${SAVE_SUFFIX}" \
    $DEBUG_FLAG

echo ""
echo "=== Visualization ==="
python "$SCRIPT_DIR/visualize.py" \
    --scores "results/head_analysis/${SAVE_SUFFIX}/head_scores.npz" \
    --save_dir "results/head_analysis/${SAVE_SUFFIX}"
