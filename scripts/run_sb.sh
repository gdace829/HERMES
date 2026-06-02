#!/bin/bash
# Run HERMES on StreamingBench with Qwen2.5-VL-7B

set -euo pipefail

# --- config ---
MODEL="qwen2.5_vl_7b"
DATASET="streamingbench"
FPS="0.5"
KV_SIZE="6000"
NUM_CHUNKS="1"          # 改为 2 可以双 GPU 并行
COMPRESS_MODE="streamingvlm"  # hermes (attention-guided) 或 streamingvlm (滑动窗口)

# --- setup ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen

echo "Model:     $MODEL"
echo "Dataset:   $DATASET"
echo "FPS:       $FPS"
echo "KV size:   $KV_SIZE"
echo "Chunks:    $NUM_CHUNKS"
echo "Compress:  $COMPRESS_MODE"
echo ""

python video_qa/run_infer.py \
    --num_chunks "$NUM_CHUNKS" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --sample_fps "$FPS" \
    --kv_size "$KV_SIZE" \
    --compress_mode "$COMPRESS_MODE"
