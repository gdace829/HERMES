#!/bin/bash
# Run HERMES (attention-guided) on StreamingBench with Qwen2.5-VL-7B

set -euo pipefail

# --- config ---
MODEL="qwen2.5_vl_7b"
DATASET="streamingbench"
FPS="0.5"
KV_SIZE="6000"
NUM_CHUNKS="1"
COMPRESS_MODE="hermes"  # attention-guided 压缩

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
