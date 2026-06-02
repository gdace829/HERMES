#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GPU="${1:-0}"
DEBUG="${2:-true}"

MODEL="${MODEL:-qwen2.5_vl_7b}"
DATASET="${DATASET:-streamingbench}"
FPS="${FPS:-0.5}"
KV_SIZE="${KV_SIZE:-6000}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
DEVICES="${DEVICES:-$GPU}"
COMPRESS_MODE="${COMPRESS_MODE:-hermes}"
LAYER_BUDGET_SCORES="${LAYER_BUDGET_SCORES:-sparsemm_qwen25}"
LAYER_BUDGET_VARIABLE="${LAYER_BUDGET_VARIABLE:-true}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python or fix the hermes-qwen env." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "Project:       $PROJECT_DIR"
echo "Python:        $PYTHON_BIN"
echo "Model:         $MODEL"
echo "Dataset:       $DATASET"
echo "FPS:           $FPS"
echo "KV size:       $KV_SIZE"
echo "Compress:      $COMPRESS_MODE"
echo "Layer scores:  $LAYER_BUDGET_SCORES"
echo "Var lengths:   $LAYER_BUDGET_VARIABLE"
echo "Num chunks:    $NUM_CHUNKS"
echo "Devices:       $DEVICES"
echo "Debug:         $DEBUG"
echo ""

"$PYTHON_BIN" "$PROJECT_DIR/video_qa/run_infer.py" \
  --num_chunks "$NUM_CHUNKS" \
  --devices "$DEVICES" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --sample_fps "$FPS" \
  --kv_size "$KV_SIZE" \
  --compress_mode "$COMPRESS_MODE" \
  --layer_budget_scores "$LAYER_BUDGET_SCORES" \
  --layer_budget_variable "$LAYER_BUDGET_VARIABLE" \
  --debug "$DEBUG"
