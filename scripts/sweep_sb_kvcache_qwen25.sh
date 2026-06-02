#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GPU="${1:-0}"
DEBUG="${2:-true}"

MODEL="${MODEL:-qwen2.5_vl_7b}"
DATASET="${DATASET:-streamingbench}"
FPS="${FPS:-0.5}"
KV_SIZES="${KV_SIZES:-2000 4000 6000 8000}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
DEVICES="${DEVICES:-$GPU}"
COMPRESS_MODE="${COMPRESS_MODE:-hermes}"
LAYER_BUDGET_SCORES="${LAYER_BUDGET_SCORES:-}"
LAYER_BUDGET_VARIABLE="${LAYER_BUDGET_VARIABLE:-false}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python or fix the hermes-qwen env." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "Project:       $PROJECT_DIR"
echo "Python:        $PYTHON_BIN"
echo "Model:         $MODEL"
echo "Dataset:       $DATASET"
echo "FPS:           $FPS"
echo "KV sizes:      $KV_SIZES"
echo "Compress:      $COMPRESS_MODE"
echo "Layer scores:  ${LAYER_BUDGET_SCORES:-none}"
echo "Var lengths:   $LAYER_BUDGET_VARIABLE"
echo "Num chunks:    $NUM_CHUNKS"
echo "Devices:       $DEVICES"
echo "Debug:         $DEBUG"
echo "Log dir:       $LOG_DIR"
echo ""

for KV_SIZE in $KV_SIZES; do
  LOG_FILE="$LOG_DIR/sb_${MODEL}_fps${FPS}_kv${KV_SIZE}_${COMPRESS_MODE}.log"
  cmd=(
    "$PYTHON_BIN" "$PROJECT_DIR/video_qa/run_infer.py"
    --num_chunks "$NUM_CHUNKS"
    --devices "$DEVICES"
    --model "$MODEL"
    --dataset "$DATASET"
    --sample_fps "$FPS"
    --kv_size "$KV_SIZE"
    --compress_mode "$COMPRESS_MODE"
    --debug "$DEBUG"
  )

  if [ -n "$LAYER_BUDGET_SCORES" ]; then
    cmd+=(--layer_budget_scores "$LAYER_BUDGET_SCORES")
    cmd+=(--layer_budget_variable "$LAYER_BUDGET_VARIABLE")
    LOG_FILE="$LOG_DIR/sb_${MODEL}_fps${FPS}_kv${KV_SIZE}_${COMPRESS_MODE}_${LAYER_BUDGET_SCORES}_var${LAYER_BUDGET_VARIABLE}.log"
  fi

  echo "===== KV_SIZE=$KV_SIZE ====="
  echo "Log: $LOG_FILE"
  "${cmd[@]}" 2>&1 | tee "$LOG_FILE"
done
