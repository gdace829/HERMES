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
KV_HEAD_BUDGET_SCORES="${KV_HEAD_BUDGET_SCORES:-sparsemm_qwen25}"
KV_HEAD_BUDGET_UNION_CAP_RATIO="${KV_HEAD_BUDGET_UNION_CAP_RATIO:-1.0}"
KV_HEAD_BUDGET_MAX_MASK_Q_LEN="${KV_HEAD_BUDGET_MAX_MASK_Q_LEN:-128}"
KV_HEAD_BUDGET_SCHEME="${KV_HEAD_BUDGET_SCHEME:-sparsemm}"
KV_HEAD_BUDGET_SPARSEMM_RATIO="${KV_HEAD_BUDGET_SPARSEMM_RATIO:-0.1}"
KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE="${KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE:-32}"
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
echo "KV-head score: $KV_HEAD_BUDGET_SCORES"
echo "Budget scheme: $KV_HEAD_BUDGET_SCHEME"
echo "SparseMM ratio:$KV_HEAD_BUDGET_SPARSEMM_RATIO"
echo "SparseMM win:  $KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE"
echo "Union cap:     $KV_HEAD_BUDGET_UNION_CAP_RATIO"
echo "Max mask q:    $KV_HEAD_BUDGET_MAX_MASK_Q_LEN"
echo "Ragged decode: true"
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
  --kv_head_budget_scores "$KV_HEAD_BUDGET_SCORES" \
  --kv_head_budget_scheme "$KV_HEAD_BUDGET_SCHEME" \
  --kv_head_budget_sparsemm_ratio "$KV_HEAD_BUDGET_SPARSEMM_RATIO" \
  --kv_head_budget_sparsemm_window_size "$KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE" \
  --kv_head_budget_union_cap_ratio "$KV_HEAD_BUDGET_UNION_CAP_RATIO" \
  --kv_head_budget_max_mask_q_len "$KV_HEAD_BUDGET_MAX_MASK_Q_LEN" \
  --kv_head_ragged_decode true \
  --debug "$DEBUG"
