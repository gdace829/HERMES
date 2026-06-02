#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# StreamingBench evaluation for the fixed SparseMM-style Qwen2.5-VL path.
#
# Defaults:
#   - Qwen2.5-VL-7B
#   - StreamingBench, fps=0.5
#   - kv_size=6000
#   - SparseMM-style per-KV-head budget
#   - fixed SparseMM score loading: mean over each visual-head score list
#   - protected recent visual window: 32 tokens counted inside each head budget
#   - physical per-KV-head ragged prefill
#
# Full run on GPU0:
#   bash scripts/eval_streamingbench_qwen25_sparsemm_fixed.sh 0
#
# Full run split across 4 GPUs:
#   bash scripts/eval_streamingbench_qwen25_sparsemm_fixed.sh 0,1,2,3
#
# Smoke/debug run:
#   DEBUG=true bash scripts/eval_streamingbench_qwen25_sparsemm_fixed.sh 0
#
# Evaluate existing results only:
#   ONLY_EVAL=true bash scripts/eval_streamingbench_qwen25_sparsemm_fixed.sh 0

DEVICES="${1:-${DEVICES:-0}}"
IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICES"

MODEL="${MODEL:-qwen2.5_vl_7b}"
DATASET="${DATASET:-streamingbench}"
FPS="${FPS:-0.5}"
KV_SIZE="${KV_SIZE:-6000}"
NUM_CHUNKS="${NUM_CHUNKS:-${#DEVICE_ARRAY[@]}}"
COMPRESS_MODE="${COMPRESS_MODE:-hermes}"
DEBUG="${DEBUG:-false}"
ONLY_EVAL="${ONLY_EVAL:-false}"

KV_HEAD_BUDGET_SCORES="${KV_HEAD_BUDGET_SCORES:-/home/sjs/SparseMM/visual_head/head_score/qwen2.5-vl.json}"
KV_HEAD_BUDGET_SCHEME="${KV_HEAD_BUDGET_SCHEME:-sparsemm}"
KV_HEAD_BUDGET_SPARSEMM_RATIO="${KV_HEAD_BUDGET_SPARSEMM_RATIO:-0.1}"
KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE="${KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE:-32}"
KV_HEAD_BUDGET_UNION_CAP_RATIO="${KV_HEAD_BUDGET_UNION_CAP_RATIO:-1.0}"
KV_HEAD_BUDGET_MAX_MASK_Q_LEN="${KV_HEAD_BUDGET_MAX_MASK_Q_LEN:-128}"
KV_HEAD_RAGGED_PREFILL="${KV_HEAD_RAGGED_PREFILL:-true}"

PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python or fix the hermes-qwen env." >&2
  exit 1
fi

if [ "$NUM_CHUNKS" -gt "${#DEVICE_ARRAY[@]}" ]; then
  echo "NUM_CHUNKS=$NUM_CHUNKS but only ${#DEVICE_ARRAY[@]} device(s) were provided: $DEVICES" >&2
  exit 1
fi

if [ ! -f "$KV_HEAD_BUDGET_SCORES" ]; then
  echo "KV-head score file not found: $KV_HEAD_BUDGET_SCORES" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

format_g() {
  "$PYTHON_BIN" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$1"
}

SCORE_TAG="$(basename "$KV_HEAD_BUDGET_SCORES")"
SCORE_TAG="${SCORE_TAG%.*}"
UNION_TAG="$(format_g "$KV_HEAD_BUDGET_UNION_CAP_RATIO")"
RATIO_TAG="$(format_g "$KV_HEAD_BUDGET_SPARSEMM_RATIO")"
OUT_DIR="$PROJECT_DIR/results/$MODEL/$DATASET/fps${FPS}-kv${KV_SIZE}-${COMPRESS_MODE}-kvheadbudget-${SCORE_TAG}-${KV_HEAD_BUDGET_SCHEME}-union${UNION_TAG}-r${RATIO_TAG}-w${KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE}-raggedprefill"

RUN_NAME="streamingbench-qwen25-kv${KV_SIZE}-${KV_HEAD_BUDGET_SCHEME}-${SCORE_TAG}-meanloader-protectedw${KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE}-r${RATIO_TAG}-raggedprefill"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"

cmd=(
  "$PYTHON_BIN" "$PROJECT_DIR/video_qa/run_infer.py"
  --num_chunks "$NUM_CHUNKS"
  --devices "$DEVICES"
  --model "$MODEL"
  --dataset "$DATASET"
  --sample_fps "$FPS"
  --kv_size "$KV_SIZE"
  --compress_mode "$COMPRESS_MODE"
  --kv_head_budget_scores "$KV_HEAD_BUDGET_SCORES"
  --kv_head_budget_scheme "$KV_HEAD_BUDGET_SCHEME"
  --kv_head_budget_sparsemm_ratio "$KV_HEAD_BUDGET_SPARSEMM_RATIO"
  --kv_head_budget_sparsemm_window_size "$KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE"
  --kv_head_budget_union_cap_ratio "$KV_HEAD_BUDGET_UNION_CAP_RATIO"
  --kv_head_budget_max_mask_q_len "$KV_HEAD_BUDGET_MAX_MASK_Q_LEN"
  --kv_head_ragged_prefill "$KV_HEAD_RAGGED_PREFILL"
  --debug "$DEBUG"
)

if [ "$ONLY_EVAL" = "true" ]; then
  cmd+=(--only_eval)
fi

echo "Project:        $PROJECT_DIR"
echo "Python:         $PYTHON_BIN"
echo "Model:          $MODEL"
echo "Dataset:        $DATASET"
echo "FPS:            $FPS"
echo "KV size:        $KV_SIZE"
echo "Compress:       $COMPRESS_MODE"
echo "KV-head score:  $KV_HEAD_BUDGET_SCORES"
echo "Budget scheme:  $KV_HEAD_BUDGET_SCHEME"
echo "SparseMM ratio: $KV_HEAD_BUDGET_SPARSEMM_RATIO"
echo "SparseMM win:   $KV_HEAD_BUDGET_SPARSEMM_WINDOW_SIZE"
echo "Union cap:      $KV_HEAD_BUDGET_UNION_CAP_RATIO"
echo "Max mask q:     $KV_HEAD_BUDGET_MAX_MASK_Q_LEN"
echo "Ragged prefill: $KV_HEAD_RAGGED_PREFILL"
echo "Num chunks:     $NUM_CHUNKS"
echo "Devices:        $DEVICES"
echo "Debug:          $DEBUG"
echo "Only eval:      $ONLY_EVAL"
echo "Output dir:     $OUT_DIR"
echo "Log file:       $LOG_FILE"
echo ""
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
