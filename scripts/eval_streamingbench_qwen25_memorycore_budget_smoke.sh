#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Smoke/full StreamingBench evaluation for memory-core KV-head budget scores.
#
# Smoke on GPU1:
#   DEBUG=true bash scripts/eval_streamingbench_qwen25_memorycore_budget_smoke.sh 1 ours
#
# Modes:
#   ours | uniform | inverted | random | sparsemm
#
# Common overrides:
#   KV_SIZE=6000 RATIO=0.2 WINDOW=32 SCHEME=sparsemm ...

DEVICES="${1:-${DEVICES:-1}}"
MODE="${2:-${MODE:-ours}}"
IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICES"

PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"
MODEL="${MODEL:-qwen2.5_vl_7b}"
DATASET="${DATASET:-streamingbench}"
FPS="${FPS:-0.5}"
KV_SIZE="${KV_SIZE:-6000}"
NUM_CHUNKS="${NUM_CHUNKS:-${#DEVICE_ARRAY[@]}}"
COMPRESS_MODE="${COMPRESS_MODE:-hermes}"
DEBUG="${DEBUG:-true}"
ONLY_EVAL="${ONLY_EVAL:-false}"

PROFILE_CSV="${PROFILE_CSV:-results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv}"
SCORE_DIR="${SCORE_DIR:-results/observations/memory_core_budget_scores}"
SEED="${SEED:-0}"
SCHEME="${SCHEME:-sparsemm}"
RATIO="${RATIO:-0.2}"
WINDOW="${WINDOW:-32}"
UNION_CAP="${UNION_CAP:-1.0}"
MAX_MASK_Q_LEN="${MAX_MASK_Q_LEN:-128}"
RAGGED_PREFILL="${RAGGED_PREFILL:-true}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$SCORE_DIR" logs

if [ "$MODE" != "sparsemm" ]; then
  "$PYTHON_BIN" head_analysis/build_memory_core_budget_scores.py \
    --profile_csv "$PROFILE_CSV" \
    --save_dir "$SCORE_DIR" \
    --seed "$SEED"
fi

case "$MODE" in
  ours)
    SCORE_PATH="$SCORE_DIR/ours_top100.csv"
    ;;
  uniform)
    SCORE_PATH="$SCORE_DIR/uniform.csv"
    ;;
  inverted)
    SCORE_PATH="$SCORE_DIR/inverted_top100.csv"
    ;;
  random)
    SCORE_PATH="$SCORE_DIR/random_top100_seed${SEED}.csv"
    ;;
  sparsemm)
    SCORE_PATH="${SPARSEMM_SCORE_PATH:-/home/sjs/SparseMM/visual_head/head_score/qwen2.5-vl.json}"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use ours, uniform, inverted, random, sparsemm." >&2
    exit 1
    ;;
esac

if [ ! -f "$SCORE_PATH" ]; then
  echo "Score file not found: $SCORE_PATH" >&2
  exit 1
fi

RUN_NAME="streamingbench-qwen25-kv${KV_SIZE}-memorycore-${MODE}-${SCHEME}-r${RATIO}-w${WINDOW}-debug${DEBUG}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

cmd=(
  "$PYTHON_BIN" video_qa/run_infer.py
  --num_chunks "$NUM_CHUNKS"
  --devices "$DEVICES"
  --model "$MODEL"
  --dataset "$DATASET"
  --sample_fps "$FPS"
  --kv_size "$KV_SIZE"
  --compress_mode "$COMPRESS_MODE"
  --kv_head_budget_scores "$SCORE_PATH"
  --kv_head_budget_scheme "$SCHEME"
  --kv_head_budget_sparsemm_ratio "$RATIO"
  --kv_head_budget_sparsemm_window_size "$WINDOW"
  --kv_head_budget_union_cap_ratio "$UNION_CAP"
  --kv_head_budget_max_mask_q_len "$MAX_MASK_Q_LEN"
  --kv_head_ragged_prefill "$RAGGED_PREFILL"
  --debug "$DEBUG"
)

if [ "$ONLY_EVAL" = "true" ]; then
  cmd+=(--only_eval)
fi

echo "Project:        $PROJECT_DIR"
echo "Mode:           $MODE"
echo "Score path:     $SCORE_PATH"
echo "Devices:        $DEVICES"
echo "Debug:          $DEBUG"
echo "Scheme:         $SCHEME"
echo "Ratio/window:   $RATIO / $WINDOW"
echo "Ragged prefill: $RAGGED_PREFILL"
echo "Log file:       $LOG_FILE"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
