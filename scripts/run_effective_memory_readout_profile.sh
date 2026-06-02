#!/bin/bash
# True-query internal-memory readout profiling.
#
# Debug:
#   GPU=1 bash scripts/run_effective_memory_readout_profile.sh
#
# Main n4/o80:
#   MODE=n4_o80 GPU=1 bash scripts/run_effective_memory_readout_profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

MODEL="${MODEL:-qwen2.5_vl_7b}"
ANNO_PATH="${ANNO_PATH:-data/streamingbench/streamingbench_realtime.json}"
FPS="${FPS:-0.5}"
KV_SIZE="${KV_SIZE:-6000}"
COMPRESS_MODE="${COMPRESS_MODE:-hermes}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-0}"
MODE="${MODE:-debug}"
TOP_K="${TOP_K:-32}"
BOUNDARY_WINDOW="${BOUNDARY_WINDOW:-64}"
CURRENT_BOUNDARY_WINDOW="${CURRENT_BOUNDARY_WINDOW:-$BOUNDARY_WINDOW}"
QUERY_POOL="${QUERY_POOL:-mean}"
LAST_N="${LAST_N:-4}"
CLASS_METRIC="${CLASS_METRIC:-readout_shape_score}"
SEED="${SEED:-0}"

case "$MODE" in
    debug)
        NUM_VIDEOS="${NUM_VIDEOS:-1}"
        MAX_QUESTIONS="${MAX_QUESTIONS:-2}"
        MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-4}"
        SAVE_DIR="${SAVE_DIR:-results/observations/effective_memory_readout_debug}"
        ;;
    smoke)
        NUM_VIDEOS="${NUM_VIDEOS:-2}"
        MAX_QUESTIONS="${MAX_QUESTIONS:-4}"
        MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-16}"
        SAVE_DIR="${SAVE_DIR:-results/observations/effective_memory_readout_smoke}"
        ;;
    n4_o80)
        NUM_VIDEOS="${NUM_VIDEOS:-4}"
        MAX_QUESTIONS="${MAX_QUESTIONS:-}"
        MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-80}"
        SAVE_DIR="${SAVE_DIR:-results/observations/effective_memory_readout_n4_o80}"
        ;;
    *)
        echo "Unknown MODE=$MODE. Use debug, smoke, or n4_o80." >&2
        exit 1
        ;;
esac

echo "Project:           $PROJECT_DIR"
echo "Python:            $PYTHON_BIN"
echo "Model:             $MODEL"
echo "Anno:              $ANNO_PATH"
echo "FPS:               $FPS"
echo "KV size:           $KV_SIZE"
echo "Compress mode:     $COMPRESS_MODE"
echo "GPU visible:       $GPU"
echo "Device in process: $DEVICE"
echo "Mode:              $MODE"
echo "Num videos:        $NUM_VIDEOS"
echo "Max questions:     ${MAX_QUESTIONS:-all}"
echo "Max observations:  $MAX_OBSERVATIONS"
echo "Top-K:             $TOP_K"
echo "Boundary window:   $BOUNDARY_WINDOW"
echo "Current boundary:  $CURRENT_BOUNDARY_WINDOW"
echo "Query pool:        $QUERY_POOL"
echo "Last N:            $LAST_N"
echo "Class metric:      $CLASS_METRIC"
echo "Save dir:          $SAVE_DIR"

CMD=(
    "$PYTHON_BIN" head_analysis/profile_effective_memory_readout.py
    --model "$MODEL"
    --anno_path "$ANNO_PATH"
    --sample_fps "$FPS"
    --kv_size "$KV_SIZE"
    --compress_mode "$COMPRESS_MODE"
    --device "$DEVICE"
    --num_videos "$NUM_VIDEOS"
    --max_observations "$MAX_OBSERVATIONS"
    --top_k "$TOP_K"
    --boundary_window "$BOUNDARY_WINDOW"
    --current_boundary_window "$CURRENT_BOUNDARY_WINDOW"
    --query_pool "$QUERY_POOL"
    --last_n "$LAST_N"
    --class_metric "$CLASS_METRIC"
    --seed "$SEED"
    --save_dir "$SAVE_DIR"
)

if [ -n "$MAX_QUESTIONS" ]; then
    CMD+=(--max_questions "$MAX_QUESTIONS")
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
