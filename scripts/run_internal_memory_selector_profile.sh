#!/bin/bash
# Boundary-aware internal-memory selector profiling.
#
# Default smoke run:
#   GPU=1 bash scripts/run_internal_memory_selector_profile.sh
#
# Larger run:
#   MODE=n4_o80 GPU=1 bash scripts/run_internal_memory_selector_profile.sh

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
TOP_K="${TOP_K:-128}"
BOUNDARY_WINDOW="${BOUNDARY_WINDOW:-64}"
RANDOM_TRIALS_ENV="${RANDOM_TRIALS:-}"
FUTURE_HEAD_POOL="${FUTURE_HEAD_POOL:-same_kv}"
SEED="${SEED:-0}"

case "$MODE" in
    debug)
        NUM_VIDEOS="${NUM_VIDEOS:-1}"
        MAX_QUESTIONS="${MAX_QUESTIONS:-2}"
        MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-4}"
        RANDOM_TRIALS="${RANDOM_TRIALS_ENV:-8}"
        SAVE_DIR="${SAVE_DIR:-results/observations/internal_memory_selector_debug}"
        ;;
    smoke)
        NUM_VIDEOS="${NUM_VIDEOS:-2}"
        MAX_QUESTIONS="${MAX_QUESTIONS:-4}"
        MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-16}"
        RANDOM_TRIALS="${RANDOM_TRIALS_ENV:-16}"
        SAVE_DIR="${SAVE_DIR:-results/observations/internal_memory_selector_smoke}"
        ;;
    n4_o80)
        NUM_VIDEOS="${NUM_VIDEOS:-4}"
        MAX_QUESTIONS="${MAX_QUESTIONS:-}"
        MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-80}"
        RANDOM_TRIALS="${RANDOM_TRIALS_ENV:-32}"
        SAVE_DIR="${SAVE_DIR:-results/observations/internal_memory_selector_n4_o80}"
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
echo "Future head pool:  $FUTURE_HEAD_POOL"
echo "Save dir:          $SAVE_DIR"

CMD=(
    "$PYTHON_BIN" head_analysis/profile_internal_memory_selector.py
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
    --random_trials "$RANDOM_TRIALS"
    --future_head_pool "$FUTURE_HEAD_POOL"
    --seed "$SEED"
    --save_dir "$SAVE_DIR"
)

if [ -n "$MAX_QUESTIONS" ]; then
    CMD+=(--max_questions "$MAX_QUESTIONS")
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
