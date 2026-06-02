#!/bin/bash
# Run Forcing-KV-style head-wise context-access ablations on StreamingBench.
#
# Defaults run a small smoke test:
#   bash scripts/run_context_denial_ablation.sh
#
# Formal n=50 run:
#   MODE=n50 CROSS=1 RANDOM_SEEDS=0,1,2 bash scripts/run_context_denial_ablation.sh
#
# Use another visible GPU:
#   GPU=1 bash scripts/run_context_denial_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "${SKIP_CONDA:-0}" != "1" ] && [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-hermes-qwen}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-qwen2.5_vl_7b}"
ANNO_PATH="${ANNO_PATH:-data/streamingbench/streamingbench_realtime.json}"
FPS="${FPS:-0.5}"
KV_SIZE="${KV_SIZE:-6000}"
COMPRESS_MODE="${COMPRESS_MODE:-hermes}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-smoke}"
CROSS="${CROSS:-0}"
MAX_MASK_Q_LEN="${MAX_MASK_Q_LEN:-512}"
HEAD_PROFILE_CSV="${HEAD_PROFILE_CSV:-results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full/head_profile_scores.csv}"
RAW_PROFILE_CSV="${RAW_PROFILE_CSV:-results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full/raw_prev_current_attention.csv}"
TASKS="${TASKS:-Counting,Causal Reasoning,Clips Summarize,Prospective Reasoning,Attribute Recognition,Object Recognition,Action Recognition}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0}"
HEAD_GRANULARITY="${HEAD_GRANULARITY:-query}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
KV_AGGREGATION="${KV_AGGREGATION:-mean}"
KV_SCORE_MODE="${KV_SCORE_MODE:-aggregate}"

if [ -z "${PROFILE_CSV+x}" ]; then
    if [ "$HEAD_GRANULARITY" = "kv" ] && [ "$KV_SCORE_MODE" = "pooled" ]; then
        PROFILE_CSV="$RAW_PROFILE_CSV"
    else
        PROFILE_CSV="$HEAD_PROFILE_CSV"
    fi
fi

if [ -z "${HEAD_CLASSES+x}" ]; then
    if [ "$HEAD_GRANULARITY" = "kv" ]; then
        HEAD_CLASSES="results/observations/head_classes_prev_current/head_classes_kv_${KV_SCORE_MODE}_${KV_AGGREGATION}.json"
    else
        HEAD_CLASSES="results/observations/head_classes_prev_current/head_classes.json"
    fi
fi

case "$MODE" in
    smoke)
        MAX_QUESTIONS_PER_TASK="${MAX_QUESTIONS_PER_TASK:-20}"
        OUT_ROOT="${OUT_ROOT:-results/observations/obs_context_denial_smoke}"
        ;;
    n50|formal)
        MAX_QUESTIONS_PER_TASK="${MAX_QUESTIONS_PER_TASK:-50}"
        OUT_ROOT="${OUT_ROOT:-results/observations/obs_context_denial_n50}"
        ;;
    n100)
        MAX_QUESTIONS_PER_TASK="${MAX_QUESTIONS_PER_TASK:-100}"
        OUT_ROOT="${OUT_ROOT:-results/observations/obs_context_denial_n100}"
        ;;
    *)
        echo "Unknown MODE=$MODE. Use smoke, n50/formal, or n100." >&2
        exit 1
        ;;
esac

echo "Project:      $PROJECT_DIR"
echo "Model:        $MODEL"
echo "Anno:         $ANNO_PATH"
echo "FPS:          $FPS"
echo "KV size:      $KV_SIZE"
echo "Compress:     $COMPRESS_MODE"
echo "GPU:          $GPU"
echo "Device:       $DEVICE"
echo "Mode:         $MODE"
echo "Questions:    $MAX_QUESTIONS_PER_TASK per task"
echo "Tasks:        $TASKS"
echo "Output root:  $OUT_ROOT"
echo "Head classes: $HEAD_CLASSES"
echo "Head gran.:   $HEAD_GRANULARITY"
echo "KV heads:     $NUM_KV_HEADS"
echo "KV agg.:      $KV_AGGREGATION"
echo "KV score:     $KV_SCORE_MODE"
echo "Profile CSV:  $PROFILE_CSV"
echo "Python:       $PYTHON_BIN"
echo ""

if [ ! -f "$HEAD_CLASSES" ]; then
    echo "Building head classes from $PROFILE_CSV"
    "$PYTHON_BIN" head_analysis/build_context_head_classes.py \
        --profile_csv "$PROFILE_CSV" \
        --metric b_log_per_token_ratio \
        --quantile 0.2 \
        --head_granularity "$HEAD_GRANULARITY" \
        --num_kv_heads "$NUM_KV_HEADS" \
        --kv_aggregation "$KV_AGGREGATION" \
        --kv_score_mode "$KV_SCORE_MODE" \
        --output "$HEAD_CLASSES"
fi

run_setting() {
    local setting="$1"
    local seed="$2"
    local save_name="$setting"
    if [[ "$setting" == random_layer_matched_* ]]; then
        save_name="${setting}_seed${seed}"
    fi
    local save_dir="$OUT_ROOT/$save_name"

    echo ""
    echo "=== Running setting=$setting seed=$seed ==="
    echo "Save dir: $save_dir"

    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" head_analysis/run_context_denial_ablation.py \
        --model "$MODEL" \
        --anno_path "$ANNO_PATH" \
        --sample_fps "$FPS" \
        --kv_size "$KV_SIZE" \
        --compress_mode "$COMPRESS_MODE" \
        --device "$DEVICE" \
        --head_classes "$HEAD_CLASSES" \
        --head_granularity "$HEAD_GRANULARITY" \
        --num_kv_heads "$NUM_KV_HEADS" \
        --kv_aggregation "$KV_AGGREGATION" \
        --kv_score_mode "$KV_SCORE_MODE" \
        --setting "$setting" \
        --seed "$seed" \
        --max_mask_q_len "$MAX_MASK_Q_LEN" \
        --tasks "$TASKS" \
        --max_questions_per_task "$MAX_QUESTIONS_PER_TASK" \
        --save_dir "$save_dir"
}

run_setting full 0

if [ "$HEAD_GRANULARITY" = "kv" ]; then
    run_setting deny_previous_to_memory_kv_heads 0
    run_setting deny_current_to_current_kv_heads 0

    if [ "$CROSS" = "1" ]; then
        run_setting deny_current_to_memory_kv_heads 0
        run_setting deny_previous_to_current_kv_heads 0
    fi

    IFS=',' read -r -a SEEDS <<< "$RANDOM_SEEDS"
    for seed in "${SEEDS[@]}"; do
        run_setting random_layer_matched_previous_kv_denial "$seed"
        run_setting random_layer_matched_current_kv_denial "$seed"
    done
else
    run_setting deny_memory_to_memory_heads 0
    run_setting deny_current_to_current_heads 0

    if [ "$CROSS" = "1" ]; then
        run_setting deny_current_to_memory_heads 0
        run_setting deny_memory_to_current_heads 0
    fi

    IFS=',' read -r -a SEEDS <<< "$RANDOM_SEEDS"
    for seed in "${SEEDS[@]}"; do
        run_setting random_layer_matched_memory_denial "$seed"
        run_setting random_layer_matched_current_denial "$seed"
    done
fi

echo ""
echo "Done. Summaries:"
find "$OUT_ROOT" -maxdepth 2 -name summary.json -print | sort
