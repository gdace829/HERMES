#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-results/observations/rolekv_subset_gpu2_core6_q5}"
HEAD_CLASSES="${HEAD_CLASSES:-results/observations/head_classes_prev_current/head_classes.json}"
TASKS="${TASKS:-Object Recognition,Attribute Recognition,Clips Summarize,Prospective Reasoning,Counting,Causal Reasoning}"
MAX_QUESTIONS_PER_TASK="${MAX_QUESTIONS_PER_TASK:-5}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
MODES="${MODES:-baseline rolekv random inverted}"
LAMBDA_MEMORY="${LAMBDA_MEMORY:-0.2}"
LAMBDA_CURRENT="${LAMBDA_CURRENT:-0.2}"
SEED="${SEED:-0}"

mkdir -p "${SAVE_ROOT}"

for MODE in ${MODES}; do
  SAVE_DIR="${SAVE_ROOT}/${MODE}"
  mkdir -p "${SAVE_DIR}"
  echo "[run_rolekv_subset_gpu2] mode=${MODE} save_dir=${SAVE_DIR}"

  EXTRA_ARGS=()
  if [[ -n "${MAX_VIDEOS}" ]]; then
    EXTRA_ARGS+=(--max_videos "${MAX_VIDEOS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" head_analysis/run_rolekv_subset.py \
    --model qwen2.5_vl_7b \
    --anno_path data/streamingbench/streamingbench_realtime.json \
    --sample_fps 0.5 \
    --kv_size 6000 \
    --compress_mode hermes \
    --device cuda:0 \
    --head_classes "${HEAD_CLASSES}" \
    --rolekv_mode "${MODE}" \
    --lambda_memory "${LAMBDA_MEMORY}" \
    --lambda_current "${LAMBDA_CURRENT}" \
    --seed "${SEED}" \
    --tasks "${TASKS}" \
    --max_questions_per_task "${MAX_QUESTIONS_PER_TASK}" \
    --save_dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"
done

"${PYTHON_BIN}" head_analysis/summarize_rolekv_subset.py \
  --root "${SAVE_ROOT}" \
  --modes "$(echo "${MODES}" | tr ' ' ',')" \
  --output_prefix comparison
