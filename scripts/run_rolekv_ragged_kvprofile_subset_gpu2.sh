#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-results/observations/rolekv_ragged_kvprofile_subset_gpu2_core6_q5}"
KV_PROFILE="${KV_PROFILE:-results/observations/obs_prev_current_chunk_attention_eager_gpu1_n4_full_kv/kv_head_profile_scores.csv}"
TASKS="${TASKS:-Object Recognition,Attribute Recognition,Clips Summarize,Prospective Reasoning,Counting,Causal Reasoning}"
MAX_QUESTIONS_PER_TASK="${MAX_QUESTIONS_PER_TASK:-5}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
MODES="${MODES:-baseline rolekv random inverted}"
QUOTA_RATIO="${QUOTA_RATIO:-0.7}"
LAMBDA_MEMORY="${LAMBDA_MEMORY:-0.2}"
LAMBDA_CURRENT="${LAMBDA_CURRENT:-0.2}"
KV_PROFILE_METRIC="${KV_PROFILE_METRIC:-b_log_per_token_ratio}"
KV_PROFILE_QUANTILE="${KV_PROFILE_QUANTILE:-0.2}"
SEED="${SEED:-0}"

mkdir -p "${SAVE_ROOT}"

for MODE in ${MODES}; do
  SAVE_DIR="${SAVE_ROOT}/${MODE}"
  mkdir -p "${SAVE_DIR}"
  echo "[run_rolekv_ragged_kvprofile_subset_gpu2] mode=${MODE} save_dir=${SAVE_DIR}"

  EXTRA_ARGS=()
  if [[ -n "${MAX_VIDEOS}" ]]; then
    EXTRA_ARGS+=(--max_videos "${MAX_VIDEOS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" head_analysis/run_rolekv_ragged_subset.py \
    --model qwen2.5_vl_7b \
    --anno_path data/streamingbench/streamingbench_realtime.json \
    --sample_fps 0.5 \
    --kv_size 6000 \
    --compress_mode hermes \
    --device cuda:0 \
    --kv_profile_scores "${KV_PROFILE}" \
    --kv_profile_metric "${KV_PROFILE_METRIC}" \
    --kv_profile_quantile "${KV_PROFILE_QUANTILE}" \
    --mode "${MODE}" \
    --quota_ratio "${QUOTA_RATIO}" \
    --lambda_memory "${LAMBDA_MEMORY}" \
    --lambda_current "${LAMBDA_CURRENT}" \
    --seed "${SEED}" \
    --tasks "${TASKS}" \
    --max_questions_per_task "${MAX_QUESTIONS_PER_TASK}" \
    --kv_head_budget_scores sparsemm_qwen25 \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.1 \
    --kv_head_budget_sparsemm_window_size 32 \
    --save_dir "${SAVE_DIR}" \
    "${EXTRA_ARGS[@]}"
done

"${PYTHON_BIN}" head_analysis/summarize_rolekv_subset.py \
  --root "${SAVE_ROOT}" \
  --modes "$(echo "${MODES}" | tr ' ' ',')" \
  --output_prefix comparison
