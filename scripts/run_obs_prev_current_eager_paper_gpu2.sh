#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjs/.conda/envs/hermes-qwen/bin/python}"
SAVE_DIR="${SAVE_DIR:-results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper}"
KV_DIR="${KV_DIR:-${SAVE_DIR}_kv}"
NUM_VIDEOS="${NUM_VIDEOS:-16}"
MAX_QUESTIONS="${MAX_QUESTIONS:-4}"
MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-500}"
ENCODE_CHUNK_SIZE="${ENCODE_CHUNK_SIZE:-16}"
KV_SIZE="${KV_SIZE:-6000}"
SAMPLE_FPS="${SAMPLE_FPS:-0.5}"
ANNO_PATH="${ANNO_PATH:-data/streamingbench/streamingbench_realtime.json}"

mkdir -p "${SAVE_DIR}" "${KV_DIR}"

echo "[paper obs] GPU=${GPU}"
echo "[paper obs] SAVE_DIR=${SAVE_DIR}"
echo "[paper obs] KV_DIR=${KV_DIR}"
echo "[paper obs] NUM_VIDEOS=${NUM_VIDEOS} MAX_QUESTIONS=${MAX_QUESTIONS} MAX_OBSERVATIONS=${MAX_OBSERVATIONS}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" head_analysis/obs_prev_current_chunk_attention_eager.py \
  --model qwen2.5_vl_7b \
  --kv_size "${KV_SIZE}" \
  --compress_mode hermes \
  --sample_fps "${SAMPLE_FPS}" \
  --anno_path "${ANNO_PATH}" \
  --device 0 \
  --num_videos "${NUM_VIDEOS}" \
  --max_questions "${MAX_QUESTIONS}" \
  --max_observations "${MAX_OBSERVATIONS}" \
  --encode_chunk_size "${ENCODE_CHUNK_SIZE}" \
  --save_dir "${SAVE_DIR}"

"${PYTHON_BIN}" head_analysis/generate_prev_current_profile_artifacts.py \
  --raw_csv "${SAVE_DIR}/raw_prev_current_attention.csv" \
  --out_dir "${SAVE_DIR}"

"${PYTHON_BIN}" head_analysis/aggregate_prev_current_to_kv_heads.py \
  --raw_csv "${SAVE_DIR}/raw_prev_current_attention.csv" \
  --out_dir "${KV_DIR}" \
  --num_query_heads 28 \
  --num_kv_heads 4

"${PYTHON_BIN}" head_analysis/build_kv_head_classes.py \
  --raw_kv_csv "${KV_DIR}/raw_prev_current_attention_kv.csv" \
  --profile_csv "${KV_DIR}/kv_head_profile_scores.csv" \
  --pooling pooled_median \
  --quantile 0.2 \
  --output "${KV_DIR}/head_classes_kv_pooled_median.json"

echo "[paper obs] done"
echo "[paper obs] query artifacts: ${SAVE_DIR}"
echo "[paper obs] kv artifacts: ${KV_DIR}"
