#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${2:-0}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
NV="${3:-20}"
echo "=== Temporal Receptive Field ==="
python "$SCRIPT_DIR/temporal_receptive_field.py" \
    --model qwen2.5_vl_7b --sample_fps 0.5 --num_videos "$NV" \
    --save_dir "results/head_analysis/receptive_field-v${NV}"
