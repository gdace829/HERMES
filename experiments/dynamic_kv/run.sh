#!/bin/bash
# Per-Head Dynamic KV Cache Experiment
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${3:-0}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
S="${1:-pseudo}"
N="${2:-5}"
echo "=== Dynamic Per-Head KV: $S ==="
python run.py --scores "$S" --num_videos "$N" --device 0
