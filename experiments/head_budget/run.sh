#!/bin/bash
# A/B Test: 对比 Uniform HERMES vs Head-Weighted HERMES
# 用法: bash run.sh [sparsemm|pseudo] [num_videos]
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${3:-0}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
S="${1:-pseudo}"
N="${2:-10}"
echo "=== Head Budget A/B Test: $S ==="
python compare.py --scores "$S" --num_videos "$N" --device 0
