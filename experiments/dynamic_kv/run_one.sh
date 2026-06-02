#!/bin/bash
set -euo pipefail
cd /home/sjs/HERMES
export PYTHONPATH=/home/sjs/HERMES
export CUDA_VISIBLE_DEVICES="${1:-3}"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
python experiments/dynamic_kv/run.py \
    --scores /home/sjs/SparseMM/visual_head/head_score/qwen2.5-vl.json \
    --num_videos 1 --device 0
