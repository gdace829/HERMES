#!/bin/bash
# Per-Head Dynamic KV — StreamingBench 全量 (GPU 3, 后台)
set -euo pipefail
GPU="${1:-3}"
cd /home/sjs/HERMES
export PYTHONPATH=/home/sjs/HERMES
export CUDA_VISIBLE_DEVICES="$GPU"
source /opt/conda/etc/profile.d/conda.sh
conda activate hermes-qwen
nohup python experiments/dynamic_kv/run.py \
    --scores /home/sjs/SparseMM/visual_head/head_score/qwen2.5-vl.json \
    --num_videos 498 --device 0 \
    > logs/per_head_sb_gpu${GPU}.log 2>&1 &
echo "PID: $! | GPU: $GPU | log: logs/per_head_sb_gpu${GPU}.log"
