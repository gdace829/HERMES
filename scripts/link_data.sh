#!/bin/bash
# Symlink existing benchmark videos into the paths HERMES expects.
# Run: bash scripts/link_data.sh

set -e

echo "=== Linking StreamingBench ==="
mkdir -p /data/streamingbench/videos

sb_linked=0
sb_missing=0
for sample_dir in /data/StreamingBench/data/real/sample_*/; do
    sample_id=$(basename "$sample_dir")
    src="${sample_dir}video.mp4"
    dst="/data/streamingbench/videos/${sample_id}_real.mp4"
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        sb_linked=$((sb_linked + 1))
    else
        sb_missing=$((sb_missing + 1))
    fi
done
echo "StreamingBench: $sb_linked linked, $sb_missing missing"

echo ""
echo "=== Linking OVOBench ==="
mkdir -p /data/ovobench/videos

# AutoEvalMetaData (154 files)
if [ -d /data/OVOBench/src_videos/AutoEvalMetaData ]; then
    ln -sfn /data/OVOBench/src_videos/AutoEvalMetaData /data/ovobench/videos/AutoEvalMetaData
    echo "OVOBench/AutoEvalMetaData: symlinked"
else
    echo "OVOBench/AutoEvalMetaData: NOT FOUND"
fi

# Ego4D - actual data is at OVOBench/src_videos/Ego4D/{clips,video}
if [ -d /data/OVOBench/OVOBench/src_videos/Ego4D ]; then
    mkdir -p /data/ovobench/videos/Ego4D
    if [ -d /data/OVOBench/OVOBench/src_videos/Ego4D/clips ]; then
        ln -sfn /data/OVOBench/OVOBench/src_videos/Ego4D/clips /data/ovobench/videos/Ego4D/clips
        echo "OVOBench/Ego4D/clips: symlinked"
    fi
    if [ -d /data/OVOBench/OVOBench/src_videos/Ego4D/video ]; then
        ln -sfn /data/OVOBench/OVOBench/src_videos/Ego4D/video /data/ovobench/videos/Ego4D/video
        echo "OVOBench/Ego4D/video: symlinked"
    fi
else
    echo "OVOBench/Ego4D: NOT FOUND"
fi

# YouTube_Games
if [ -d /data/OVOBench/OVOBench/src_videos/YouTube_Games ]; then
    ln -sfn /data/OVOBench/OVOBench/src_videos/YouTube_Games /data/ovobench/videos/YouTube_Games
    echo "OVOBench/YouTube_Games: symlinked"
else
    echo "OVOBench/YouTube_Games: NOT FOUND"
fi

# Other OVOBench categories not available in current data
for cat in COIN OpenEQA cross_task hirest star youcook2; do
    echo "OVOBench/$cat: NOT FOUND (no source data)"
done

echo ""
echo "=== Coverage Summary ==="
echo "StreamingBench: ready for inference"
echo "OVOBench: partial — AutoEvalMetaData available; Ego4D, YouTube_Games partial; COIN/OpenEQA/cross_task/hirest/star/youcook2 missing"
