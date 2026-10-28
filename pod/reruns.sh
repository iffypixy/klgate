#!/bin/sh
# Extra fixed-build reruns, to measure each detector's false alarms on the baseline.
# Usage: nohup pod/reruns.sh [runs] entry... > reruns.log 2>&1 &
export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/hf
export DRIFTGATE_BUILDS=/workspace/builds
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/driftgate
mkdir -p logs

runs=$1
shift
for id in "$@"; do
    uv run python -m driftgate.run "bench/entries/$id.toml" --runs "$runs" --workload gsm8k-5shot-serve --side fixed > "logs/$id-reruns.log" 2>&1
    echo "$id reruns exit $?"
done
echo reruns done
