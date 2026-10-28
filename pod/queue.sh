#!/bin/sh
# Run every entry on every workload, one after another, on the pod.
# Usage: nohup pod/queue.sh [runs] > queue.log 2>&1 &
export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/hf
export DRIFTGATE_BUILDS=/workspace/builds
# setup.sh downloads every model; offline keeps runs from touching the hub mid-run.
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/driftgate
mkdir -p logs

runs=${1:-3}
for workload in gsm8k-5shot-serve gsm8k-5shot; do
    for entry in bench/entries/*.toml; do
        id=$(basename "$entry" .toml)
        uv run python -m driftgate.run "$entry" --runs "$runs" --workload "$workload" > "logs/$id-$workload.log" 2>&1
        echo "$id $workload exit $?"
    done
done
echo queue done
