#!/bin/sh
# One 40-prompt served run per build of every entry: catches broken entries cheaply.
export UV_CACHE_DIR=/workspace/.uv-cache HF_HOME=/workspace/hf DRIFTGATE_BUILDS=/workspace/builds HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/driftgate
mkdir -p logs
for entry in bench/entries/*.toml; do
    id=$(basename "$entry" .toml)
    uv run python -m driftgate.run "$entry" --runs 1 --workload smoke > "logs/$id-smoke.log" 2>&1
    echo "$id smoke exit $?"
done
echo smoke done
