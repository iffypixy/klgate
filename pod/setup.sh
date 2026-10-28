#!/bin/sh
# Prepare a rented GPU pod: uv, the harness env, every entry's builds, every entry's model.
# Everything heavy lives on /workspace (the pod's volume), so builds survive a restart and
# uv can hardlink build venvs from one cache.
set -eu

export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/hf
export DRIFTGATE_BUILDS=/workspace/builds
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/driftgate
uv sync -q
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python3 -m driftgate.build bench/entries/*.toml
for model in $(grep -h '^model = ' bench/entries/*.toml | cut -d'"' -f2 | sort -u); do
    uvx -q --from huggingface_hub hf download "$model" >/dev/null && echo "model $model"
done
echo setup done
