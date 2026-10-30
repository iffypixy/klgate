#!/bin/sh
# Prepare a rented GPU pod: uv, the harness env, and the builds and models of the given
# entries (all of them by default).
# Usage: pod/setup.sh [entry id ...]
# Everything heavy lives on /workspace (the pod's volume), so builds survive a restart and
# uv can hardlink build venvs from one cache.
set -eu

export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/hf
export KLGATE_BUILDS=/workspace/builds
# Build venvs point at uv's Python, which would otherwise sit on the container disk that a
# pod stop wipes.
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/klgate
if [ $# -gt 0 ]; then
    entries=$(for id in "$@"; do echo "bench/entries/$id.toml"; done)
else
    entries=$(ls bench/entries/*.toml)
fi
uv sync -q
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python3 -m klgate.build $entries
for model in $(grep -h '^model = ' $entries | cut -d'"' -f2 | sort -u); do
    uvx -q --from huggingface_hub hf download "$model" >/dev/null && echo "model $model"
done
echo setup done
