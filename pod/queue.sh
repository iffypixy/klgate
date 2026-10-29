#!/bin/sh
# Run the jobs in a file, one per line: <entry id> <workload> <runs per build> [fixed|buggy].
# The file is reread after every job, so jobs can be appended while it runs. Once it runs
# out, it waits 20 minutes for more, then prints "queue done". Stop the pod after that:
# an idle GPU still bills.
# Usage: setsid nohup pod/queue.sh jobs.txt > /workspace/queue.log 2>&1 < /dev/null &
export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/hf
export NOISEFLOOR_BUILDS=/workspace/builds
export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
# setup.sh downloads the models up front; offline keeps runs off the hub mid-run.
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
jobs=$(realpath "$1")
cd /workspace/noisefloor
mkdir -p logs

n=1
idle=0
while [ "$idle" -lt 20 ]; do
    line=$(sed -n "${n}p" "$jobs")
    if [ -z "$line" ]; then
        sleep 60
        idle=$((idle + 1))
        continue
    fi
    idle=0
    set -- $line
    side=${4:+--side $4}
    echo "$(date -u +%H:%M) start $line"
    uv run python -m noisefloor.run "bench/entries/$1.toml" --workload "$2" --runs "$3" $side >> "logs/$1-$2.log" 2>&1
    status=$?
    echo "$(date -u +%H:%M) exit $status $line"
    n=$((n + 1))
done
echo "$(date -u +%H:%M) queue done"
