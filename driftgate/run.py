"""Run a benchmark entry: alternate the buggy and fixed builds, one fresh engine per run.

Usage: python -m driftgate.run bench/entries/<id>.toml [--runs N] [--workload NAME]
Runs land in runs/<entry>/<workload>/<fixed|buggy>/<i>.npz.
"""

import argparse
import json
import os
import subprocess
import tomllib
from pathlib import Path

from driftgate import build

Root = Path(__file__).resolve().parent.parent
Capture = Root / "driftgate" / "capture.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", type=Path)
    ap.add_argument("--runs", type=int, default=3, help="runs per build")
    ap.add_argument("--workload", help="override the entry's workload")
    args = ap.parse_args()

    entry = load_entry(args.entry, args.workload)
    for i in range(args.runs):
        for side in ("fixed", "buggy"):
            path = runs_dir(entry) / side / f"{i}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{entry['id']} {entry['workload']}: {side} run {i}", flush=True)
            capture(entry, entry[side], path)


def load_entry(path: Path, workload: str | None = None) -> dict:
    """An entry with its workload's fields merged in."""
    entry = tomllib.loads(path.read_text())
    entry["workload"] = workload or entry["workload"]
    spec = tomllib.loads((Root / "bench" / "workloads" / f"{entry['workload']}.toml").read_text())
    return spec | entry


def runs_dir(entry: dict) -> Path:
    return Root / "runs" / entry["id"] / entry["workload"]


def capture(entry: dict, sha: str, path: Path) -> None:
    cmd = [
        str(build.python_for(sha)), str(Capture),
        "--mode", entry["mode"],
        "--model", entry["model"],
        "--prompts", str(Root / entry["prompts"]),
        "--max-tokens", str(entry["max_tokens"]),
        "--stop", json.dumps(entry["stop"]),
        "--engine", json.dumps(entry.get("engine", {})),
        "--concurrency", str(entry.get("concurrency", 1)),
        "--out", str(path),
    ]
    # A fresh process per run, so run-to-run noise includes launch effects.
    env = os.environ | (WslEnv if in_wsl() else {}) | entry.get("env", {})
    subprocess.run(cmd, check=True, cwd="/", env=env)


# vLLM disables pinned memory under WSL, which Model Runner V2 needs for its UVA
# buffers, and FlashInfer's sampler JIT-compiles with nvcc, which the laptop lacks.
WslEnv = {"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def in_wsl() -> bool:
    return "microsoft" in Path("/proc/version").read_text().lower()


if __name__ == "__main__":
    main()
