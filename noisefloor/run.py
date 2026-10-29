"""Run a benchmark entry: alternate the buggy and fixed builds, one fresh engine per run.

Usage: python -m noisefloor.run bench/entries/<id>.toml [--runs N] [--workload NAME] [--side fixed|buggy]
Runs land in runs/<gpu>/<entry>/<workload>/<fixed|buggy>/<i>.npz.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from noisefloor import build

Root = Path(__file__).resolve().parent.parent
Capture = Root / "noisefloor" / "capture.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", type=Path)
    ap.add_argument("--runs", type=int, default=3, help="runs per build")
    ap.add_argument("--workload", help="override the entry's workload")
    ap.add_argument("--side", choices=("fixed", "buggy"), help="run only this build, e.g. extra baseline reruns")
    args = ap.parse_args()

    entry = load_entry(args.entry, args.workload)
    need = entry.get("min_compute_capability")
    if need and compute_capability() < need:
        sys.exit(f"skip {entry['id']}: needs compute capability {need}+, this GPU is {compute_capability()}")
    for i in range(args.runs):
        for side in [args.side] if args.side else ("fixed", "buggy"):
            path = runs_dir(entry) / side / f"{i}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{entry['id']} {entry['workload']}: {side} run {i}", flush=True)
            capture(entry, entry[side], path)


def load_entry(path: Path, workload: str | None = None) -> dict:
    """An entry with its workload's fields merged in."""
    entry = tomllib.loads(path.read_text())
    if entry["kind"] == "null-pair":
        # Runs and reports call the calibrated build "fixed" and the one under test "buggy".
        entry["fixed"], entry["buggy"] = entry["baseline"], entry["candidate"]
    entry["workload"] = workload or entry["workload"]
    spec = tomllib.loads((Root / "bench" / "workloads" / f"{entry['workload']}.toml").read_text())
    return spec | entry


def runs_dir(entry: dict) -> Path:
    return Root / "runs" / hardware() / entry["id"] / entry["workload"]


def hardware() -> str:
    """The GPU these runs come from, so runs from different machines never mix.

    NOISEFLOOR_HW overrides it, to report on runs copied from another machine.
    """
    if hw := os.environ.get("NOISEFLOOR_HW"):
        return hw
    name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return re.sub(r"[^a-z0-9]+", "-", name.lower().removeprefix("nvidia ")).strip("-")


def compute_capability() -> float:
    cap = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(cap)


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
    env = os.environ | BaseEnv | (WslEnv if in_wsl() else {}) | entry.get("env", {})
    subprocess.run(cmd, check=True, cwd="/", env=env)


# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc and ninja at engine warmup,
# which neither the laptop nor the pod image has. Greedy decoding never uses it.
BaseEnv = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}

# vLLM disables pinned memory under WSL, which Model Runner V2 needs for its UVA buffers.
WslEnv = {"VLLM_USE_V2_MODEL_RUNNER": "0"}


def in_wsl() -> bool:
    return "microsoft" in Path("/proc/version").read_text().lower()


if __name__ == "__main__":
    main()
