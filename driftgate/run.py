"""Run a benchmark entry: alternate the buggy and fixed builds, one fresh engine per run.

Usage: python -m driftgate.run bench/entries/<id>.toml [--runs N]
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
    args = ap.parse_args()

    entry = load_entry(args.entry)
    out = Root / "runs" / entry["id"]
    for i in range(args.runs):
        for side in ("fixed", "buggy"):
            path = out / side / f"{i}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{entry['id']}: {side} run {i}", flush=True)
            capture(entry, entry[side], path)


def load_entry(path: Path) -> dict:
    """An entry with its workload's fields merged in."""
    entry = tomllib.loads(path.read_text())
    workload = tomllib.loads((Root / "bench" / "workloads" / f"{entry['workload']}.toml").read_text())
    return workload | entry


def capture(entry: dict, sha: str, path: Path) -> None:
    cmd = [
        str(build.python_for(sha)), str(Capture),
        "--model", entry["model"],
        "--prompts", str(Root / entry["prompts"]),
        "--max-tokens", str(entry["max_tokens"]),
        "--stop", json.dumps(entry["stop"]),
        "--engine", json.dumps(entry.get("engine", {})),
        "--out", str(path),
    ]
    # A fresh process per run, so run-to-run noise includes launch effects.
    subprocess.run(cmd, check=True, cwd="/", env=os.environ | (WslEnv if in_wsl() else {}))


# vLLM disables pinned memory under WSL, which Model Runner V2 needs for its UVA
# buffers, and FlashInfer's sampler JIT-compiles with nvcc, which the laptop lacks.
WslEnv = {"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}


def in_wsl() -> bool:
    return "microsoft" in Path("/proc/version").read_text().lower()


if __name__ == "__main__":
    main()
