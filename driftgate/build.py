"""Install vLLM at an exact commit into its own venv, from vLLM's per-commit wheels."""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

BuildsDir = Path.home() / ".cache" / "driftgate" / "builds"
WheelIndex = "https://wheels.vllm.ai/{sha}/vllm/"
Python = "3.12"


def venv_path(sha: str) -> Path:
    return BuildsDir / sha


def python_for(sha: str) -> Path:
    py = venv_path(sha) / "bin" / "python"
    if not py.exists():
        sys.exit(f"build {sha[:12]} is not installed: run `python -m driftgate.build {sha}`")
    return py


def install(sha: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        sys.exit(f"need a full 40-character commit sha, got {sha!r}")
    venv = venv_path(sha)
    if (venv / ".done").exists():
        return venv
    wheel = find_wheel(sha)
    subprocess.run(["uv", "venv", "-q", "--python", Python, str(venv)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python"), wheel],
        check=True,
    )
    (venv / ".done").write_text(wheel + "\n")
    return venv


def find_wheel(sha: str) -> str:
    index = WheelIndex.format(sha=sha)
    try:
        html = urllib.request.urlopen(index).read().decode()
    except urllib.error.HTTPError as e:
        sys.exit(f"no wheels for {sha[:12]} ({e.code}): this commit needs a source build")
    names = re.findall(r'href="([^"]*x86_64\.whl)"', html)
    if not names:
        sys.exit(f"no x86_64 wheel listed at {index}")
    return urllib.request.urljoin(index, names[0])


if __name__ == "__main__":
    for sha in sys.argv[1:]:
        print(install(sha))
