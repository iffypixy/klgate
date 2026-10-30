"""Install vLLM at an exact commit into its own venv, from vLLM's per-commit wheels.

Usage: python3 -m klgate.build <tag | branch | sha | bench/entries/<id>.toml> ...
An entry file installs both of its builds.
"""

import os
import re
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BuildsDir = Path(os.environ.get("KLGATE_BUILDS", Path.home() / ".cache" / "klgate" / "builds"))
WheelIndex = "https://wheels.vllm.ai/{sha}/vllm/"
CommitApi = "https://api.github.com/repos/vllm-project/vllm/commits/{ref}"
Python = "3.12"


def venv_path(sha: str) -> Path:
    return BuildsDir / sha


def python_for(sha: str) -> Path:
    py = venv_path(sha) / "bin" / "python"
    if not (venv_path(sha) / ".done").exists():
        sys.exit(f"build {sha[:12]} is not installed: run `python3 -m klgate.build {sha}`")
    return py


def resolve(ref: str) -> str:
    """The full commit sha of a vLLM tag, branch or commit, short or full."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    request = urllib.request.Request(CommitApi.format(ref=urllib.parse.quote(ref)), headers={"Accept": "application/vnd.github.sha"})
    try:
        sha = urllib.request.urlopen(request, timeout=30).read().decode().strip()
    except urllib.error.HTTPError as e:
        sys.exit(f"can't find {ref!r} in vLLM's GitHub repo ({e.code}): pass a tag like v0.20.0, a branch, or a commit sha")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        sys.exit(f"GitHub answered {sha[:80]!r} for {ref!r}, not a commit sha")
    return sha


def install(sha: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        sys.exit(f"need a full 40-character commit sha, got {sha!r}")
    venv = venv_path(sha)
    if (venv / ".done").exists():
        return venv
    wheel = find_wheel(sha)
    subprocess.run(["uv", "venv", "-q", "--clear", "--python", Python, str(venv)], check=True)
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
        sys.exit(
            f"vLLM has no prebuilt wheel for {sha[:12]} ({e.code}): the newest commits on main get "
            "theirs once CI finishes, and some commits never do"
        )
    names = re.findall(r'href="([^"]*x86_64\.whl)"', html)
    if not names:
        sys.exit(f"no x86_64 wheel listed at {index}")
    return urllib.request.urljoin(index, names[0])


def shas(arg: str) -> list[str]:
    if arg.endswith(".toml"):
        entry = tomllib.loads(Path(arg).read_text())
        return [entry[k] for k in ("buggy", "fixed", "baseline", "candidate") if k in entry]
    return [resolve(arg)]


if __name__ == "__main__":
    todo = list(dict.fromkeys(sha for arg in sys.argv[1:] for sha in shas(arg)))
    # Installs are network- and disk-bound; a few at a time keeps the mirror happy.
    with ThreadPoolExecutor(max_workers=3) as pool:
        for venv in pool.map(install, todo):
            print(venv, flush=True)
