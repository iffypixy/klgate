# driftgate

A benchmark of real inference-engine regressions, and a fair comparison of the checks that catch them.

Each entry is a vLLM fix PR's buggy parent and fixed commit, installed from vLLM's per-commit
wheels and run on a shared, pinned workload. Same-build reruns measure noise; the buggy-vs-fixed
difference is what a detector must find.

## Layout

| Path | What |
| --- | --- |
| `bench/entries/` | One TOML per regression: the commit pair, model, serving flags, provenance |
| `bench/workloads/` | Pinned workloads; `gsm8k-5shot` mirrors vLLM's own GSM8K gate |
| `driftgate/build.py` | Installs vLLM at an exact commit into its own venv |
| `driftgate/capture.py` | Runs inside a build's venv; saves tokens and top-20 logprobs |
| `driftgate/run.py` | Alternates buggy and fixed builds, one fresh engine per run |
| `driftgate/compare.py` | Bucketed KL on shared greedy prefixes, every run pair |
| `driftgate/gsm8k.py` | GSM8K accuracy as vLLM scores it, and an exact paired McNemar test |
| `data/mining/` | Candidate fix PRs mined from vLLM, 2025–2026 |

## Run an entry

```sh
uv sync
python3 -m driftgate.build <buggy sha> <fixed sha>    # once per build, ~9 GB each
uv run python -m driftgate.run bench/entries/vllm-42379.toml --runs 3
uv run python -m driftgate.compare runs/vllm-42379/fixed runs/vllm-42379/buggy
uv run python -m driftgate.gsm8k bench/entries/vllm-42379.toml
```

Under WSL the runner sets `VLLM_USE_V2_MODEL_RUNNER=0` (Model Runner V2 needs pinned memory,
which vLLM disables in WSL) and `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT-compiles with nvcc).
