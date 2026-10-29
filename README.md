# driftgate

A benchmark of real inference-engine regressions, and a fair comparison of the checks that catch them.

Each entry is a vLLM fix PR's buggy parent and fixed commit, installed from vLLM's per-commit
wheels and run on a shared, pinned workload. Same-build reruns measure noise; the buggy-vs-fixed
difference is what a detector must find.

## Layout

| Path | What |
| --- | --- |
| `bench/entries/` | One TOML per regression: the commit pair, model, serving flags, provenance |
| `bench/workloads/` | Pinned workloads; `gsm8k-5shot` mirrors vLLM's own GSM8K gate, `-serve` runs it over HTTP under load |
| `driftgate/build.py` | Installs vLLM at an exact commit into its own venv |
| `driftgate/capture.py` | Runs inside a build's venv; saves tokens and top-20 logprobs |
| `driftgate/run.py` | Alternates buggy and fixed builds, one fresh engine per run |
| `driftgate/compare.py` | Bucketed KL on shared greedy prefixes |
| `driftgate/gsm8k.py` | GSM8K accuracy as vLLM scores it, and an exact paired McNemar test |
| `driftgate/report.py` | Every detector on every run pair: false alarms and detections |
| `driftgate/diff.py` | One command: does a candidate build change a model's outputs beyond noise? |
| `data/mining/` | Candidate fix PRs mined from vLLM, 2025–2026 |

## Run an entry

```sh
uv sync
python3 -m driftgate.build <buggy sha> <fixed sha>    # once per build, ~9 GB each
uv run python -m driftgate.run bench/entries/vllm-42379.toml --runs 3
uv run python -m driftgate.report bench/entries/vllm-42379.toml
```

## Check a build

```sh
uv run python -m driftgate.diff <baseline sha> <candidate sha> --model Qwen/Qwen3-0.6B
```

It installs both builds, runs 5 baseline and 3 candidate engines on 250 GSM8K questions
served under load, and says "same within noise" or "changed", with the prompts that changed
most. Each candidate run is scored by its KL to the baseline runs over the baselines' KL to
each other, and the verdict takes the middle of the three scores, so one off launch can't
flip it. The 1.25× threshold was set on unchanged code across 4 models and 2 GPUs: at most 5%
false alarms on each, and up to 7% on a model held out of the fit. "Changed" means the outputs moved, not
that they got worse.

Under WSL the runner sets `VLLM_USE_V2_MODEL_RUNNER=0` (Model Runner V2 needs pinned memory,
which vLLM disables in WSL) and `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT-compiles with nvcc).
