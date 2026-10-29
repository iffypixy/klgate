# driftgate

Catch silent numeric regressions in vLLM before they ship.

A change to vLLM can shift a model's outputs without failing a single test: a rounding step
in the wrong place, a stale cache slot, a kernel that truncates instead of rounding. vLLM's
GSM8K gate only fails on an 8-point accuracy drop, so it misses most of these. driftgate runs
the old and new builds a few times each and asks one question: do the new build's outputs
differ from the old one's more than the old one differs from itself?

## Check a build

```sh
uv sync
uv run python -m driftgate.diff <baseline> <candidate> --model Qwen/Qwen3-0.6B
```

Each build is a vLLM tag, branch or commit: `v0.20.0`, `main`, `124fac10`. It installs both
from vLLM's prebuilt wheels (about 9 GB each), runs 5 baseline and 3 candidate engines on 250
GSM8K questions served 32 at a time, and prints a verdict. Here it is with the fix for #36580
as the baseline and the commit before it as the candidate, with `--env
VLLM_USE_V2_MODEL_RUNNER=1` since that bug lives in Model Runner V2:

```
baseline 9efc3bdcd674 → candidate 156e33553ccd · Qwen/Qwen3-0.6B · a40 · 250 prompts

Verdict: CHANGED (middle of 6 candidate runs 61.83× the noise; threshold 1.25×)
Candidate runs: 66.91×, 56.76×, 78.52×, 66.99×, 48.49×, 50.62× the baseline-vs-baseline KL of 4.60e-04
The first 3 runs read over the threshold, so 3 more ran and all 6 decide.
GSM8K accuracy: baseline 0.418, candidate 0.387 (context, not part of the verdict)

Most-changed prompts whose text differs (baseline run 1 vs candidate run 1, from the first differing token):
  #223, token 17:
    baseline   " that geraniums. So he plants 90 - 40 = <<90-40=50>>50 petunias.\nSo total flower"
    candidate  " than geraniums. So he plants 90 - 40 = <<90-40=50>>50 petunias.\nSo total flower"
  #141, token 8:
    baseline   " - 4 = <<20-12-4=4>>4 cars are semi-automatic.\nSo the percentage is 4/20 = <<4/2"
    candidate  " + 4 = <<12+4=16>>16 semi-automatic cars.\nSo the percentage is 16/20 = <<16/20=8"
  ...
```

A check takes about 25 minutes on an A40, 8 more when it confirms, so roughly $0.25–0.35 on
a rented one, plus the first install of each build. Pass the engine settings you deploy with
(`--engine '{"max_model_len": 2048}'`) and any environment (`--env VLLM_USE_V2_MODEL_RUNNER=1`):
a check only covers the code paths those settings run.

It works best on one commit or a short range. A release brings hundreds of commits, and some
intended numerics change among them is likely; then the shown prompts and accuracy are what
tell you whether it matters.

### How it decides

Even the same build rarely reproduces its own outputs exactly when serving, because batch
composition changes with request timing. So every candidate run gets a score: its KL to the
baseline runs over the baselines' KL to each other. Unchanged code scores about 1×.

The verdict takes the middle of the three scores. That matters because now and then an engine
launches into a slightly different numeric state for its whole run and disagrees with every
other run by 20–50%. When the first three runs read "changed", three more run and the middle
of all six decides, so two stray launches can't flip it either.

The 1.25× threshold was set on unchanged code only: reruns of one build, and pairs of
commits that change no engine code, across 4 models on 2 GPUs. `python -m
driftgate.threshold` replays the rule on those runs (kept in `data/kl`) and prints the rates
and scores below.

## How well it works

Seven real vLLM bugs, each replayed from its fix PR's buggy parent and fixed commit:

| Fix | What broke | GPU | vLLM's gate | driftgate.diff |
| --- | --- | --- | --- | --- |
| [#45361](https://github.com/vllm-project/vllm/pull/45361) | INT8 KV cache writes truncate instead of rounding | A5000 | caught | caught, 86× |
| [#36580](https://github.com/vllm-project/vllm/pull/36580) | CUDA-graph padding tokens overwrite real KV cache entries | A40 | missed | caught, 49× |
| [#30915](https://github.com/vllm-project/vllm/pull/30915) | FlexAttention maps reused blocks to the wrong logical block | A5000 | missed | caught, 9.5× |
| [#42379](https://github.com/vllm-project/vllm/pull/42379) | RMSNorm upcasts the weight before multiplying | A5000; A40 eager | missed | caught, 1.57× and 1.45× |
| [#31931](https://github.com/vllm-project/vllm/pull/31931) | Fused MoE applies the router weight after casting to bf16 | A40 | missed | 9 checks in 10, 1.35× |
| [#32118](https://github.com/vllm-project/vllm/pull/32118) | A 1-token Mamba prefill starts from another request's state | A40 | missed | missed, 0.98× |
| [#53877](https://github.com/vllm-project/vllm/pull/53877) | DeltaNet decode rounds beta to bf16 early | A5000 | missed | missed, 1.03× |

Each score is the typical first-round score over 4,000 simulated checks; "9 checks in 10" is
the share that said "changed". #45361 and #53877 have only 3 runs a side, so theirs is one
check on all of them. On unchanged code, across 9 sets of runs, at most 1.5% of checks say
"changed"; without the confirmation round it was up to 7.4%. The gate column is its best
case: it was given the fixed build's own accuracy as its reference.

Three end-to-end checks on a rented A40, all with the compiled engine:

| Check | First round | Verdict |
| --- | --- | --- |
| Two builds of identical engine code | 0.98× | same |
| #36580, fixed → buggy | 66.9× | changed, 61.8× over 6 runs |
| #42379, fixed → buggy | 1.26×, from two off launches | same, 1.02× over 6 runs |

The last one is right: compiled vLLM swaps the RMSNorm CUDA kernel for a compiled one
(`rms_norm=['native']` in its startup log), so the bug never runs. In eager mode the same
pair reads 1.41×, changed.

## Limits

- It finds changes, not bugs. "Changed" means the outputs moved beyond noise; an intended
  numerics change reads the same way. Read the shown prompts and the diff.
- A change smaller than rerun noise is invisible: #32118 and #53877 above.
- A check covers the settings it ran. #42379's kernel only runs in eager mode, so with the
  default compiled engine it reads "same".
- Off launches set the floor. Why an engine sometimes starts in a different numeric state is
  open; one build here did it in 4 of 15 runs and its identical-code twin in 0 of 11.
- It needs vLLM's prebuilt wheels, which the newest commits on main get only after CI, and an
  NVIDIA GPU. It was tested on an A40 and an RTX A5000, with models of 0.6–1.3B parameters.
- Gated models need a Hugging Face token for the tokenizer.

## The benchmark

Each entry is a vLLM fix PR's buggy parent and fixed commit, installed from vLLM's per-commit
wheels and run on a shared, pinned workload. Same-build reruns measure noise; the
buggy-vs-fixed difference is what a detector must find.

```sh
python3 -m driftgate.build <buggy sha> <fixed sha>    # once per build
uv run python -m driftgate.run bench/entries/vllm-42379.toml --runs 3
uv run python -m driftgate.report bench/entries/vllm-42379.toml
```

| Path | What |
| --- | --- |
| `bench/entries/` | One TOML per regression: the commit pair, model, serving flags, provenance |
| `bench/workloads/` | Pinned workloads; `gsm8k-5shot` mirrors vLLM's own GSM8K gate, `diff` is what a check runs |
| `driftgate/build.py` | Installs vLLM at an exact commit into its own venv |
| `driftgate/capture.py` | Runs inside a build's venv; saves tokens and top-20 logprobs |
| `driftgate/run.py` | Alternates buggy and fixed builds, one fresh engine per run |
| `driftgate/compare.py` | Bucketed KL on shared greedy prefixes |
| `driftgate/gsm8k.py` | GSM8K accuracy as vLLM scores it, and an exact paired McNemar test |
| `driftgate/report.py` | Every detector on every run pair: false alarms and detections |
| `driftgate/diff.py` | One command: does a candidate build change a model's outputs beyond noise? |
| `driftgate/threshold.py` | Replays the diff rule on stored run pairs: its threshold, false alarms, detections |
| `data/kl/` | Per-prompt KL between every pair of runs, per GPU and entry |
| `data/mining/` | Candidate fix PRs mined from vLLM, 2025–2026 |

Under WSL the runner sets `VLLM_USE_V2_MODEL_RUNNER=0` (Model Runner V2 needs pinned memory,
which vLLM disables in WSL) and `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT-compiles with nvcc).
