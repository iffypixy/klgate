# klgate

Catch silent numeric changes between two vLLM builds.

A kernel or cache change can shift a model's outputs without failing a single test: a
rounding step in the wrong place, a stale cache slot, a kernel that truncates instead of
rounding. vLLM's GSM8K gate fails only on an 8-point accuracy drop, and it caught 1 of the 7
bugs below. klgate scores both builds a few times and asks whether the candidate differs
from the reference by more than reruns differ from each other.

## Check a build

This checks the buggy parent of #42379 (the candidate) against the fix (the reference). The
bug's kernel runs only in eager mode, hence `--engine`.

```sh
ref=$(uv run python -m klgate.build 124fac10)    # a tag, branch or commit; prints its python
new=$(uv run python -m klgate.build e9499996)
for seed in 0 1 2; do
  PYTHONPATH=. $ref -m klgate.kl score --model Qwen/Qwen3-0.6B --engine '{"enforce_eager": true}' \
    --prompts bench/gsm8k-5shot-250.jsonl --seed $seed --out runs/ref/$seed.npz
done
for seed in 3 4 5; do
  PYTHONPATH=. $new -m klgate.kl score --model Qwen/Qwen3-0.6B --engine '{"enforce_eager": true}' \
    --prompts bench/gsm8k-5shot-250.jsonl --seed $seed --out runs/new/$seed.npz
done
uv run python -m klgate.kl compare --reference runs/ref/{0,1,2}.npz --candidate runs/new/{3,4,5}.npz
```

On an A40 this prints:

```
Probes, first reference vs first candidate, 16384 positions:
  Mean KLD : 0.001125 ± 0.000013
  Median   : 0.000810   99%: 0.006218   Max: 0.094194
  RMS Δp   : 0.983 %
  Same top : 97.998 ± 0.109 %
Traffic: 45.6 % of outputs differ

Verdict: CHANGED (probes 9.36× the rerun noise, threshold 2.0×; traffic 1.17×, threshold 3.0×)
```

Builds install from vLLM's per-commit wheels into `~/.cache/klgate/builds`, about 9 GB
each; `KLGATE_BUILDS` moves them. A run takes about a minute on an A40 with the models
below. The seed reorders requests so batches differ, and every run needs its own. Run both
builds on the same GPU type with the engine settings and environment you serve with, such
as `--engine '{"max_model_len": 2048}' --env VLLM_USE_V2_MODEL_RUNNER=1`; `compare`
refuses runs whose settings differ.

## What it measures

Each run sends two kinds of requests to one engine:

- **Probes**: 32 chunks of 1,024 WikiText-2 tokens, scored with prompt logprobs on their
  last 512 positions. Both builds see identical tokens, so a kernel that rounds
  differently stands out even when no answer changes. The chunks get no BOS token, and KL
  is taken over the top 20 tokens plus one bucket for the rest, a lower bound on the true KL.
- **Traffic**: 250 GSM8K questions behind a shared 5-shot prefix, generated greedily, 32
  sequences at a time. This runs prefix caching, chunked prefill mixed with decode and KV
  block reuse, as serving does. Probes can't reach these: prompt logprobs skip the prefix
  cache. Two outputs are compared up to and including their first differing token.

## How it decides

Each part scores the mean KL across builds over the mean KL between reruns.

- Probes: changed above 2×. Probe reruns are often bit-identical, so the noise is pooled
  from both builds.
- Traffic: changed above 3×, with the noise from the reference's reruns alone. A bug that
  corrupts the KV cache at random makes the candidate's runs disagree with each other, and
  pooling would count that as noise.

## How well it works

Seven vLLM fix PRs, each replayed at its buggy parent (the candidate) and fixed commit (the
reference), and two pairs of commits with identical engine code. Each build ran 6 times on
an A40 with its entry's settings; the scores use reference seeds 0 1 2 against candidate
seeds 3 4 5.

| PR | What broke | Probes | Traffic | klgate | GSM8K | vLLM's gate |
| --- | --- | ---: | ---: | --- | --- | --- |
| [#30915](https://github.com/vllm-project/vllm/pull/30915) | FlexAttention maps reused blocks to the wrong logical block | 0.95× | 5.43× | changed | 0.255 → 0.247 | passes |
| [#31931](https://github.com/vllm-project/vllm/pull/31931) | Fused MoE applies the router weight after casting to bf16 | 2.56× | 1.40× | changed | 0.223 → 0.225 | passes |
| [#32118](https://github.com/vllm-project/vllm/pull/32118) | A 1-token Mamba prefill starts from another request's state | 1.00× | 1.02× | same | 0.525 → 0.541 | passes |
| [#36580](https://github.com/vllm-project/vllm/pull/36580) | CUDA-graph padding tokens overwrite real KV cache entries | 0.91× | 5.63× | changed | 0.417 → 0.391 | passes |
| [#42379](https://github.com/vllm-project/vllm/pull/42379) | RMSNorm upcasts the weight before multiplying; [#46070](https://github.com/vllm-project/vllm/pull/46070) reverted the fix on main | 12.47× | 1.20× | changed | 0.415 → 0.415 | passes |
| [#45361](https://github.com/vllm-project/vllm/pull/45361) | INT8 KV cache writes truncate instead of rounding | 1,647.16× | 49.50× | changed | 0.359 → 0.032 | fails |
| [#53877](https://github.com/vllm-project/vllm/pull/53877) | DeltaNet decode rounds beta to bf16 early | 1.16× | 1.02× | same | 0.273 → 0.268 | passes |
| [#30225](https://github.com/vllm-project/vllm/pull/30225) | Nothing: an example connector the engine never loads | 1.01× | 0.79× | same | 0.255 → 0.244 | passes |
| [#43571](https://github.com/vllm-project/vllm/pull/43571) | Nothing: tests only | 1.06× | 0.91× | same | 0.420 → 0.415 | passes |

GSM8K is the mean accuracy of the same runs. The gate column applies vLLM's 8-point
tolerance with the reference's accuracy as the expected value, its best case.

Splitting each build's 6 runs 3 against 3 gives 320 checks of unchanged code (16 distinct
builds; two appear in two entries). Probes peak at 1.18× and traffic at 2.26×, so none
crosses a threshold.

```sh
uv run python -m klgate.bench                                     # this table, from data/bench
uv run python -m klgate.bench run bench/entries/vllm-36580.toml   # rescore an entry on a GPU
```

## Limits

- It finds changes, not bugs. An intended numerics change reads "changed" too.
- A change below rerun noise is invisible: #32118 and #53877.
- The thresholds were set on these bugs; an unseen bug may score lower.
- #36580 corrupts each run differently, so 3 runs a side leave a thin margin. Use more
  runs a side when it matters.
- A check covers only the settings you pass. #42379's kernel runs only in eager mode.
- It needs vLLM's per-commit wheels, which the newest commits on main get only after CI,
  and an NVIDIA GPU. It was tested on an A40 with models under 1.5B parameters.
