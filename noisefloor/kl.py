"""Compare two vLLM builds on fixed probes and on realistic traffic, in one engine launch.

`score` runs inside a build's venv and sends two kinds of requests:

- Probes: fixed text chunks scored with prompt logprobs (prefill only), as llama.cpp's
  --kl-divergence does. Every position is compared on identical context, so even a
  small numeric change in a kernel stands out. Prompt logprobs bypass the prefix
  cache, though, and these requests never decode.
- Traffic: few-shot prompts that share a prefix, generated greedily with logprobs, more
  of them than fit in one batch and finishing at different times. This exercises what
  serving does: prefix caching, chunked prefill mixed with decode, CUDA-graph padding,
  KV block reuse. Outputs are compared up to their first differing token.

`compare` needs only numpy. It reports llama.cpp's statistics on the probes for the
first reference against the first candidate, and a verdict per part:

- probes: mean KL across builds over the mean KL between reruns of either build. Probe
  reruns are often bit-identical, so the noise is pooled from both sides.
- traffic: mean KL across builds over the mean KL between reference reruns. This also
  flags a candidate whose reruns disagree with each other, as random KV corruption
  makes them do.

Usage:
  python -m noisefloor.kl score --model NAME --text wiki.test.raw --prompts gsm8k.jsonl --out run.npz [--seed N]
  python -m noisefloor.kl compare --reference r0.npz r1.npz ... --candidate c0.npz ...
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from noisefloor.compare import Run, bucketed_kl, compare as compare_runs

TopK = 20
Context = 1024  # tokens per probe chunk; the second half is scored, as in llama.cpp
Chunks = 32
TrafficTokens = 256
# Set on the README's 7 bugs and same-build reruns: no-change splits peaked at 1.18x
# (probes) and 2.26x (traffic), and the weakest bugs caught scored 2.8x and 3.3x.
Thresholds = {"probes": 2.0, "traffic": 3.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score one build")
    s.add_argument("--model", required=True)
    s.add_argument("--text", required=True, help="plain text for the probes, e.g. wiki.test.raw")
    s.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field per line, for the traffic")
    s.add_argument("--stop", default='["Question", "Assistant:", "<|separator|>"]', help="json list of stop strings")
    s.add_argument("--out", required=True, help=".npz path")
    s.add_argument("--seed", type=int, default=0, help="shuffles request order, so batches differ per run")
    s.add_argument("--engine", default="{}", help="json of engine arguments")
    c = sub.add_parser("compare", help="compare scored runs")
    c.add_argument("--reference", nargs="+", required=True, help="at least 2 runs of the trusted build")
    c.add_argument("--candidate", nargs="+", required=True, help="at least 2 runs of the build to check")
    c.add_argument("--json", help="also write the result to this path")
    args = ap.parse_args()
    if args.command == "score":
        score(args)
    else:
        result = compare([load(p) for p in args.reference], [load(p) for p in args.candidate])
        print(render(result))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2))


def score(args: argparse.Namespace) -> None:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine = json.loads(args.engine)
    llm = LLM(model=args.model, seed=0, max_logprobs=TopK, **engine)
    rng = np.random.default_rng(args.seed)

    prompts = [json.loads(line)["prompt"] for line in open(args.prompts)]
    order = rng.permutation(len(prompts))
    params = SamplingParams(temperature=0, max_tokens=TrafficTokens, stop=json.loads(args.stop), logprobs=TopK)
    run = {
        "gen_tokens": np.full((len(prompts), TrafficTokens), -1, np.int32),
        "gen_top_ids": np.full((len(prompts), TrafficTokens, TopK), -1, np.int32),
        "gen_top_lp": np.full((len(prompts), TrafficTokens, TopK), -np.inf, np.float32),
        "gen_lengths": np.zeros(len(prompts), np.int32),
    }
    for i, out in zip(order, llm.generate([prompts[i] for i in order], params)):
        gen = out.outputs[0]
        run["gen_lengths"][i] = len(gen.token_ids)
        run["gen_tokens"][i, : len(gen.token_ids)] = gen.token_ids
        for t, lps in enumerate(gen.logprobs):
            put(run["gen_top_ids"][i, t], run["gen_top_lp"][i, t], lps)

    ids = llm.get_tokenizer().encode(Path(args.text).read_text(), add_special_tokens=False)
    if len(ids) < Chunks * Context:
        raise SystemExit(f"{args.text} has {len(ids)} tokens; {Chunks * Context} are needed")
    chunks = np.array(ids[: Chunks * Context], np.int32).reshape(Chunks, Context)
    order = rng.permutation(Chunks)
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=chunks[i].tolist()) for i in order],
        SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=TopK),
    )
    first = Context // 2
    run |= {
        "tokens": chunks[:, first:],
        "top_ids": np.full((Chunks, Context - first, TopK), -1, np.int32),
        "top_lp": np.full((Chunks, Context - first, TopK), -np.inf, np.float32),
        "token_lp": np.zeros((Chunks, Context - first), np.float32),
    }
    for i, out in zip(order, outs):
        if len(out.prompt_logprobs) != Context:
            raise SystemExit(f"this vLLM build returned {len(out.prompt_logprobs)} prompt logprobs "
                             f"for a {Context}-token prompt; it can't be scored")
        for t in range(first, Context):
            lps = out.prompt_logprobs[t]
            run["token_lp"][i, t - first] = lps[int(chunks[i, t])].logprob
            put(run["top_ids"][i, t - first], run["top_lp"][i, t - first], lps)

    meta = {"vllm": vllm.__version__, "model": args.model, "engine": engine, "seed": args.seed}
    np.savez_compressed(args.out, **run, meta=json.dumps(meta))
    print(json.dumps(meta))


def put(ids: np.ndarray, lps: np.ndarray, logprobs: dict) -> None:
    """Write one position's top-k ids and logprobs, best first."""
    ranked = sorted(logprobs.items(), key=lambda kv: -kv[1].logprob)[:TopK]
    for k, (tid, lp) in enumerate(ranked):
        ids[k], lps[k] = tid, lp.logprob


def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "meta"}


def compare(refs: list[dict], cands: list[dict]) -> dict:
    if len(refs) < 2 or len(cands) < 2:
        raise SystemExit("at least 2 runs of each build are needed to measure rerun noise")
    for run in refs + cands:
        if not np.array_equal(run["tokens"], refs[0]["tokens"]):
            raise SystemExit("runs scored different probe tokens; use the same model, text and noisefloor version")
    scores = {"probes": pooled(refs, cands, probe_kl), "traffic": against_reference(refs, cands, traffic_kl)}
    return {"stats": stats(refs[0], cands[0]), "scores": scores, "thresholds": Thresholds,
            "changed": any(scores[k] > Thresholds[k] for k in scores)}


def pooled(refs: list[dict], cands: list[dict], kl) -> float:
    """Mean KL across builds over mean KL between reruns of each build."""
    within = [kl(a, b) for side in (refs, cands) for a, b in itertools.permutations(side, 2)]
    return ratio(np.mean([kl(r, c) for r in refs for c in cands]), np.mean(within))


def against_reference(refs: list[dict], cands: list[dict], kl) -> float:
    """Mean KL across builds over mean KL between reference reruns."""
    within = [kl(a, b) for a, b in itertools.permutations(refs, 2)]
    return ratio(np.mean([kl(r, c) for r in refs for c in cands]), np.mean(within))


def ratio(across: float, noise: float) -> float:
    # Bit-identical reruns leave no noise to scale by: any difference at all is beyond it.
    if noise == 0:
        return 0.0 if across == 0 else float("inf")
    return float(across / noise)


def probe_kl(a: dict, b: dict) -> float:
    return float(position_kl(a, b).mean())


def traffic_kl(a: dict, b: dict) -> float:
    """Mean KL over the generated tokens both runs share, up to their first difference."""
    return float(np.mean([d.kl for d in compare_runs(*traffic_runs(a, b))]))


def position_kl(a: dict, b: dict) -> np.ndarray:
    """KL(a || b) at every probe position, flattened."""
    k = a["top_ids"].shape[-1]
    return bucketed_kl(a["top_ids"].reshape(-1, k), a["top_lp"].reshape(-1, k),
                       b["top_ids"].reshape(-1, k), b["top_lp"].reshape(-1, k))


def traffic_runs(a: dict, b: dict) -> tuple[Run, Run]:
    return (Run(a["gen_tokens"], a["gen_top_ids"], a["gen_top_lp"], a["gen_lengths"]),
            Run(b["gen_tokens"], b["gen_top_ids"], b["gen_top_lp"], b["gen_lengths"]))


def stats(ref: dict, cand: dict) -> dict:
    d = position_kl(ref, cand)
    dp = np.exp(cand["token_lp"].ravel().astype(np.float64)) - np.exp(ref["token_lp"].ravel().astype(np.float64))
    same = ref["top_ids"][..., 0] == cand["top_ids"][..., 0]
    n = d.size
    return {
        "positions": n,
        "kl_mean": float(d.mean()), "kl_se": float(d.std(ddof=1) / np.sqrt(n)),
        "kl_median": float(np.median(d)), "kl_p99": float(np.quantile(d, 0.99)), "kl_max": float(d.max()),
        "rms_dp": float(np.sqrt((dp**2).mean())),
        "same_top": float(same.mean()), "same_top_se": float(np.sqrt(same.mean() * (1 - same.mean()) / n)),
        "traffic_diverged": float(np.mean([d.diverged for d in compare_runs(*traffic_runs(ref, cand))])),
    }


def render(r: dict) -> str:
    s, sc, th = r["stats"], r["scores"], r["thresholds"]
    verdict = "CHANGED" if r["changed"] else "same within noise"
    return "\n".join([
        f"Probes, first reference vs first candidate, {s['positions']} positions:",
        f"  Mean KLD : {s['kl_mean']:.6f} ± {s['kl_se']:.6f}",
        f"  Median   : {s['kl_median']:.6f}   99%: {s['kl_p99']:.6f}   Max: {s['kl_max']:.6f}",
        f"  RMS Δp   : {100 * s['rms_dp']:.3f} %",
        f"  Same top : {100 * s['same_top']:.3f} ± {100 * s['same_top_se']:.3f} %",
        f"Traffic: {100 * s['traffic_diverged']:.1f} % of answers differ",
        "",
        f"Verdict: {verdict} (probes {sc['probes']:.2f}× the rerun noise, threshold {th['probes']}×; "
        f"traffic {sc['traffic']:.2f}×, threshold {th['traffic']}×)",
    ])


if __name__ == "__main__":
    main()
