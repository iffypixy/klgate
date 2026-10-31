"""Check whether a candidate vLLM build changes a model's outputs beyond rerun noise.

`score` runs inside a build's venv and sends two kinds of requests in one engine launch:

- Probes: fixed text chunks scored with prompt logprobs (prefill only), in the style of
  llama.cpp's --kl-divergence but with no BOS token. Every position is compared on
  identical context, so even a small numeric change in a kernel stands out. Prompt
  logprobs bypass the prefix cache, though, and these requests never decode.
- Traffic: few-shot prompts that share a prefix, generated greedily with logprobs, more
  of them than fit in one batch and finishing at different times. This exercises what
  serving does: prefix caching, chunked prefill mixed with decode, CUDA-graph padding,
  KV block reuse. Outputs are compared up to their first differing token.

`compare` needs only numpy. It reports llama.cpp-style statistics on the probes for the
first reference against the first candidate, and a verdict per part:

- probes: mean KL across builds over the mean KL between reruns of either build. Probe
  reruns are often bit-identical, so the noise is pooled from both sides.
- traffic: mean KL across builds over the mean KL between reference reruns. This also
  flags a candidate whose reruns disagree with each other, as random KV corruption
  makes them do.

KL is taken over the top-20 ids both runs list plus one bucket for the rest, a lower bound
on the true KL.

Usage:
  python -m klgate.kl score --model NAME --prompts gsm8k.jsonl --seed N --out run.npz [--text wiki.test.raw]
  python -m klgate.kl compare --reference r0.npz r1.npz ... --candidate c0.npz c1.npz ...
"""

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

TopK = 20
Context = 1024  # tokens per probe chunk; the second half is scored, as in llama.cpp
Chunks = 32
TrafficTokens = 256
Stop = ["Question", "Assistant:", "<|separator|>"]
Engine = {"max_num_seqs": 32}  # fewer than the traffic prompts, so they queue and batches churn
Env = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}  # FlashInfer's sampler JIT-compiles with nvcc, which many machines lack
Wiki = Path.home() / ".cache" / "klgate" / "wiki.test.raw"
WikiUrl = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WikiMember = "wikitext-2-raw/wiki.test.raw"
WikiSha = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
TailFloor = 1e-9
# Picked by looking at the README's 7 bugs and same-build reruns; an unseen bug may score lower.
Thresholds = {"probes": 2.0, "traffic": 3.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score one build")
    s.add_argument("--model", required=True)
    s.add_argument("--text", help=f"plain text for the probes; default {Wiki}, downloaded once")
    s.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field per line, for the traffic")
    s.add_argument("--seed", type=int, required=True, help="shuffles request order, so batches differ per run")
    s.add_argument("--out", required=True, help=".npz path")
    s.add_argument("--engine", default="{}", help="json of engine arguments")
    s.add_argument("--env", type=key_value, action="append", default=[], metavar="KEY=VALUE",
                   help="environment variable to set before vLLM loads; repeatable")
    c = sub.add_parser("compare", help="compare scored runs")
    c.add_argument("--reference", nargs="+", required=True, help="at least 2 runs of the trusted build")
    c.add_argument("--candidate", nargs="+", required=True, help="at least 2 runs of the build to check")
    c.add_argument("--json", help="also write the result to this path")
    args = ap.parse_args()
    if args.command == "score":
        score(args)
    else:
        result = judge([load(p) for p in args.reference], [load(p) for p in args.candidate])
        print(render(result))
        if args.json:
            Path(args.json).write_text(json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n")


def key_value(arg: str) -> tuple[str, str]:
    key, sep, value = arg.partition("=")
    if not key or not sep:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {arg!r}")
    return key, value


def score(args: argparse.Namespace) -> None:
    env = {k: os.environ.get(k, v) for k, v in Env.items()} | dict(args.env)
    os.environ.update(env)
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    text = Path(args.text) if args.text else wiki_text()
    prompts_file, text_file = Path(args.prompts).read_bytes(), text.read_bytes()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = Engine | json.loads(args.engine)
    llm = LLM(model=args.model, seed=0, max_logprobs=TopK, **engine)
    rng = np.random.default_rng(args.seed)

    prompts = [json.loads(line)["prompt"] for line in prompts_file.decode().splitlines()]
    params = SamplingParams(temperature=0, max_tokens=TrafficTokens, stop=Stop, logprobs=TopK)
    run = {
        "traffic_tokens": np.full((len(prompts), TrafficTokens), -1, np.int32),
        "traffic_ids": np.full((len(prompts), TrafficTokens, TopK), -1, np.int32),
        "traffic_lp": np.full((len(prompts), TrafficTokens, TopK), -np.inf, np.float32),
        "traffic_lengths": np.zeros(len(prompts), np.int32),
    }
    for i, output in enumerate(shuffled(lambda ps: llm.generate(ps, params), prompts, rng)):
        gen = output.outputs[0]
        logprobs = gen.logprobs or []
        if len(logprobs) != len(gen.token_ids):
            raise SystemExit(f"this vLLM build returned {len(logprobs)} logprobs for {len(gen.token_ids)} "
                             "generated tokens; it can't be scored")
        run["traffic_lengths"][i] = len(gen.token_ids)
        run["traffic_tokens"][i, : len(gen.token_ids)] = gen.token_ids
        for t, lps in enumerate(logprobs):
            put(run["traffic_ids"][i, t], run["traffic_lp"][i, t], lps)

    ids = llm.get_tokenizer().encode(text_file.decode(), add_special_tokens=False)
    if len(ids) < Chunks * Context:
        raise SystemExit(f"{text} has {len(ids)} tokens; {Chunks * Context} are needed")
    chunks = np.array(ids[: Chunks * Context], np.int32).reshape(Chunks, Context)
    first = Context // 2
    run |= {
        "probe_tokens": chunks[:, first:],
        "probe_ids": np.full((Chunks, Context - first, TopK), -1, np.int32),
        "probe_lp": np.full((Chunks, Context - first, TopK), -np.inf, np.float32),
        "probe_token_lp": np.zeros((Chunks, Context - first), np.float32),
    }
    probes = [TokensPrompt(prompt_token_ids=chunk.tolist()) for chunk in chunks]
    params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=TopK)
    for i, output in enumerate(shuffled(lambda ps: llm.generate(ps, params), probes, rng)):
        if len(output.prompt_logprobs) != Context:
            raise SystemExit(f"this vLLM build returned {len(output.prompt_logprobs)} prompt logprobs for a "
                             f"{Context}-token prompt, as Model Runner V2 does now and then at some commits; rerun")
        for t in range(first, Context):
            lps = output.prompt_logprobs[t]
            run["probe_token_lp"][i, t - first] = lps[int(chunks[i, t])].logprob
            put(run["probe_ids"][i, t - first], run["probe_lp"][i, t - first], lps)

    meta = {"vllm": vllm.__version__, "seed": args.seed, "model": args.model, "engine": engine, "env": env,
            "prompts": hashlib.sha256(prompts_file).hexdigest(), "text": hashlib.sha256(text_file).hexdigest(),
            "gpu": torch.cuda.get_device_name()}
    np.savez_compressed(out, **run, meta=json.dumps(meta))
    print(json.dumps(meta))


def wiki_text() -> Path:
    """llama.cpp's copy of the WikiText-2 test set, downloaded once and checked on every use."""
    if not Wiki.exists():
        with urllib.request.urlopen(WikiUrl, timeout=60) as response:
            data = zipfile.ZipFile(io.BytesIO(response.read())).read(WikiMember)
        Wiki.parent.mkdir(parents=True, exist_ok=True)
        Wiki.write_bytes(data)
    if hashlib.sha256(Wiki.read_bytes()).hexdigest() != WikiSha:
        raise SystemExit(f"{Wiki} is not the WikiText-2 test set the results were scored on: "
                         "delete it to download it again")
    return Wiki


def shuffled[T, R](generate: Callable[[list[T]], Sequence[R]], items: list[T], rng: np.random.Generator) -> list[R]:
    """generate's outputs for the items, requested in a random order and returned in the items' order."""
    order = rng.permutation(len(items))
    outputs = generate([items[i] for i in order])
    return [outputs[i] for i in np.argsort(order)]


def put(ids_out: np.ndarray, lp_out: np.ndarray, logprobs: dict) -> None:
    """Write one position's top-k ids and logprobs, best first."""
    ranked = sorted(logprobs.items(), key=lambda kv: -kv[1].logprob)[:TopK]
    for k, (tid, lp) in enumerate(ranked):
        ids_out[k], lp_out[k] = tid, lp.logprob


def load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "meta"} | {"meta": json.loads(str(z["meta"]))}


def judge(refs: list[dict], cands: list[dict]) -> dict:
    if len(refs) < 2 or len(cands) < 2:
        raise SystemExit("at least 2 runs of each build are needed to measure rerun noise")
    for side, runs in (("reference", refs), ("candidate", cands)):
        if len(versions := {run["meta"]["vllm"] for run in runs}) > 1:
            raise SystemExit(f"the {side} runs come from vLLM {', '.join(sorted(versions))}: "
                             "pass the runs of one build per side")
    seeds = [run["meta"]["seed"] for run in refs + cands]
    if repeated := sorted({s for s in seeds if seeds.count(s) > 1}):
        raise SystemExit(f"runs repeat seed {', '.join(map(str, repeated))}: give every run its own seed, "
                         "e.g. 0 1 2 for the reference and 3 4 5 for the candidate")
    check_comparable(refs + cands)
    s = scores(probe_kl, traffic_kl, refs, cands)
    return {"stats": stats(refs[0], cands[0]), "scores": s, "thresholds": Thresholds, "changed": changed(s)}


def check_comparable(runs: list[dict]) -> None:
    """Exit unless the runs differ only in vLLM build and seed."""
    setups = [{k: v for k, v in run["meta"].items() if k not in ("vllm", "seed")} for run in runs]
    for setup in setups[1:]:
        if differ := sorted(k for k in setup.keys() | setups[0].keys() if setup.get(k) != setups[0].get(k)):
            raise SystemExit(f"runs differ in {', '.join(differ)}: score every run with the same model, "
                             "engine, env, prompts and text")
    for run in runs[1:]:
        if not np.array_equal(run["probe_tokens"], runs[0]["probe_tokens"]):
            raise SystemExit("runs scored different probe tokens: the builds tokenize the text differently")


def scores[R](probe: Callable[[R, R], float], traffic: Callable[[R, R], float],
              refs: list[R], cands: list[R]) -> dict[str, float]:
    return {"probes": ratio_to_noise(probe, refs, cands, (refs, cands)),
            "traffic": ratio_to_noise(traffic, refs, cands, (refs,))}


def changed(scores: dict[str, float]) -> bool:
    return any(not scores[k] <= Thresholds[k] for k in scores)  # NaN reads changed


def ratio_to_noise[R](kl: Callable[[R, R], float], refs: list[R], cands: list[R],
                      noise_sides: tuple[list[R], ...]) -> float:
    """Mean KL across builds over the mean KL between reruns within each of noise_sides, both ways round."""
    across = np.mean([kl(r, c) for r in refs for c in cands])
    noise = np.mean([kl(a, b) for side in noise_sides for a, b in itertools.permutations(side, 2)])
    # Bit-identical reruns leave no noise to scale by: any difference at all is beyond it.
    if noise == 0:
        return 0.0 if across == 0 else math.inf
    return float(across / noise)


def probe_kl(a: dict, b: dict) -> float:
    return float(bucketed_kl(a["probe_ids"], a["probe_lp"], b["probe_ids"], b["probe_lp"]).mean())


def traffic_kl(a: dict, b: dict) -> float:
    """Mean over prompts of the mean KL over the generated positions both runs condition alike.

    Greedy outputs share their text up to their first differing token, so the positions up
    to and including it, or up to the shorter output's end, are compared. A prompt with no
    such position counts as 0.
    """
    n = np.minimum(a["traffic_lengths"], b["traffic_lengths"])
    differ = a["traffic_tokens"] != b["traffic_tokens"]
    before = np.cumsum(differ, axis=1) - differ  # differing tokens before each position
    compared = (before == 0) & (np.arange(differ.shape[1]) < n[:, None])
    kl = np.zeros(compared.shape)
    kl[compared] = bucketed_kl(a["traffic_ids"][compared], a["traffic_lp"][compared],
                               b["traffic_ids"][compared], b["traffic_lp"][compared])
    return float(np.mean(kl.sum(axis=1) / np.maximum(compared.sum(axis=1), 1)))


def bucketed_kl(ids_a: np.ndarray, lp_a: np.ndarray, ids_b: np.ndarray, lp_b: np.ndarray) -> np.ndarray:
    """KL(a || b) per top-k list on the last axis, over the ids both list plus one bucket for the rest.

    Merging ids into a bucket never raises KL, so this is a lower bound on the true KL.
    """
    pa = np.exp(lp_a.astype(np.float64))
    # Padding ids (-1) have -inf logprobs: like any id a gives no mass, they join the bucket.
    # NaN stays in, so a corrupt run reads as changed.
    match = (ids_a[..., :, None] == ids_b[..., None, :]) & (pa != 0)[..., None]
    shared = match.any(axis=-1)
    pa = np.where(shared, pa, 0.0)
    pb = (match * np.exp(lp_b.astype(np.float64))[..., None, :]).sum(axis=-1)  # b's prob of a's shared ids
    tail_a = np.maximum(1 - pa.sum(axis=-1), TailFloor)
    tail_b = np.maximum(1 - pb.sum(axis=-1), TailFloor)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shared, pa * (np.log(pa) - np.log(pb)), 0.0)
    return terms.sum(axis=-1) + tail_a * np.log(tail_a / tail_b)


def stats(ref: dict, cand: dict) -> dict:
    d = bucketed_kl(ref["probe_ids"], ref["probe_lp"], cand["probe_ids"], cand["probe_lp"]).ravel()
    dp = np.exp(cand["probe_token_lp"].astype(np.float64)) - np.exp(ref["probe_token_lp"].astype(np.float64))
    same = ref["probe_ids"][..., 0] == cand["probe_ids"][..., 0]
    n = d.size
    return {
        "positions": n,
        "kl_mean": float(d.mean()), "kl_se": float(d.std(ddof=1) / np.sqrt(n)),
        "kl_median": float(np.median(d)), "kl_p99": float(np.quantile(d, 0.99)), "kl_max": float(d.max()),
        "rms_dp": float(np.sqrt((dp**2).mean())),
        "same_top": float(same.mean()), "same_top_se": float(np.sqrt(same.mean() * (1 - same.mean()) / n)),
        # Outputs are padded with -1 past their end, so outputs of different lengths differ here too.
        "traffic_differ": float((ref["traffic_tokens"] != cand["traffic_tokens"]).any(axis=1).mean()),
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
        f"Traffic: {100 * s['traffic_differ']:.1f} % of outputs differ",
        "",
        f"Verdict: {verdict} (probes {sc['probes']:.2f}× the rerun noise, threshold {th['probes']}×; "
        f"traffic {sc['traffic']:.2f}×, threshold {th['traffic']}×)",
    ])


def jsonable(x):
    """x with its infinite and NaN floats as the strings "inf", "-inf" and "nan", which JSON lacks."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


if __name__ == "__main__":
    main()
