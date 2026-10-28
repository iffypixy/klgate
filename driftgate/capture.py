"""Generate with one vLLM build and save each request's tokens and top-k logprobs.

Runs inside a build's venv (vLLM + numpy only), one fresh engine per run.
"""

import argparse
import json
import time

import numpy as np
from vllm import LLM, SamplingParams
import vllm

TopK = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field per line")
    ap.add_argument("--out", required=True, help=".npz path")
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--stop", default="[]", help="json list of stop strings")
    ap.add_argument("--engine", default="{}", help="json of extra LLM(...) kwargs")
    args = ap.parse_args()

    prompts = [json.loads(line)["prompt"] for line in open(args.prompts)]
    engine = json.loads(args.engine)
    llm = LLM(model=args.model, seed=0, **engine)
    # Stop strings need detokenized text, so detokenization stays on.
    params = SamplingParams(
        temperature=0, max_tokens=args.max_tokens, stop=json.loads(args.stop), logprobs=TopK,
    )

    start = time.time()
    outputs = llm.generate(prompts, params)
    seconds = time.time() - start

    n, T = len(outputs), args.max_tokens
    tokens = np.full((n, T), -1, np.int32)
    top_ids = np.full((n, T, TopK), -1, np.int32)
    top_lp = np.full((n, T, TopK), -np.inf, np.float32)
    token_lp = np.full((n, T), -np.inf, np.float32)
    lengths = np.zeros(n, np.int32)
    for i, out in enumerate(outputs):
        seq = out.outputs[0]
        lengths[i] = len(seq.token_ids)
        for t, (tok, lps) in enumerate(zip(seq.token_ids, seq.logprobs)):
            tokens[i, t] = tok
            token_lp[i, t] = lps[tok].logprob
            ranked = sorted(lps.items(), key=lambda kv: kv[1].rank)[:TopK]
            for k, (tid, lp) in enumerate(ranked):
                top_ids[i, t, k] = tid
                top_lp[i, t, k] = lp.logprob

    meta = {
        "vllm": vllm.__version__,
        "model": args.model,
        "engine": engine,
        "max_tokens": T,
        "seconds": seconds,
    }
    np.savez_compressed(
        args.out, tokens=tokens, top_ids=top_ids, top_lp=top_lp, token_lp=token_lp,
        lengths=lengths, meta=json.dumps(meta),
    )
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
