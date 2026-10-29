"""Generate with one vLLM build and save each request's tokens and top-k logprobs.

Runs inside a build's venv (vLLM's own dependencies only), one fresh engine per run.
Offline mode batches every prompt in one engine call, which is deterministic run to run.
Serve mode launches `vllm serve` and sends requests concurrently over HTTP, so batch
composition varies with timing as it does in production.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import aiohttp
import numpy as np
import vllm

TopK = 20

# One generated sequence: its token ids, and per position {token id: logprob} holding the
# top-k ids plus the chosen token.
Sequence = tuple[list[int], list[dict[int, float]]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "serve"], default="offline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with a 'prompt' field per line")
    ap.add_argument("--out", required=True, help=".npz path")
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--stop", default="[]", help="json list of stop strings")
    ap.add_argument("--engine", default="{}", help="json of engine arguments")
    ap.add_argument("--concurrency", type=int, default=32, help="serve mode: requests in flight")
    args = ap.parse_args()

    prompts = [json.loads(line)["prompt"] for line in open(args.prompts)]
    engine, stop = json.loads(args.engine), json.loads(args.stop)
    start = time.time()
    if args.mode == "offline":
        seqs = offline(args.model, engine, prompts, args.max_tokens, stop)
    else:
        seqs = serve(args.model, engine, prompts, args.max_tokens, stop, args.concurrency)
    meta = {
        "vllm": vllm.__version__,
        "mode": args.mode,
        "model": args.model,
        "engine": engine,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency if args.mode == "serve" else None,
        "seconds": time.time() - start,
    }
    save(args.out, seqs, args.max_tokens, meta)
    print(json.dumps(meta))


def offline(model: str, engine: dict, prompts: list[str], max_tokens: int, stop: list[str]) -> list[Sequence]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, seed=0, **engine)
    params = SamplingParams(temperature=0, max_tokens=max_tokens, stop=stop, logprobs=TopK)
    seqs = []
    for out in llm.generate(prompts, params):
        seq = out.outputs[0]
        seqs.append((list(seq.token_ids), [{tid: lp.logprob for tid, lp in d.items()} for d in seq.logprobs]))
    return seqs


def serve(
    model: str, engine: dict, prompts: list[str], max_tokens: int, stop: list[str], concurrency: int
) -> list[Sequence]:
    port = free_port()
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--port", str(port), "--seed", "0", *cli_flags(engine)]
    # Its own process group, so the engine core it spawns dies with it; an orphaned
    # engine would hold GPU memory and starve the next run.
    server = subprocess.Popen(cmd, start_new_session=True)
    try:
        wait_healthy(server, port)
        return asyncio.run(send_all(port, model, prompts, max_tokens, stop, concurrency))
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)


def cli_flags(engine: dict) -> list[str]:
    flags = []
    for key, value in engine.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value is not False:
            flags += [flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return flags


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(server: subprocess.Popen, port: int, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm server exited with code {server.returncode} before becoming healthy")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"vllm server not healthy after {timeout:.0f}s")


async def send_all(
    port: int, model: str, prompts: list[str], max_tokens: int, stop: list[str], concurrency: int
) -> list[Sequence]:
    gate = asyncio.Semaphore(concurrency)
    url = f"http://127.0.0.1:{port}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def one(prompt: str) -> Sequence:
            body = {
                "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                "stop": stop, "logprobs": TopK, "return_tokens_as_token_ids": True,
            }
            async with gate, session.post(url, json=body) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"request failed with {resp.status}: {await resp.text()}")
                lp = (await resp.json())["choices"][0]["logprobs"]
            tokens = [token_id(t) for t in lp["tokens"]]
            tops = [
                {token_id(t): v for t, v in top.items()} | {tok: chosen}
                for tok, chosen, top in zip(tokens, lp["token_logprobs"], lp["top_logprobs"])
            ]
            return tokens, tops

        return await asyncio.gather(*(one(p) for p in prompts))


def token_id(token: str) -> int:
    # With return_tokens_as_token_ids, tokens come back as "token_id:<id>".
    prefix, _, tid = token.partition(":")
    if prefix != "token_id":
        raise ValueError(f"expected a 'token_id:<id>' token, got {token!r}")
    return int(tid)


def save(path: str, seqs: list[Sequence], max_tokens: int, meta: dict) -> None:
    n = len(seqs)
    tokens = np.full((n, max_tokens), -1, np.int32)
    top_ids = np.full((n, max_tokens, TopK), -1, np.int32)
    top_lp = np.full((n, max_tokens, TopK), -np.inf, np.float32)
    token_lp = np.full((n, max_tokens), -np.inf, np.float32)
    lengths = np.zeros(n, np.int32)
    for i, (ids, lps) in enumerate(seqs):
        lengths[i] = len(ids)
        for t, (tok, lp) in enumerate(zip(ids, lps)):
            tokens[i, t] = tok
            token_lp[i, t] = lp[tok]
            ranked = sorted(lp.items(), key=lambda kv: -kv[1])[:TopK]
            for k, (tid, v) in enumerate(ranked):
                top_ids[i, t, k] = tid
                top_lp[i, t, k] = v
    np.savez_compressed(
        path, tokens=tokens, top_ids=top_ids, top_lp=top_lp, token_lp=token_lp,
        lengths=lengths, meta=json.dumps(meta),
    )


if __name__ == "__main__":
    main()
