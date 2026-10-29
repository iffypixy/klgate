"""Does a candidate vLLM build change a model's outputs beyond rerun noise?

Usage: python -m driftgate.diff <baseline> <candidate> --model NAME [--engine JSON] [--env KEY=VALUE ...]

Each build is a vLLM tag, branch or commit: v0.20.0, main, 124fac10.

Runs 5 baseline and 3 candidate engines, alternating, on the diff workload. Each candidate
run is scored by its median KL to the baseline runs, over the median KL between baseline
runs. The verdict uses the middle candidate score: an occasional launch lands in a slightly
different numeric state and disagrees with every other run, but a real change shows in
every candidate run. Off launches are common enough that two of three can land together,
so a first-round "changed" is confirmed with 3 more candidate runs and the middle of all 6
decides. "Changed" means the outputs moved, not that they got worse; accuracy is shown for
context.
"""

import argparse
import hashlib
import itertools
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from driftgate import build, gsm8k
from driftgate.compare import Run, compare, load
from driftgate.run import Root, capture, hardware

BaselineRuns = 5
CandidateRuns = 3  # per round; a first-round "changed" adds a second round
# Set on unchanged code (reruns of one build, and commit pairs that change no engine code)
# across 4 models on 2 GPUs: with the confirmation round, at most 1.5% of checks on any of
# them say "changed". `python -m driftgate.threshold` replays it.
Threshold = 1.25
Shown = 5  # most-changed prompts to print


@dataclass(frozen=True)
class Verdict:
    noise: float  # median mean-token KL between baseline runs
    scores: list[float]  # per candidate run, in run order
    per_prompt: np.ndarray  # mean candidate-vs-baseline KL minus mean baseline-vs-baseline KL

    @property
    def score(self) -> float:
        return float(np.median(self.scores))

    @property
    def changed(self) -> bool:
        return self.score > Threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=build.resolve, help="the vLLM build you trust: a tag, branch or commit")
    ap.add_argument("candidate", type=build.resolve, help="the vLLM build to check")
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default="{}", help="json of engine arguments, e.g. '{\"max_model_len\": 2048}'")
    ap.add_argument("--env", action="append", default=[], type=env_var, metavar="KEY=VALUE", help="environment for both builds")
    args = ap.parse_args()

    spec = tomllib.loads((Root / "bench" / "workloads" / "diff.toml").read_text())
    setup = spec | {"model": args.model, "engine": json.loads(args.engine), "env": dict(args.env)}
    for sha in (args.baseline, args.candidate):
        build.install(sha)
    # Runs from different engine settings must never mix, so the settings name the folder.
    fingerprint = hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:8]
    out = Root / "runs" / hardware() / "diff" / f"{args.baseline[:12]}-{args.candidate[:12]}-{slug(args.model)}-{fingerprint}"

    base, cand = collect(args, setup, out, CandidateRuns)
    first = judge(base, cand)
    if first.changed:
        print(f"the first {CandidateRuns} candidate runs read {first.score:.2f}×; confirming with {CandidateRuns} more", flush=True)
        base, cand = collect(args, setup, out, 2 * CandidateRuns)
    print(render(args, setup, base, cand, judge(base, cand)))


def collect(args: argparse.Namespace, setup: dict, out: Path, candidates: int) -> tuple[list[Run], list[Run]]:
    """Capture whichever runs are missing, alternating the two builds, and load them."""
    for i in range(max(BaselineRuns, candidates)):
        for side, sha, n in (("baseline", args.baseline, BaselineRuns), ("candidate", args.candidate, candidates)):
            path = out / side / f"{i}.npz"
            if i < n and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"{side} run {i + 1}/{n}", flush=True)
                capture(setup, sha, path)
    base = [load(out / "baseline" / f"{i}.npz") for i in range(BaselineRuns)]
    cand = [load(out / "candidate" / f"{i}.npz") for i in range(candidates)]
    return base, cand


def judge(base: list[Run], cand: list[Run]) -> Verdict:
    within = [[d.kl for d in compare(a, b)] for a, b in itertools.combinations(base, 2)]
    cross = [[[d.kl for d in compare(b, c)] for b in base] for c in cand]  # [candidate][baseline][prompt]
    noise = float(np.median([np.mean(k) for k in within]))
    if noise == 0:
        raise SystemExit(
            "most baseline runs are bit-identical, so there is no rerun noise to measure a change "
            "against: delete the baseline runs and run again"
        )
    scores = [float(np.median([np.mean(k) for k in per_base])) / noise for per_base in cross]
    per_prompt = np.mean(cross, axis=(0, 1)) - np.mean(within, axis=0)
    return Verdict(noise, scores, per_prompt)


def render(args: argparse.Namespace, setup: dict, base: list[Run], cand: list[Run], v: Verdict) -> str:
    tok = Tokenizer.from_pretrained(args.model)
    gold = gsm8k.answers(Root / setup["prompts"])
    accuracy = lambda runs: np.mean([gsm8k.correct(r, gold, tok, setup["stop"]).mean() for r in runs])
    verdict = "CHANGED" if v.changed else "same within noise"
    lines = [
        f"baseline {args.baseline[:12]} → candidate {args.candidate[:12]} · {args.model} · {hardware()} · {len(gold)} prompts",
        "",
        f"Verdict: {verdict} (middle of {len(v.scores)} candidate runs {v.score:.2f}× the noise; threshold {Threshold}×)",
        f"Candidate runs: {', '.join(f'{s:.2f}×' for s in v.scores)} the baseline-vs-baseline KL of {v.noise:.2e}",
        f"GSM8K accuracy: baseline {accuracy(base):.3f}, candidate {accuracy(cand):.3f} (context, not part of the verdict)",
    ]
    if len(v.scores) > CandidateRuns:
        lines.insert(4, f"The first {CandidateRuns} runs read over the threshold, so {len(v.scores) - CandidateRuns} more ran and all {len(v.scores)} decide.")
    if v.changed:
        a, b = base[0], cand[0]
        lines += ["", "Most-changed prompts whose text differs (baseline run 1 vs candidate run 1, from the first differing token):"]
        for i in [i for i in np.argsort(v.per_prompt)[::-1] if first_difference(a, b, i) is not None][:Shown]:
            start = first_difference(a, b, i)
            snippet = lambda r: json.dumps(tok.decode(r.tokens[i, start : r.lengths[i]].tolist())[:80], ensure_ascii=False)
            lines += [f"  #{i + 1}, token {start + 1}:", f"    baseline   {snippet(a)}", f"    candidate  {snippet(b)}"]
    return "\n".join(lines)


def first_difference(a: Run, b: Run, i: int) -> int | None:
    """The first token position where prompt i's outputs differ, or None if the text is the same."""
    n = min(a.lengths[i], b.lengths[i])
    differ = np.nonzero(a.tokens[i, :n] != b.tokens[i, :n])[0]
    if len(differ):
        return int(differ[0])
    return int(n) if a.lengths[i] != b.lengths[i] else None


def env_var(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


if __name__ == "__main__":
    main()
