"""Score every detector on every pair of runs of an entry.

Same-build pairs measure false alarms; buggy-vs-fixed pairs measure detection. The KL
threshold for judging a pair is the largest KL among the entry's other same-build pairs,
so no pair is judged by a threshold fitted on itself.

Usage: python -m driftgate.report bench/entries/<id>.toml [workload]
"""

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from driftgate import gsm8k
from driftgate.compare import compare, load
from driftgate.run import Root, load_entry, runs_dir

GateTolerance = 0.08  # vLLM's default GSM8K tolerance
Alpha = 0.05


@dataclass(frozen=True)
class Pair:
    a: str  # "fixed/0"
    b: str
    same_build: bool
    kl: float  # mean over prompts of the bucketed per-prompt KL
    diverged: float  # fraction of prompts whose greedy outputs differ
    accuracy_drop: float  # accuracy(a) - accuracy(b)
    mcnemar_p: float


def main() -> None:
    entry = load_entry(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
    pairs = score(entry)
    print(render(entry, pairs))


def score(entry: dict) -> list[Pair]:
    gold = gsm8k.answers(Root / entry["prompts"])
    tok = Tokenizer.from_pretrained(entry["model"])
    names = [f"{side}/{p.stem}" for side in ("fixed", "buggy") for p in sorted((runs_dir(entry) / side).glob("*.npz"))]
    if not any(n.startswith("buggy") for n in names) or not any(n.startswith("fixed") for n in names):
        sys.exit(f"no runs for both builds in {runs_dir(entry)}: run `python -m driftgate.run` first")
    runs = {n: load(runs_dir(entry) / f"{n}.npz") for n in names}
    correct = {n: gsm8k.correct(r, gold, tok) for n, r in runs.items()}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        diffs = compare(runs[a], runs[b])
        pairs.append(Pair(
            a, b,
            same_build=a.split("/")[0] == b.split("/")[0],
            kl=float(np.mean([d.kl for d in diffs])),
            diverged=float(np.mean([d.diverged for d in diffs])),
            accuracy_drop=float(correct[a].mean() - correct[b].mean()),
            mcnemar_p=gsm8k.mcnemar(correct[a], correct[b]),
        ))
    return pairs


def render(entry: dict, pairs: list[Pair]) -> str:
    same = [p for p in pairs if p.same_build]
    cross = [p for p in pairs if not p.same_build]
    detectors = {
        f"GSM8K gate ({GateTolerance:.0%} drop)": lambda p: abs(p.accuracy_drop) > GateTolerance,
        f"Paired McNemar (p < {Alpha})": lambda p: p.mcnemar_p < Alpha,
        "Token KL above rerun max": lambda p: p.kl > max((q.kl for q in same if q is not p), default=0.0),
    }
    rows = [
        f"| {name} | {sum(map(flag, same))}/{len(same)} | {sum(map(flag, cross))}/{len(cross)} |"
        for name, flag in detectors.items()
    ]
    accuracy = lambda ps: f"{min(abs(p.accuracy_drop) for p in ps):.3f}–{max(abs(p.accuracy_drop) for p in ps):.3f}"
    span = lambda ps, f: f"{min(map(f, ps)):.2e}–{max(map(f, ps)):.2e}"
    return "\n".join([
        f"## {entry['id']} · {entry['workload']}",
        "",
        entry["summary"],
        "",
        "| Detector | Same-build pairs flagged | Buggy-vs-fixed pairs flagged |",
        "| --- | --- | --- |",
        *rows,
        "",
        "| Measure | Same-build pairs | Buggy-vs-fixed pairs |",
        "| --- | --- | --- |",
        f"| Mean token KL | {span(same, lambda p: p.kl)} | {span(cross, lambda p: p.kl)} |",
        f"| Prompts whose output differs | {span(same, lambda p: p.diverged)} | {span(cross, lambda p: p.diverged)} |",
        f"| GSM8K accuracy gap | {accuracy(same)} | {accuracy(cross)} |",
        f"| Smallest McNemar p | {min(p.mcnemar_p for p in same):.3f} | {min(p.mcnemar_p for p in cross):.3f} |",
    ])


if __name__ == "__main__":
    main()
