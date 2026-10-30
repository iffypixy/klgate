"""Score every detector on every pair of runs of an entry.

Pairs of fixed-build reruns measure false alarms; buggy-vs-fixed pairs measure detection.
The KL threshold for judging a pair is the 95th percentile of KL over the fixed-build pairs
that share no run with it, so no pair is judged by a threshold fitted on its own runs. Buggy-vs-buggy pairs are left
out: a CI gate only ever has reruns of its known-good baseline to calibrate on, and some
bugs make the buggy build disagree with itself.

Usage: python -m klgate.report bench/entries/<id>.toml [workload]
       python -m klgate.report all [workload]    # one row per entry with runs
"""

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from klgate import gsm8k
from klgate.compare import compare, load
from klgate.run import Root, load_entry, runs_dir

# vLLM's gate fails when accuracy < expected - 0.08. Its expected value is a stored
# per-model number; here it is the fixed build's own accuracy in the pair, the tightest
# reference the gate could have, so these gate results are its best case.
GateTolerance = 0.08
Alpha = 0.05
KlQuantile = 0.95  # the spec's 5% false-alarm target on baseline reruns
MinCalibration = 20  # pairs a 95th percentile needs; 8 fixed runs give 21 per buggy pair


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
    workload = sys.argv[2] if len(sys.argv) > 2 else None
    if sys.argv[1] != "all":
        entry = load_entry(Path(sys.argv[1]), workload)
        print(render(entry, *score(entry)))
        return
    entries = [load_entry(p, workload) for p in sorted((Root / "bench" / "entries").glob("*.toml"))]
    ready = [e for e in entries if all(any((runs_dir(e) / s).glob("*.npz")) for s in ("fixed", "buggy"))]
    print(summary(ready, workload or "each entry's own workload"))


def score(entry: dict) -> tuple[list[Pair], dict[str, float]]:
    """Every run pair, and each build's mean GSM8K accuracy."""
    gold = gsm8k.answers(Root / entry["prompts"])
    tok = Tokenizer.from_pretrained(entry["model"])
    names = [f"{side}/{p.stem}" for side in ("fixed", "buggy") for p in sorted((runs_dir(entry) / side).glob("*.npz"))]
    if not any(n.startswith("buggy") for n in names) or not any(n.startswith("fixed") for n in names):
        sys.exit(f"no runs for both builds in {runs_dir(entry)}: run `python -m klgate.run` first")
    runs = {n: load(runs_dir(entry) / f"{n}.npz") for n in names}
    correct = {n: gsm8k.correct(r, gold, tok, entry["stop"]) for n, r in runs.items()}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        if a.startswith("buggy"):  # names list fixed first, so both are buggy
            continue
        diffs = compare(runs[a], runs[b])
        pairs.append(Pair(
            a, b,
            same_build=a.split("/")[0] == b.split("/")[0],
            kl=float(np.mean([d.kl for d in diffs])),
            diverged=float(np.mean([d.diverged for d in diffs])),
            accuracy_drop=float(correct[a].mean() - correct[b].mean()),
            mcnemar_p=gsm8k.mcnemar(correct[a], correct[b]),
        ))
    accuracy = {side: float(np.mean([c.mean() for n, c in correct.items() if n.startswith(side)])) for side in ("fixed", "buggy")}
    return pairs, accuracy


def detectors(same: list[Pair]) -> dict:
    return {
        f"GSM8K gate ({GateTolerance:.0%} drop)": lambda p: p.accuracy_drop > GateTolerance,
        f"Paired McNemar (p < {Alpha})": lambda p: p.mcnemar_p < Alpha,
        f"Token KL above rerun {KlQuantile:.0%} quantile": lambda p: None if (t := kl_threshold(p, same)) is None else p.kl > t,
    }


def kl_threshold(p: Pair, same: list[Pair]) -> float | None:
    """The KL quantile over fixed-build pairs disjoint from p, or None with too few to set it."""
    rest = [q.kl for q in same if not {q.a, q.b} & {p.a, p.b}]
    return float(np.quantile(rest, KlQuantile)) if len(rest) >= MinCalibration else None


def flagged(flag, pairs: list[Pair]) -> str:
    """How many pairs a detector flags, or a dash when it cannot judge them yet."""
    verdicts = [flag(p) for p in pairs]
    return "—" if None in verdicts else f"{sum(verdicts)}/{len(verdicts)}"


def summary(entries: list[dict], workload: str) -> str:
    """One row per entry: each detector's detections, with its false alarms in brackets."""
    names = list(detectors([]))
    lines = [
        f"## All entries · {workload}",
        "",
        f"Each cell: buggy-vs-fixed pairs flagged (fixed-build reruns flagged). KL shows a dash until an entry has the {MinCalibration} rerun pairs its threshold needs. A null pair changes no engine code, so it runs its baseline as \"fixed\" and its candidate as \"buggy\", and any flag on it is a false positive.",
        "",
        "| Entry | GSM8K accuracy, fixed → buggy | Mean KL, reruns → buggy vs fixed | " + " | ".join(names) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for entry in entries:
        pairs, accuracy = score(entry)
        same = [p for p in pairs if p.same_build]
        cross = [p for p in pairs if not p.same_build]
        cells = [f"{flagged(f, cross)} ({flagged(f, same)})" for f in detectors(same).values()]
        mean = lambda ps: f"{np.mean([p.kl for p in ps]):.1e}" if ps else "—"
        kl = f"{mean(same)} → {mean(cross)}"
        lines.append(f"| {entry['id']} | {accuracy['fixed']:.3f} → {accuracy['buggy']:.3f} | {kl} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(entry: dict, pairs: list[Pair], accuracy: dict[str, float]) -> str:
    same = [p for p in pairs if p.same_build]
    cross = [p for p in pairs if not p.same_build]
    rows = [
        f"| {name} | {flagged(flag, same)} | {flagged(flag, cross)} |"
        for name, flag in detectors(same).items()
    ]
    gap = lambda ps: f"{min(abs(p.accuracy_drop) for p in ps):.3f}–{max(abs(p.accuracy_drop) for p in ps):.3f}" if ps else "—"
    span = lambda ps, f: f"{min(map(f, ps)):.2e}–{max(map(f, ps)):.2e}" if ps else "—"
    smallest = lambda ps: f"{min(p.mcnemar_p for p in ps):.3f}" if ps else "—"
    null = entry["kind"] == "null-pair"
    base, test = ("baseline", "candidate") if null else ("fixed", "buggy")
    reruns, versus = f"{base.capitalize()} reruns", f"{test.capitalize()}-vs-{base} pairs"
    return "\n".join([
        f"## {entry['id']} · {entry['workload']}",
        "",
        entry["summary"],
        "",
        f"GSM8K accuracy: {base} {accuracy['fixed']:.3f}, {test} {accuracy['buggy']:.3f}.",
        "",
        f"| Detector | {reruns} flagged | {versus} flagged{' (false positives)' if null else ''} |",
        "| --- | --- | --- |",
        *rows,
        "",
        f"| Measure | {reruns} | {versus} |",
        "| --- | --- | --- |",
        f"| Mean token KL | {span(same, lambda p: p.kl)} | {span(cross, lambda p: p.kl)} |",
        f"| Prompts whose output differs | {span(same, lambda p: p.diverged)} | {span(cross, lambda p: p.diverged)} |",
        f"| GSM8K accuracy gap | {gap(same)} | {gap(cross)} |",
        f"| Smallest McNemar p | {smallest(same)} | {smallest(cross)} |",
    ])


if __name__ == "__main__":
    main()
