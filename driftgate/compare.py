"""Compare two captured runs on the positions where both saw the same prefix.

Greedy runs share a prefix until their first differing token, so up to and including
that position the two next-token distributions are conditioned on the same text and can
be compared directly, with no reference model. KL is computed on the ids both top-20
lists hold plus one bucket for the rest, which by the data-processing inequality never
exceeds the true KL.
"""

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TailFloor = 1e-9


@dataclass(frozen=True)
class Run:
    tokens: np.ndarray  # [prompts, T], -1 past the end
    top_ids: np.ndarray  # [prompts, T, 20]
    top_lp: np.ndarray  # [prompts, T, 20]
    lengths: np.ndarray  # [prompts]


@dataclass(frozen=True)
class PromptDiff:
    kl: float  # mean bucketed KL(a || b) over compared positions
    positions: int  # positions compared
    diverged: bool  # greedy outputs differ


def load(path: Path) -> Run:
    z = np.load(path)
    return Run(z["tokens"], z["top_ids"], z["top_lp"], z["lengths"])


def compare(a: Run, b: Run) -> list[PromptDiff]:
    if a.tokens.shape != b.tokens.shape:
        raise ValueError(f"runs cover different prompts or lengths: {a.tokens.shape} vs {b.tokens.shape}")
    return [prompt_diff(a, b, i) for i in range(len(a.tokens))]


def prompt_diff(a: Run, b: Run, i: int) -> PromptDiff:
    n = min(a.lengths[i], b.lengths[i])
    differ = np.nonzero(a.tokens[i, :n] != b.tokens[i, :n])[0]
    diverged = len(differ) > 0 or a.lengths[i] != b.lengths[i]
    stop = differ[0] + 1 if len(differ) else n
    kls = [bucketed_kl(a.top_ids[i, t], a.top_lp[i, t], b.top_ids[i, t], b.top_lp[i, t]) for t in range(stop)]
    return PromptDiff(float(np.mean(kls)) if kls else 0.0, int(stop), diverged)


def bucketed_kl(ids_a: np.ndarray, lp_a: np.ndarray, ids_b: np.ndarray, lp_b: np.ndarray) -> float:
    common, ia, ib = np.intersect1d(ids_a, ids_b, assume_unique=True, return_indices=True)
    pa, pb = np.exp(lp_a[ia].astype(np.float64)), np.exp(lp_b[ib].astype(np.float64))
    tail_a, tail_b = max(1 - pa.sum(), TailFloor), max(1 - pb.sum(), TailFloor)
    return float(np.sum(pa * (np.log(pa) - np.log(pb))) + tail_a * np.log(tail_a / tail_b))


def summarize(diffs: list[PromptDiff]) -> str:
    kl = np.array([d.kl for d in diffs])
    diverged = sum(d.diverged for d in diffs)
    return (
        f"mean KL {kl.mean():.2e}  median {np.median(kl):.2e}  max {kl.max():.2e}  "
        f"diverged {diverged}/{len(diffs)}"
    )


def main() -> None:
    """Usage: compare.py base_dir cand_dir — every run pair within and across the two."""
    base = sorted(Path(sys.argv[1]).glob("*.npz"))
    cand = sorted(Path(sys.argv[2]).glob("*.npz"))
    runs = {p: load(p) for p in base + cand}
    for label, pairs in [
        ("base vs base", itertools.combinations(base, 2)),
        ("cand vs cand", itertools.combinations(cand, 2)),
        ("base vs cand", itertools.product(base, cand)),
    ]:
        for x, y in pairs:
            print(f"{label}  {x.stem} {y.stem}  {summarize(compare(runs[x], runs[y]))}")


if __name__ == "__main__":
    main()
