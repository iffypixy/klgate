"""Where driftgate.diff's threshold comes from, and how often its rule is right.

Usage: python -m driftgate.threshold                     # simulate from data/kl
       python -m driftgate.threshold cache NAME RUNS_DIR    # add a source to data/kl

A source is one GPU's runs of one setup: "fixed" runs of the trusted build and "buggy" runs of
the other one. data/kl keeps the per-prompt KL of every pair of a source's runs, so the rule
can be replayed on thousands of random picks of runs and prompts without a GPU. Unchanged-code
sources (reruns of one build, null pairs) give false alarms; bug sources give detections.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

from driftgate.compare import compare, load
from driftgate.diff import BaselineRuns, CandidateRuns, Threshold
from driftgate.run import Root

Data = Root / "data" / "kl"
Prompts = 250
Draws = 4000


def main() -> None:
    if sys.argv[1:2] == ["cache"]:
        cache(sys.argv[2], Path(sys.argv[3]))
        return
    rng = np.random.default_rng(0)
    rows, few = [], []
    for path in sorted(Data.glob("*.npz")):
        K, fixed, buggy = read(path)
        if len(fixed) < BaselineRuns:
            few.append(f"{path.stem}: {len(fixed)} fixed and {len(buggy)} buggy runs, too few to simulate; "
                       f"scored once on all runs and prompts, the middle candidate reads {one_check(K, fixed, buggy):.2f}×")
            continue
        if len(fixed) >= BaselineRuns + 2 * CandidateRuns:
            first, final = simulate(K, fixed, fixed, rng)
            rows.append((f"{path.stem} reruns", "false alarm", first, final))
        unchanged = "-null-" in path.stem
        first, final = simulate(K, fixed, buggy, rng)
        rows.append((f"{path.stem} {'pair' if unchanged else 'bug'}", "false alarm" if unchanged else "detection", first, final))

    worst = max(np.mean(final) for name, kind, first, final in rows if kind == "false alarm")
    print(f"{BaselineRuns} baseline + {CandidateRuns} candidate runs (+{CandidateRuns} to confirm), {Prompts} prompts, {Draws} draws each")
    print(f"Threshold {Threshold}: at most {worst:.1%} false alarms on any unchanged-code source\n")
    print(f"{'source':34s} {'counts as':12s} {'one round':>9s} {'confirmed':>9s} {'median score':>12s}")
    for name, kind, first, final in rows:
        print(f"{name:34s} {kind:12s} {np.mean(first > Threshold):9.1%} {np.mean(final):9.1%} {np.nanmedian(first):11.2f}×")
    print("", *few, sep="\n")


def one_check(K: np.ndarray, base: list[int], cand: list[int]) -> float:
    M = K.mean(-1)
    noise = np.median([M[i, j] for i, j in itertools.combinations(base, 2)])
    return float(np.median([np.median(M[base, c]) for c in cand]) / noise)


def simulate(K: np.ndarray, base: list[int], cand: list[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One-round scores, and whether the confirmed verdict says changed, per random draw."""
    n = K.shape[-1]
    first, final = np.empty(Draws), np.empty(Draws, bool)
    for d in range(Draws):
        if base is cand:
            pick = rng.choice(base, BaselineRuns + 2 * CandidateRuns, replace=False)
            B, C = pick[:BaselineRuns], pick[BaselineRuns:]
        else:
            B = rng.choice(base, BaselineRuns, replace=False)
            C = rng.choice(cand, CandidateRuns, replace=False)
            # Bug sources hold only 3 buggy runs, so the confirmation round redraws from them.
            more = rng.choice([c for c in cand if c not in C], CandidateRuns, replace=False) \
                if len(cand) >= 2 * CandidateRuns else rng.choice(cand, CandidateRuns)
            C = np.r_[C, more]
        M = K[:, :, rng.choice(n, Prompts, replace=False)].mean(-1)
        noise = np.median([M[i, j] for i, j in itertools.combinations(B, 2)])
        if noise == 0:  # the tool stops and asks for fresh baselines: no verdict
            first[d], final[d] = np.nan, False
            continue
        scores = np.array([np.median(M[B, c]) for c in C]) / noise
        first[d] = np.median(scores[:CandidateRuns])
        final[d] = first[d] > Threshold and np.median(scores) > Threshold
    return first, final


def read(path: Path) -> tuple[np.ndarray, list[int], list[int]]:
    z = np.load(path)
    names = list(z["names"])
    K = np.zeros((len(names), len(names), z["K"].shape[-1]), np.float32)
    i, j = np.triu_indices(len(names), 1)
    K[i, j] = K[j, i] = z["K"]
    return K, [k for k, s in enumerate(names) if s.startswith("fixed/")], [k for k, s in enumerate(names) if s.startswith("buggy/")]


def cache(name: str, runs: Path) -> None:
    """Store the per-prompt KL(i || j), i < j, of every pair of runs/{fixed,buggy}/*.npz."""
    names = [f"{side}/{p.stem}" for side in ("fixed", "buggy")
             for p in sorted((runs / side).glob("*.npz"), key=lambda p: int(p.stem))]
    loaded = [load(runs / f"{s}.npz") for s in names]
    K = np.array([[d.kl for d in compare(loaded[i], loaded[j])] for i, j in itertools.combinations(range(len(names)), 2)], np.float32)
    Data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Data / f"{name}.npz", K=K, names=np.array(names))
    print(f"{name}: {len(names)} runs, {K.shape[-1]} prompts")


if __name__ == "__main__":
    main()
