"""Compare two captured runs on the positions where both saw the same prefix.

Greedy runs share a prefix until their first differing token, so up to and including
that position the two next-token distributions are conditioned on the same text and can
be compared directly, with no reference model. KL is computed on the ids both top-20
lists hold plus one bucket for the rest, which by the data-processing inequality never
exceeds the true KL.
"""

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
    if stop == 0:
        return PromptDiff(0.0, 0, diverged)
    kls = bucketed_kl(a.top_ids[i, :stop], a.top_lp[i, :stop], b.top_ids[i, :stop], b.top_lp[i, :stop])
    return PromptDiff(float(kls.mean()), int(stop), diverged)


def bucketed_kl(ids_a: np.ndarray, lp_a: np.ndarray, ids_b: np.ndarray, lp_b: np.ndarray) -> np.ndarray:
    """KL(a || b) per position over [positions, k] top-k lists, on their shared ids plus one bucket."""
    match = (ids_a[:, :, None] == ids_b[:, None, :]) & (ids_a[:, :, None] >= 0)  # [S, k, k]
    shared = match.any(axis=2)  # a's ids that b also lists
    pa = np.exp(lp_a.astype(np.float64))
    pb = (match * np.exp(lp_b.astype(np.float64))[:, None, :]).sum(axis=2)  # b's prob of a's id
    pa_s, pb_s = np.where(shared, pa, 0.0), np.where(shared, pb, 1.0)
    tail_a = np.maximum(1 - pa_s.sum(axis=1), TailFloor)
    tail_b = np.maximum(1 - np.where(shared, pb, 0.0).sum(axis=1), TailFloor)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shared, pa_s * (np.log(pa_s) - np.log(pb_s)), 0.0)
    return terms.sum(axis=1) + tail_a * np.log(tail_a / tail_b)
