import numpy as np
import pytest

from noisefloor.compare import Run
from noisefloor.diff import judge

Prompts, Vocab = 200, 20
rng = np.random.default_rng(0)
Logits = rng.normal(0, 3, (Prompts, 1, Vocab))  # one position per prompt, the same text in every run
Shift = rng.normal(0, 1, (Prompts, 1, Vocab))


def run(noise: float, shift: float = 0.0) -> Run:
    logits = Logits + shift * Shift + rng.normal(0, noise, Logits.shape)
    order = np.argsort(-logits, axis=-1)
    lp = np.take_along_axis(logits - np.log(np.exp(logits).sum(-1, keepdims=True)), order, -1)
    return Run(np.zeros((Prompts, 1), int), order, lp.astype(np.float32), np.ones(Prompts, int))


def test_same_build_reads_same():
    assert not judge([run(0.05) for _ in range(5)], [run(0.05) for _ in range(3)]).changed


def test_a_shift_in_every_candidate_reads_changed():
    assert judge([run(0.05) for _ in range(5)], [run(0.05, shift=0.05) for _ in range(3)]).changed


def test_one_off_launch_cannot_flip_the_verdict():
    cand = [run(0.05, shift=0.2), run(0.05), run(0.05)]
    assert not judge([run(0.05) for _ in range(5)], cand).changed


def test_bit_identical_baselines_stop_with_advice():
    base = run(0.05)
    with pytest.raises(SystemExit, match="bit-identical"):
        judge([base] * 5, [run(0.05) for _ in range(3)])
