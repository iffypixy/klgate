"""Fake scored runs, built from full logits so every KL has a known source."""

import numpy as np
import pytest

from klgate import kl

Meta = {"vllm": "0.0.0", "model": "m", "engine": kl.Engine, "env": kl.Env, "prompts": "p", "text": "t", "gpu": "g"}


def make_top4(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    top = np.argsort(-lp, axis=-1)[..., :4]
    return lp, top.astype(np.int32), np.take_along_axis(lp, top, axis=-1).astype(np.float32)


def make_scored(probes: np.ndarray, traffic: np.ndarray, seed: int) -> dict:
    """A run from full logits [chunks or prompts, positions, vocab]; greedy picks the argmax."""
    tokens = np.zeros(probes.shape[:2], np.int64)
    lp, ids, top_lp = make_top4(probes)
    _, gen_ids, gen_lp = make_top4(traffic)
    return {
        "probe_tokens": tokens,
        "probe_ids": ids,
        "probe_lp": top_lp,
        "probe_token_lp": np.take_along_axis(lp, tokens[..., None], axis=-1)[..., 0].astype(np.float32),
        "traffic_tokens": gen_ids[..., 0],
        "traffic_ids": gen_ids,
        "traffic_lp": gen_lp,
        "traffic_lengths": np.full(len(traffic), traffic.shape[1], np.int32),
        "meta": Meta | {"seed": seed},
    }


def make_reruns(rng, probes, traffic, seeds, probe_noise=0.01, traffic_noise=0.01) -> list[dict]:
    return [make_scored(probes + probe_noise * rng.normal(size=probes.shape),
                        traffic + traffic_noise * rng.normal(size=traffic.shape), s) for s in seeds]


def make_world(rng) -> tuple[np.ndarray, np.ndarray]:
    """Probe and traffic logits for one model."""
    return rng.normal(size=(8, 16, 32)), rng.normal(size=(8, 6, 32))


@pytest.fixture
def top4():
    return make_top4


@pytest.fixture
def scored():
    return make_scored


@pytest.fixture
def reruns():
    return make_reruns


@pytest.fixture
def world():
    return make_world
