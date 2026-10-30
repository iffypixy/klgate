import numpy as np

from klgate.compare import bucketed_kl
from klgate.kl import compare

Rng = np.random.default_rng(0)


def top4(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    top = np.argsort(-lp, axis=-1)[..., :4]
    return lp, top.astype(np.int32), np.take_along_axis(lp, top, axis=-1).astype(np.float32)


def run(probes: np.ndarray, traffic: np.ndarray, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """A scored run from full logits [chunks or prompts, positions, vocab]; greedy picks the argmax."""
    lp, ids, top_lp = top4(probes)
    _, gen_ids, gen_lp = top4(traffic)
    return {
        "tokens": tokens,
        "top_ids": ids,
        "top_lp": top_lp,
        "token_lp": np.take_along_axis(lp, tokens[..., None], axis=-1)[..., 0].astype(np.float32),
        "gen_tokens": gen_ids[..., 0],
        "gen_top_ids": gen_ids,
        "gen_top_lp": gen_lp,
        "gen_lengths": np.full(len(traffic), traffic.shape[1], np.int32),
    }


def noisy(probes, traffic, tokens, probe_noise, traffic_noise):
    return run(probes + probe_noise * Rng.normal(size=probes.shape),
               traffic + traffic_noise * Rng.normal(size=traffic.shape), tokens)


Probes, Traffic = Rng.normal(size=(8, 16, 32)), Rng.normal(size=(8, 6, 32))
Tokens = Rng.integers(0, 32, size=(8, 16))


def test_bucketed_kl_is_exact_when_k_covers_the_vocab():
    lp_a = np.log(Rng.dirichlet(np.ones(6), size=5))
    lp_b = np.log(Rng.dirichlet(np.ones(6), size=5))
    ids = np.tile(np.arange(6), (5, 1))
    exact = (np.exp(lp_a) * (lp_a - lp_b)).sum(axis=1)
    np.testing.assert_allclose(bucketed_kl(ids, lp_a, ids, lp_b), exact, rtol=1e-6)


def test_reruns_of_one_build_read_as_noise():
    refs = [noisy(Probes, Traffic, Tokens, 0.01, 0.01) for _ in range(3)]
    result = compare(refs, [noisy(Probes, Traffic, Tokens, 0.01, 0.01) for _ in range(3)])
    assert not result["changed"]
    assert max(result["scores"].values()) < 1.5


def test_a_consistent_shift_is_flagged_by_the_probes():
    refs = [noisy(Probes, Traffic, Tokens, 0.01, 0.01) for _ in range(3)]
    shifted = Probes + 0.1 * Rng.normal(size=Probes.shape)
    result = compare(refs, [noisy(shifted, Traffic, Tokens, 0.01, 0.01) for _ in range(3)])
    assert result["changed"]
    assert result["scores"]["probes"] > 10
    assert result["stats"]["same_top"] < 1


def test_reruns_that_disagree_are_flagged_by_the_traffic():
    # Random KV corruption: every candidate run goes wrong in its own way.
    refs = [noisy(Probes, Traffic, Tokens, 0.01, 0.01) for _ in range(3)]
    result = compare(refs, [noisy(Probes, Traffic, Tokens, 0.01, 0.3) for _ in range(3)])
    assert result["changed"]
    assert result["scores"]["traffic"] > 3
    assert result["scores"]["probes"] < 2


def test_identical_runs_have_no_divergence():
    r = run(Probes, Traffic, Tokens)
    result = compare([r, r], [r, r])
    assert result["scores"] == {"probes": 0.0, "traffic": 0.0}
    assert not result["changed"]
    assert result["stats"]["kl_max"] == 0
    assert result["stats"]["traffic_diverged"] == 0
