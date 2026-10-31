import json
from types import SimpleNamespace

import numpy as np
import pytest

from klgate import kl

def test_reruns_of_one_build_read_as_noise(world, reruns):
    rng = np.random.default_rng(0)
    probes, traffic = world(rng)
    refs, cands = reruns(rng, probes, traffic, [0, 1, 2]), reruns(rng, probes, traffic, [3, 4, 5])
    result = kl.judge(refs, cands)
    assert not result["changed"]
    assert max(result["scores"].values()) < 1.5
    assert kl.judge(refs[::-1], cands[::-1])["scores"] == pytest.approx(result["scores"], rel=1e-12)


def test_a_consistent_shift_is_flagged_by_the_probes(world, reruns):
    rng = np.random.default_rng(1)
    probes, traffic = world(rng)
    shifted = probes + 0.1 * rng.normal(size=probes.shape)
    result = kl.judge(reruns(rng, probes, traffic, [0, 1, 2]), reruns(rng, shifted, traffic, [3, 4, 5]))
    assert result["changed"]
    assert result["scores"]["probes"] > 10
    assert result["stats"]["same_top"] < 1


def test_reruns_that_disagree_are_flagged_by_the_traffic(world, reruns):
    # Random KV corruption: every candidate run goes wrong in its own way.
    rng = np.random.default_rng(2)
    probes, traffic = world(rng)
    result = kl.judge(reruns(rng, probes, traffic, [0, 1, 2]), reruns(rng, probes, traffic, [3, 4, 5], traffic_noise=0.3))
    assert result["changed"]
    assert result["scores"]["traffic"] > 3
    assert result["scores"]["probes"] < 2


def test_identical_runs_score_zero(world, scored):
    probes, traffic = world(np.random.default_rng(3))
    result = kl.judge([scored(probes, traffic, s) for s in (0, 1)], [scored(probes, traffic, s) for s in (2, 3)])
    assert result["scores"] == {"probes": 0.0, "traffic": 0.0}
    assert not result["changed"]
    assert result["stats"]["kl_max"] == 0
    assert result["stats"]["traffic_differ"] == 0


def test_probe_noise_is_pooled_from_both_builds(world, scored, reruns):
    # Bit-identical reference reruns leave the candidate's own rerun noise to scale by.
    rng = np.random.default_rng(10)
    probes, traffic = world(rng)
    refs = [scored(probes, traffic, s) for s in (0, 1, 2)]
    result = kl.judge(refs, reruns(rng, probes, traffic, [3, 4, 5]))
    assert result["scores"]["probes"] < 2


@pytest.mark.parametrize("side", [0, 1])
def test_a_nan_in_either_build_reads_changed(world, reruns, side):
    rng = np.random.default_rng(4)
    probes, traffic = world(rng)
    sides = reruns(rng, probes, traffic, [0, 1, 2]), reruns(rng, probes, traffic, [3, 4, 5])
    sides[side][0]["probe_lp"][0, 0, 0] = np.nan
    assert kl.judge(*sides)["changed"]


def test_any_difference_from_bit_identical_reruns_reads_changed(world, scored):
    rng = np.random.default_rng(5)
    probes, traffic = world(rng)
    shifted = probes + 1e-3 * rng.normal(size=probes.shape)
    result = kl.judge([scored(probes, traffic, s) for s in (0, 1)], [scored(shifted, traffic, s) for s in (2, 3)])
    assert result["scores"]["probes"] == np.inf
    assert result["changed"]
    assert json.loads(json.dumps(kl.jsonable(result), allow_nan=False))["scores"]["probes"] == "inf"


def test_bucketed_kl_is_exact_when_k_covers_the_vocab_and_a_lower_bound_when_not():
    rng = np.random.default_rng(6)
    lp_a, lp_b = np.log(rng.dirichlet(np.ones(6), size=(2, 5))), np.log(rng.dirichlet(np.ones(6), size=(2, 5)))
    exact = (np.exp(lp_a) * (lp_a - lp_b)).sum(axis=-1)
    ids = np.broadcast_to(np.arange(6), lp_a.shape)
    np.testing.assert_allclose(kl.bucketed_kl(ids, lp_a, ids, lp_b), exact, rtol=1e-6)
    top_a, top_b = np.argsort(-lp_a, axis=-1)[..., :3], np.argsort(-lp_b, axis=-1)[..., :3]
    bucketed = kl.bucketed_kl(top_a, np.take_along_axis(lp_a, top_a, -1), top_b, np.take_along_axis(lp_b, top_b, -1))
    assert np.all(bucketed <= exact + 1e-12)


def test_an_id_a_gives_no_mass_adds_no_kl_but_a_nan_spreads():
    ids, lp_b = np.array([[0, 1, 2]]), np.log([[0.5, 0.3, 0.2]])
    with_zero = kl.bucketed_kl(ids, np.array([[np.log(0.6), np.log(0.4), -np.inf]]), ids, lp_b)
    assert with_zero[0] == pytest.approx(0.6 * np.log(0.6 / 0.5) + 0.4 * np.log(0.4 / 0.3))
    assert np.isnan(kl.bucketed_kl(ids, np.array([[np.log(0.6), np.nan, -np.inf]]), ids, lp_b)[0])


def test_traffic_kl_matches_a_prompt_by_prompt_loop(top4):
    rng = np.random.default_rng(7)
    prompts, length, vocab = 40, 12, 16

    def output() -> dict:
        _, ids, lp = top4(rng.normal(size=(prompts, length, vocab)))
        lengths = rng.integers(0, length + 1, size=prompts)
        tokens = np.where(np.arange(length) < lengths[:, None], rng.integers(0, 3, size=(prompts, length)), -1)
        return {"traffic_tokens": tokens, "traffic_ids": ids, "traffic_lp": lp, "traffic_lengths": lengths}

    a, b = output(), output()
    b["traffic_tokens"][0], b["traffic_lengths"][0] = a["traffic_tokens"][0], a["traffic_lengths"][0]
    b["traffic_tokens"][1, 5:], b["traffic_lengths"][1] = -1, min(a["traffic_lengths"][1], 5)
    b["traffic_tokens"][1, :5] = a["traffic_tokens"][1, :5]
    a["traffic_tokens"][2], a["traffic_lengths"][2] = -1, 0
    b["traffic_tokens"][3], b["traffic_lengths"][3] = -1, 0

    def slow(a: dict, b: dict) -> float:
        kls = []
        for i in range(prompts):
            n = min(a["traffic_lengths"][i], b["traffic_lengths"][i])
            differ = np.nonzero(a["traffic_tokens"][i, :n] != b["traffic_tokens"][i, :n])[0]
            stop = differ[0] + 1 if len(differ) else n
            kls.append(kl.bucketed_kl(a["traffic_ids"][i, :stop], a["traffic_lp"][i, :stop],
                                      b["traffic_ids"][i, :stop], b["traffic_lp"][i, :stop]).mean() if stop else 0.0)
        return float(np.mean(kls))

    assert kl.traffic_kl(a, b) == pytest.approx(slow(a, b), rel=1e-12)
    assert kl.traffic_kl(b, a) == pytest.approx(slow(b, a), rel=1e-12)


def test_put_keeps_the_best_k():
    logprobs = {tid: SimpleNamespace(logprob=-float(tid)) for tid in reversed(range(kl.TopK + 1))}
    ids, lps = np.full(kl.TopK, -1), np.full(kl.TopK, -np.inf)
    kl.put(ids, lps, logprobs)
    assert ids.tolist() == list(range(kl.TopK))
    assert lps.tolist() == [-float(t) for t in range(kl.TopK)]


def test_shuffled_returns_outputs_in_input_order():
    items = list(range(50))
    asked = []
    outputs = kl.shuffled(lambda xs: asked.extend(xs) or [x * 10 for x in xs], items, np.random.default_rng(8))
    assert outputs == [x * 10 for x in items]
    assert asked != items


def test_judge_rejects_runs_it_cannot_compare(world, reruns):
    rng = np.random.default_rng(9)
    probes, traffic = world(rng)
    refs, (c0, c1) = reruns(rng, probes, traffic, [0, 1]), reruns(rng, probes, traffic, [2, 3])

    def with_meta(run: dict, **changes) -> dict:
        return run | {"meta": run["meta"] | changes}

    with pytest.raises(SystemExit, match="at least 2 runs"):
        kl.judge(refs, [c0])
    with pytest.raises(SystemExit, match="repeat seed 1"):
        kl.judge(refs, [c0, refs[1]])
    with pytest.raises(SystemExit, match="candidate runs come from vLLM 0.0.0, 0.0.1"):
        kl.judge(refs, [c0, with_meta(c1, vllm="0.0.1")])
    with pytest.raises(SystemExit, match="differ in gpu"):
        kl.judge(refs, [c0, with_meta(c1, gpu="other")])
