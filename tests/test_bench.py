import json

import numpy as np

from klgate import bench, kl


def result(entry: dict, runs: dict[str, list[dict]]) -> dict:
    """An entry's data/bench file, as written and read back."""
    return json.loads(bench.dumps(bench.record(entry | {"model": "m"}, runs, accuracy=lambda run: 0.5)))


def test_the_table_scores_as_judge_does(world, reruns):
    rng = np.random.default_rng(0)
    probes, traffic = world(rng)
    shifted = probes + 0.05 * rng.normal(size=probes.shape)
    fixed, buggy = reruns(rng, probes, traffic, range(6)), reruns(rng, shifted, traffic, range(6))
    table = bench.table([result({"id": "vllm-1", "buggy": "b" * 40, "fixed": "f" * 40}, {"fixed": fixed, "buggy": buggy})])
    s = kl.judge(fixed[:3], buggy[3:])["scores"]
    row = next(line for line in table.splitlines() if line.startswith("vllm-1"))
    assert row.split()[1:4] == [f"{s['probes']:,.2f}×", f"{s['traffic']:,.2f}×", "changed"]


def test_null_splits_count_a_shared_build_once(world, reruns):
    rng = np.random.default_rng(1)
    probes, traffic = world(rng)
    a, b, c = (reruns(rng, probes, traffic, range(6)) for _ in range(3))
    results = [result({"id": "vllm-1", "buggy": "a" * 40, "fixed": "b" * 40}, {"buggy": a, "fixed": b}),
               result({"id": "null-1", "baseline": "b" * 40, "candidate": "c" * 40}, {"baseline": b, "candidate": c})]
    assert "60 splits of 3 builds" in bench.table(results)


def test_answers_are_read_as_vllm_s_gate_reads_them():
    assert bench.answer(bench.cut("So she has 1,234 apples.\n\nQuestion: what is 5 + 7?")) == 1234
    assert bench.answer("The answer is 05") is None
    assert bench.answer("no number") is None
