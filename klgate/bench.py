"""Replay vLLM fix PRs and identical-code pairs through klgate.kl.

Each entry in bench/entries is a fix PR's buggy parent and fixed commit, or two commits
with identical engine code. `run` scores both builds with seeds 0 to 5 on a GPU and keeps
every pairwise KL in data/bench/<id>.json. With no arguments, this prints the results from
those files: each entry's reference on seeds 0 1 2 against its candidate on seeds 3 4 5,
then every 3-against-3 split of one build's runs.

Usage:
  python -m klgate.bench                                 # CPU, seconds
  python -m klgate.bench run bench/entries/<id>.toml     # GPU
"""

import argparse
import ast
import itertools
import json
import os
import re
import subprocess
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from klgate import build, kl

Root = Path(__file__).resolve().parent.parent
Results = Root / "data" / "bench"
Prompts = Root / "bench" / "gsm8k-5shot-250.jsonl"
Seeds = range(6)
ReferenceSeeds, CandidateSeeds = [0, 1, 2], [3, 4, 5]
GateTolerance = 0.08  # vLLM's GSM8K gate fails when accuracy drops by more than this


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")
    p = sub.add_parser("run", help="score an entry's two builds and write data/bench/<id>.json")
    p.add_argument("entry", type=Path, help="bench/entries/<id>.toml")
    args = ap.parse_args()
    if args.command == "run":
        run(tomllib.loads(args.entry.read_text()))
        return
    results = [json.loads(path.read_text()) for path in sorted(Results.glob("*.json"))]
    if not results:
        raise SystemExit(f"no results in {Results}: run `python -m klgate.bench run bench/entries/<id>.toml`")
    print(table(results))


def run(entry: dict) -> None:
    names = sides(entry)
    pythons = {name: build.install(entry[name]) for name in names}
    # Runs are kept per build, so a build two entries share is scored once.
    paths = {name: [Root / "runs" / entry[name][:12] / f"{seed}.npz" for seed in Seeds] for name in names}
    # Alternating builds spreads any drift in the machine over both.
    for seed in Seeds:
        for name in names:
            if not paths[name][seed].exists():
                score(pythons[name], entry, seed, paths[name][seed])
    tokenizer = Tokenizer.from_pretrained(entry["model"])
    accuracy = partial(gsm8k_accuracy, gold=gold_answers(), tokenizer=tokenizer)
    result = record(entry, {name: [kl.load(str(p)) for p in paths[name]] for name in names}, accuracy)
    Results.mkdir(parents=True, exist_ok=True)
    (Results / f"{entry['id']}.json").write_text(dumps(result))


def sides(entry: dict) -> tuple[str, str]:
    """The names of the entry's reference and candidate builds."""
    return ("fixed", "buggy") if "fixed" in entry else ("baseline", "candidate")


def score(python: Path, entry: dict, seed: int, out: Path) -> None:
    """One run, in a fresh process so every run launches its own engine."""
    cmd = [str(python), "-m", "klgate.kl", "score", "--model", entry["model"], "--prompts", str(Prompts),
           "--seed", str(seed), "--out", str(out), "--engine", json.dumps(entry.get("engine", {})),
           *(f"--env={k}={v}" for k, v in entry.get("env", {}).items())]
    if subprocess.run(cmd, env=os.environ | {"PYTHONPATH": str(Root)}).returncode:
        raise SystemExit(f"scoring {out} failed; rerun the command to resume, finished runs are kept")


def record(entry: dict, runs: dict[str, list[dict]], accuracy: Callable[[dict], float]) -> dict:
    """What data/bench keeps of an entry: its setup, every pairwise KL and each run's accuracy."""
    reference, candidate = sides(entry)
    order = [(name, run) for name in (reference, candidate) for run in runs[name]]
    every = [run for _, run in order]
    kl.check_comparable(every)
    setup = {k: v for k, v in every[0]["meta"].items() if k not in ("vllm", "seed")}
    expected = {"model": entry["model"], "engine": kl.Engine | entry.get("engine", {})}
    if any(setup[k] != v for k, v in expected.items()) or not entry.get("env", {}).items() <= setup["env"].items():
        raise SystemExit(f"the runs for {entry['id']} were scored with another model, engine or env than its "
                         "entry: move them out of runs/ and rerun")
    return {
        "id": entry["id"],
        "builds": {name: entry[name] for name in (reference, candidate)},
        "reference": reference,
        "setup": setup,
        # Each matrix has a row and a column per run, in this order; [i][j] is KL(run i || run j).
        "runs": [{"build": name, "seed": run["meta"]["seed"], "vllm": run["meta"]["vllm"], "accuracy": accuracy(run)}
                 for name, run in order],
        "probe_kl": matrix(kl.probe_kl, every),
        "traffic_kl": matrix(kl.traffic_kl, every),
    }


def matrix(divergence: Callable[[dict, dict], float], runs: list[dict]) -> list[list[float]]:
    return [[0.0 if i == j else divergence(a, b) for j, b in enumerate(runs)] for i, a in enumerate(runs)]


def dumps(result: dict) -> str:
    text = json.dumps(result, indent=2, allow_nan=False)
    # One line per list of numbers, so a matrix reads row by row.
    return re.sub(r"\[\s*([-+.\de,\s]*?)\s*\]", lambda m: "[" + re.sub(r",\s+", ", ", m[1]) + "]", text) + "\n"


def table(results: list[dict]) -> str:
    rows = [("Entry", "Probes", "Traffic", "Verdict", "GSM8K", "vLLM's gate")]
    for r in sorted(results, key=lambda r: (r["reference"] == "baseline", r["id"])):
        reference, candidate = r["reference"], next(name for name in r["builds"] if name != r["reference"])
        refs, cands = indices(r, reference, ReferenceSeeds), indices(r, candidate, CandidateSeeds)
        s = split(r, refs, cands)
        ref_acc, cand_acc = (np.mean([r["runs"][i]["accuracy"] for i in side]) for side in (refs, cands))
        rows.append((r["id"], f"{s['probes']:,.2f}×", f"{s['traffic']:,.2f}×", "changed" if kl.changed(s) else "same",
                     f"{ref_acc:.3f} → {cand_acc:.3f}", "fails" if ref_acc - cand_acc > GateTolerance else "passes"))
    widths = [max(map(len, column)) for column in zip(*rows)]
    lines = ["  ".join(cell.rjust(w) if c in (1, 2) else cell.ljust(w) for c, (cell, w) in enumerate(zip(row, widths)))
             for row in rows]
    return "\n".join([*(line.rstrip() for line in lines), "", null_splits(results)])


def null_splits(results: list[dict]) -> str:
    """Every split of one build's runs into 3 reference and 3 candidate runs."""
    builds = {}  # a build two entries share counts once
    for r in results:
        for name, sha in r["builds"].items():
            builds.setdefault(json.dumps([sha, r["setup"]], sort_keys=True), (r, indices(r, name, list(Seeds))))
    splits = [split(r, list(refs), [i for i in runs if i not in refs])
              for r, runs in builds.values()
              for refs in itertools.combinations(runs, 3)]
    lines = [f"Null splits: {len(splits)} splits of {len(builds)} builds' own runs, 3 against 3"]
    for part, threshold in kl.Thresholds.items():
        values = [s[part] for s in splits]
        over = sum(not v <= threshold for v in values)
        lines.append(f"  {part:8} max {max(values):,.2f}×, {over} over the {threshold}× threshold")
    return "\n".join(lines)


def indices(r: dict, name: str, seeds: list[int]) -> list[int]:
    at = {(run["build"], run["seed"]): i for i, run in enumerate(r["runs"])}
    if missing := [s for s in seeds if (name, s) not in at]:
        raise SystemExit(f"{r['id']} has no {name} run with seed {', '.join(map(str, missing))}: "
                         f"run `python -m klgate.bench run bench/entries/{r['id']}.toml`")
    return [at[name, s] for s in seeds]


def split(r: dict, refs: list[int], cands: list[int]) -> dict[str, float]:
    """kl's scores for the runs at these indices of a result."""
    probe, traffic = r["probe_kl"], r["traffic_kl"]
    return kl.scores(lambda i, j: probe[i][j], lambda i, j: traffic[i][j], refs, cands)


def gold_answers() -> list[int]:
    gold = [answer(json.loads(line)["answer"]) for line in Prompts.read_text().splitlines()]
    if None in gold:
        raise SystemExit(f"{Prompts}: gold answer {gold.index(None) + 1} has no number")
    return gold


def gsm8k_accuracy(run: dict, gold: list[int], tokenizer: Tokenizer) -> float:
    outputs = [tokens[:n].tolist() for tokens, n in zip(run["traffic_tokens"], run["traffic_lengths"], strict=True)]
    texts = tokenizer.decode_batch(outputs)
    return float(np.mean([answer(cut(text)) == g for text, g in zip(texts, gold, strict=True)]))


def cut(text: str) -> str:
    """The text vLLM returns: the tokens keep the stop string, the text drops it."""
    hits = [i for i in (text.find(s) for s in kl.Stop) if i >= 0]
    return text[: min(hits)] if hits else text


def answer(text: str) -> int | None:
    """The last integer in the text, as vLLM's GSM8K gate scores it.

    Like the gate, a number with a leading zero, such as "05", is no answer: literal_eval
    rejects it.
    """
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return None
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return None


if __name__ == "__main__":
    main()
