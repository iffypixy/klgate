"""Score captured runs on GSM8K and test the two builds with an exact paired McNemar test.

Usage: python -m driftgate.gsm8k bench/entries/<id>.toml
"""

import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from driftgate.compare import load
from driftgate.run import Root, load_entry


def main() -> None:
    entry = load_entry(Path(sys.argv[1]))
    gold = [answer(json.loads(line)["answer"]) for line in open(Root / entry["prompts"])]
    tok = Tokenizer.from_pretrained(entry["model"])
    correct = {}
    for side in ("fixed", "buggy"):
        for path in sorted((Root / "runs" / entry["id"] / side).glob("*.npz")):
            run = load(path)
            texts = [tok.decode(run.tokens[i, : run.lengths[i]].tolist()) for i in range(len(gold))]
            correct[(side, path.stem)] = np.array([answer(t) == g for t, g in zip(texts, gold)])
            print(f"{side} {path.stem}  accuracy {correct[(side, path.stem)].mean():.3f}")
    fixed = [v for (s, _), v in correct.items() if s == "fixed"]
    buggy = [v for (s, _), v in correct.items() if s == "buggy"]
    b, c, p = mcnemar(fixed[0], buggy[0])
    print(f"mcnemar fixed 0 vs buggy 0: fixed-only {b}, buggy-only {c}, p = {p:.3f}")


def answer(text: str) -> int | None:
    """The last integer in the text, as vLLM's gate scores it (get_answer_value)."""
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[-1]) if numbers else None



def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired correctness."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return only_a, only_b, p


if __name__ == "__main__":
    main()
