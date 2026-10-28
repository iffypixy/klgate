"""GSM8K scoring exactly as vLLM's gate does it, and an exact paired McNemar test."""

import json
import re
from math import comb
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from driftgate.compare import Run


def answers(prompts: Path) -> list[int | None]:
    return [answer(json.loads(line)["answer"]) for line in open(prompts)]


def correct(run: Run, gold: list[int | None], tok: Tokenizer) -> np.ndarray:
    texts = tok.decode_batch([run.tokens[i, : run.lengths[i]].tolist() for i in range(len(gold))])
    return np.array([answer(t) == g for t, g in zip(texts, gold)])


def answer(text: str) -> int | None:
    """The last integer in the text, as vLLM's gate scores it (get_answer_value)."""
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[-1]) if numbers else None


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value on paired correctness."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = only_a + only_b, min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
