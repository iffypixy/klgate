"""GSM8K scoring exactly as vLLM's gate does it, and an exact paired McNemar test."""

import ast
import json
import re
from math import comb
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from driftgate.compare import Run


def answers(prompts: Path) -> list[int]:
    gold = [answer(json.loads(line)["answer"]) for line in open(prompts)]
    if None in gold:
        raise ValueError(f"{prompts}: gold answer {gold.index(None) + 1} has no number")
    return gold


def correct(run: Run, gold: list[int], tok: Tokenizer, stop: list[str]) -> np.ndarray:
    texts = tok.decode_batch([run.tokens[i, : run.lengths[i]].tolist() for i in range(len(gold))])
    return np.array([answer(cut(t, stop)) == g for t, g in zip(texts, gold)])


def cut(text: str, stop: list[str]) -> str:
    """The text the API returns: the tokens keep the stop string, the text drops it."""
    hits = [i for i in (text.find(s) for s in stop) if i >= 0]
    return text[: min(hits)] if hits else text


def answer(text: str) -> int | None:
    """The last integer in the text, as vLLM's gate scores it (get_answer_value).

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


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value on paired correctness."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = only_a + only_b, min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
