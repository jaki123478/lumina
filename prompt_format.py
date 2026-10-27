"""Prompt format shared by training, evaluation and (mirrored) the browser worker.

Pure Python, no torch import, so dataset builders can use it too.

Format without memory:  <bos> "U: " question "\nL:"
Format with memory:     <bos> "U: " previous question "\nL: " previous answer "\n" "U: " question "\nL:"

Only the most recent completed exchange is used, and only when the whole prompt
fits HISTORY_BUDGET bytes. The 165-byte context must also hold the answer
(<= 77 bytes + end token), so a larger prompt would make generation roll over.
"""
from __future__ import annotations

BOS_ID = 256
CONTEXT_LENGTH = 165
HISTORY_BUDGET = 84  # includes the <bos> token


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def clean(text: str) -> str:
    return " ".join(text.split())


def build_prompt(message: str, previous: tuple[str, str] | None = None) -> list[int]:
    # Preserve the response delimiter even when a long message is truncated.
    user = encode(message.strip())[-(CONTEXT_LENGTH - 8):]
    base = [BOS_ID] + encode("U: ") + user + encode("\nL:")
    if previous:
        question, answer = clean(previous[0]), clean(previous[1])
        if question and answer:
            history = encode("U: " + question + "\nL: " + answer + "\n")
            if 1 + len(history) + len(base) - 1 <= HISTORY_BUDGET:
                return [BOS_ID] + history + base[1:]
    return base
