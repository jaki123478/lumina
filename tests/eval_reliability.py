"""Reliability evaluation for Lumina 50K: exact-match accuracy with greedy decoding.

Groups:
  held_out_in_scope   validation topic rows (unseen wording of topics in the corpus)
  held_out_definition_frames  definitions asked with sentence frames never used in training
  held_out_arithmetic arithmetic facts (0..10) whose exact pair was never trained
  held_out_refusal    validation out-of-scope questions (different from training ones)
  held_out_memory     validation 'repeat' requests with an earlier exchange
  probes_*            a few hand-written questions that appear nowhere in the corpus

False refusal = a held-out in-scope question answered with the refusal.
Usage: python tests/eval_reliability.py [--checkpoint model.pt] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import torch

from model import load_model
from out_of_scope import REFUSAL

PROBES_IN_SCOPE = [
    ("Hey, chi sei?", "Come ti chiami?"),
    ("Mi dici come ti chiami?", "Come ti chiami?"),
    ("Spiegami cos'è Python", "Cos'è Python?"),
    ("Quanto fa 2 più 2?", "Quanto fa due più due?"),
    ("Qual è la capitale di Italia?", "Qual è la capitale d'Italia?"),
    ("Cos'è il linguaggio JavaScript?", "Cos'è JavaScript?"),
    ("Mi senti triste oggi", "Sono triste"),
]
PROBES_REFUSAL = [
    "Qual è la capitale della Grecia?", "Chi ha scritto Il Nome della Rosa?", "Quanto fa 13 + 14?",
    "Traduci cane in inglese", "Scrivi un programma in Rust", "Quanto pesa un elefante?",
]
PROBES_MEMORY = [
    (("Come ti chiami?", None), "Ripeti"),
    (("Cos'è Python?", None), "Puoi ripetere per favore?"),
    (("Sei ChatGPT?", None), "Scusa, come hai detto?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dialogues.jsonl")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    answers = {row["user"]: row["assistant"] for row in rows if "previous" not in row}

    def ask(message, previous=None):
        return model.generate(message, previous=previous)

    groups: dict[str, list] = {"held_out_in_scope": [], "held_out_definition_frames": [], "held_out_arithmetic": [],
                               "held_out_refusal": [], "held_out_memory": [],
                               "probes_in_scope": [], "probes_refusal": [], "probes_memory": []}
    kind_groups = {"topic": "held_out_in_scope", "definition": "held_out_definition_frames", "arithmetic": "held_out_arithmetic",
                   "refusal": "held_out_refusal", "memory": "held_out_memory"}
    for row in rows:
        if row["split"] != "validation":
            continue
        previous = (row["previous"]["user"], row["previous"]["assistant"]) if "previous" in row else None
        got = ask(row["user"], previous)
        key = kind_groups.get(row.get("kind"), "held_out_in_scope")
        groups[key].append({"user": row["user"], "expected": row["assistant"], "got": got, "ok": got == row["assistant"]})
    for message, source in PROBES_IN_SCOPE:
        expected = answers[source]
        got = ask(message)
        groups["probes_in_scope"].append({"user": message, "expected": expected, "got": got, "ok": got == expected})
    for message in PROBES_REFUSAL:
        got = ask(message)
        groups["probes_refusal"].append({"user": message, "expected": REFUSAL, "got": got, "ok": got == REFUSAL})
    for (question, _), follow_up in PROBES_MEMORY:
        previous = (question, answers[question])
        got = ask(follow_up, previous)
        groups["probes_memory"].append({"user": follow_up, "previous": question, "expected": previous[1], "got": got, "ok": got == previous[1]})

    summary = {name: {"n": len(items), "accuracy": round(sum(i["ok"] for i in items) / max(len(items), 1), 3)} for name, items in groups.items()}
    scoped = groups["held_out_in_scope"] + groups["held_out_definition_frames"] + groups["held_out_arithmetic"] + groups["probes_in_scope"]
    summary["false_refusal_rate"] = round(sum(i["got"] == REFUSAL for i in scoped) / max(len(scoped), 1), 3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failures = [dict(group=name, **item) for name, items in groups.items() for item in items if not item["ok"]]
    for failure in failures:
        print(json.dumps(failure, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
