"""Train Lumina 50K from random weights and export a browser-readable model.

Example: python train.py --steps 6000 --data data/dialogues.jsonl
Each JSONL record has user/assistant strings and a split ('train' or 'validation').
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import CONFIG, PRESETS, Lumina50K, encode, load_model, prompt_tokens


ROOT = Path(__file__).resolve().parent


def read_examples(path: Path, config: dict = CONFIG) -> tuple[list, list]:
    train, validation = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        previous = record.get("previous")
        prefix = prompt_tokens(record["user"], (previous["user"], previous["assistant"]) if previous else None)
        answer = encode(" " + record["assistant"]) + [config["eos_id"]]
        # Training examples retain the answer within the context window; truncate gracefully if too long
        if len(answer) >= config["context_length"] - 8:
            answer = answer[:config["context_length"] - 16] + [config["eos_id"]]
        max_prefix = config["context_length"] + 1 - len(answer)
        if len(prefix) > max_prefix:
            prefix = [config["bos_id"]] + prefix[-(max_prefix - 1):]
        ids = prefix + answer
        x, y = ids[:-1], ids[1:]
        y[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
        item = (x, y)
        (validation if record.get("split") == "validation" else train).append(item)
    if not train or not validation:
        raise ValueError("Dataset must contain train and validation records")
    return train, validation


def batch(examples: list, batch_size: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = rng.choices(examples, k=batch_size)
    length = max(len(x) for x, _ in sampled)
    x = torch.full((batch_size, length), CONFIG["pad_id"], dtype=torch.long)
    y = torch.full((batch_size, length), -100, dtype=torch.long)
    for i, (inputs, labels) in enumerate(sampled):
        x[i, :len(inputs)] = torch.tensor(inputs)
        y[i, :len(labels)] = torch.tensor(labels)
    return x, y


@torch.inference_mode()
def evaluate(model: Lumina50K, examples: list) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(examples), 24):
        group = examples[start:start + 24]
        length = max(len(x) for x, _ in group)
        x = torch.full((len(group), length), CONFIG["pad_id"], dtype=torch.long)
        y = torch.full((len(group), length), -100, dtype=torch.long)
        for i, (inputs, labels) in enumerate(group):
            x[i, :len(inputs)] = torch.tensor(inputs)
            y[i, :len(labels)] = torch.tensor(labels)
        x, y = x.to(device), y.to(device)
        total_loss += F.cross_entropy(model(x).reshape(-1, CONFIG["vocab_size"]), y.reshape(-1), ignore_index=-100, reduction="sum").item()
        total_tokens += (y != -100).sum().item()
    return total_loss / total_tokens


def export(model: Lumina50K, metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": "lumina-50k-v1",
        "config": model.config,
        "state_dict": {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()},
        "training": metadata,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000, help="Additional optimization steps")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dialogues.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="50k", help="Model size; 'large' is meant for a CUDA GPU")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: model.pt (50k) or model-<preset>.pt")
    parser.add_argument("--export", type=Path, default=None, help="Default: dist/model/weights.json (50k) or weights-<preset>.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.1, help="Training-time dropout (no extra parameters)")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()
    suffix = "" if args.preset == "50k" else "-" + args.preset
    args.checkpoint = args.checkpoint or ROOT / f"model{suffix}.pt"
    args.export = args.export or ROOT / "dist" / "model" / f"weights{suffix}.json"
    if args.steps < 0 or args.batch_size < 1 or args.threads < 1 or args.eval_every < 1:
        parser.error("steps must be nonnegative; batch size, threads and eval interval must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    config = PRESETS[args.preset]
    train, validation = read_examples(args.data, config)
    previous = {}
    if args.resume:
        model, previous = load_model(args.checkpoint, args.dropout)
    else:
        model = Lumina50K(args.dropout, PRESETS[args.preset])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(json.dumps({"preset": args.preset, "device": device, "parameter_count": model.config["parameter_count"]}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = int(previous.get("steps", 0))
    started = time.monotonic()
    initial_train = evaluate(model, train)
    initial_val = evaluate(model, validation)
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "train_examples": len(train), "validation_examples": len(validation), "initial_train_loss": initial_train, "initial_validation_loss": initial_val}), flush=True)

    def save(step: int, train_loss: float, val_loss: float) -> None:
        metadata = {
            "origin": "random initialization; original Italian teaching dialogues",
            "seed": args.seed,
            "steps": start_step + step,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_loss": round(train_loss, 6),
            "validation_loss": round(val_loss, 6),
            "initial_train_loss": previous.get("initial_train_loss", round(initial_train, 6)),
            "initial_validation_loss": previous.get("initial_validation_loss", round(initial_val, 6)),
            "elapsed_seconds": round(float(previous.get("elapsed_seconds", 0)) + time.monotonic() - started, 1),
            "corpus_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "note": "Educational miniature; validation shares topics with training and is not a general intelligence benchmark.",
        }
        checkpoint = {"config": model.config, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "training": metadata}
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.checkpoint.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(args.checkpoint)
        export(model, metadata, args.export)
        print(json.dumps({"step": start_step + step, "train_loss": train_loss, "validation_loss": val_loss, "seconds": round(time.monotonic() - started, 1)}), flush=True)

    # Only checkpoints that improve validation loss are saved, so later overfitting never overwrites them.
    best_val = initial_val
    save(0, initial_train, initial_val)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(train, args.batch_size, rng)
        x, y = x.to(device), y.to(device)
        progress = step / max(args.steps, 1)
        lr = args.learning_rate * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, CONFIG["vocab_size"]), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; checkpoint was preserved")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            train_loss, val_loss = evaluate(model, train), evaluate(model, validation)
            if val_loss < best_val:
                best_val = val_loss
                save(step, train_loss, val_loss)
            else:
                print(json.dumps({"step": start_step + step, "train_loss": train_loss, "validation_loss": val_loss, "saved": False}), flush=True)
    # Generate samples from the best saved checkpoint rather than the last step.
    model, _ = load_model(args.checkpoint)  # best checkpoint, on CPU
    print("Browser weights: " + str(args.export), flush=True)
    for question in ["Ciao", "Come ti chiami?", "Quanto fa due più due?", "Che cos'è un modello linguistico?", "Raccontami una storia"]:
        print(json.dumps({"user": question, "assistant": model.generate(question)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
