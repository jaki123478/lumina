"""Pretrain Lumina on ~51 MB of Italian literature, Wikipedia and conversation data.

Supports:
- --preset: large (2.75M) or xlarge (14.4M)
- --steps: number of optimization steps (e.g. 20000 or 50000)
- --batch-size: 32 or 64
- --learning-rate: default 5e-4
- automatic GPU acceleration (CUDA)
- periodic checkpoint saving and sample generation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np
import torch
from torch.nn import functional as F

from model import CONFIG, PRESETS, Lumina50K, decode, encode, load_model, prompt_tokens

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_texts"
DIALOGUES_PATH = ROOT / "data" / "dialogues.jsonl"


def prepare_pretraining_tokens(cache_path: Path) -> np.ndarray:
    """Read all literature, wikipedia and dialogue texts and cache as a uint16 token array."""
    if cache_path.exists():
        print(f"Loading cached pretraining tokens from {cache_path}...")
        return np.load(cache_path)

    print("Building pretraining tokens from raw texts and dialogues...")
    all_tokens: list[int] = []

    # 1. Dialogues (interleaved with conversational prefix/suffix)
    if DIALOGUES_PATH.exists():
        for line in DIALOGUES_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tokens = prompt_tokens(rec["user"]) + encode(" " + rec["assistant"]) + [CONFIG["eos_id"]]
            all_tokens.extend(tokens)

    # 2. Wikipedia and literature files
    text_files = sorted(list(RAW_DIR.glob("**/*.txt")))
    print(f"Tokenizing {len(text_files)} text files...")
    for f in text_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            # Normalize whitespace
            content = " ".join(content.split())
            if len(content) < 50:
                continue
            # Chunk long texts with bos and eos tokens
            encoded = [CONFIG["bos_id"]] + encode(content) + [CONFIG["eos_id"]]
            all_tokens.extend(encoded)
        except Exception as e:
            print(f"Warning reading {f.name}: {e}")

    arr = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens in pretraining corpus: {len(arr):,} (~{len(arr)/1_000_000:.2f}M tokens)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


class TokenSampler:
    def __init__(self, tokens: np.ndarray, context_length: int, seed: int = 42):
        self.tokens = tokens
        self.context_length = context_length
        self.max_idx = len(tokens) - context_length - 1
        self.rng = random.Random(seed)

    def sample_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [self.rng.randint(0, self.max_idx) for _ in range(batch_size)]
        x_list, y_list = [], []
        for idx in indices:
            chunk = self.tokens[idx : idx + self.context_length + 1]
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])
        x = torch.tensor(np.array(x_list), dtype=torch.long, device=device)
        y = torch.tensor(np.array(y_list), dtype=torch.long, device=device)
        return x, y


def export_weights(model: Lumina50K, metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format": "lumina-v2",
        "config": model.config,
        "state_dict": {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()},
        "training": metadata,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="xlarge")
    parser.add_argument("--steps", type=int, default=20000, help="Total pretraining steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoint")
    args = parser.parse_args()

    suffix = f"-{args.preset}"
    checkpoint_path = args.checkpoint or (ROOT / f"model{suffix}.pt")
    export_path = args.export or (ROOT / "dist" / "model" / f"weights{suffix}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pretraining preset '{args.preset}' on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    cache_file = ROOT / "data" / "pretrain_tokens.npy"
    tokens = prepare_pretraining_tokens(cache_file)

    val_split = int(len(tokens) * 0.98)
    train_tokens = tokens[:val_split]
    val_tokens = tokens[val_split:]

    config = PRESETS[args.preset]
    ctx_len = config["context_length"]

    train_sampler = TokenSampler(train_tokens, ctx_len, seed=args.seed)
    val_sampler = TokenSampler(val_tokens, ctx_len, seed=args.seed + 1)

    model = Lumina50K(dropout=0.1, config=config).to(device)
    print(f"Model initialized: {config['parameter_count']:,} parameters | Context: {ctx_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def evaluate_loss(n_batches: int = 20) -> float:
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for _ in range(n_batches):
                x, y = val_sampler.sample_batch(args.batch_size, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, config["vocab_size"]), y.reshape(-1))
                total += loss.item()
        return total / n_batches

    start_time = time.monotonic()
    start_step = 0
    best_val = float("inf")
    if args.resume and checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            meta = ckpt.get("training", {})
            start_step = int(meta.get("step", 0))
            best_val = float(meta.get("val_loss", float("inf")))
            print(f"Resumed from {checkpoint_path} at step {start_step} (val_loss: {best_val})")
        except Exception as e:
            print(f"Warning: could not resume from checkpoint: {e}")

    print("\nStarting pretraining loop...")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = train_sampler.sample_batch(args.batch_size, device)

        # Cosine learning rate schedule with warmup
        warmup_steps = 500
        if step < warmup_steps:
            lr = args.learning_rate * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
            lr = args.learning_rate * 0.1 + 0.9 * args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config["vocab_size"]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate_loss(20)
            elapsed = time.monotonic() - start_time
            rate = step / elapsed
            print(json.dumps({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "lr": round(lr, 6),
                "steps_per_sec": round(rate, 2),
                "elapsed_min": round(elapsed / 60, 1)
            }), flush=True)

            if val_loss < best_val:
                best_val = val_loss
                metadata = {
                    "preset": args.preset,
                    "parameters": config["parameter_count"],
                    "step": step,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(val_loss, 4),
                    "elapsed_sec": round(elapsed, 1),
                }
                torch.save({"config": config, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "training": metadata}, checkpoint_path)
                export_weights(model, metadata, export_path)

            # Sample generation
            sample_prompts = ["Ciao", "Come ti chiami?", "Chi è Dante Alighieri?", "Cos'è una rete neurale?"]
            print(f"\n--- Samples at step {step} ---")
            for sp in sample_prompts:
                gen = model.generate(sp, max_new_tokens=100, temperature=0.7)
                print(f"U: {sp}\nL: {gen}\n")
            print("------------------------------\n", flush=True)

    print(f"Pretraining completed. Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
