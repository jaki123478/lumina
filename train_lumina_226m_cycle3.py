"""Addestramento continuo (Cycle 3) di consolidamento fine per Lumina 226M.
1000 step loss-masked con cosine learning rate decay per massima stabilità.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from modeling_lumina_next import LuminaNextForCausalLM
from train_lumina_226m_cycle2 import load_all_curated_pairs, sample_batch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent


def main():
    steps = 1000
    lr = 2.5e-5
    batch_size = 8
    grad_accum = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"=== AVVIO ADDESTRAMENTO LUMINA 226M (CYCLE 3) SU {device.type.upper()} ===", flush=True)
    tok = ByteLevelBPETokenizer(
        str(ROOT / "tokenizer_lumina_next/vocab.json"),
        str(ROOT / "tokenizer_lumina_next/merges.txt"),
    )
    ckpt_dir = ROOT / "checkpoints_lumina_226m_candidate"
    model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=device, dtype=dtype)
    print(f"[+] Modello caricato: {model.count_parameters():,} parametri", flush=True)

    pairs = load_all_curated_pairs(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.10)

    model.train()
    model.enable_gradient_checkpointing()
    t0 = time.time()
    acc_loss = 0.0
    opt.zero_grad(set_to_none=True)

    for st in range(1, steps + 1):
        for _ in range(grad_accum):
            x, y = sample_batch(pairs, seq_len=384, batch_size=batch_size, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, model.config["vocab_size"]), y.view(-1), ignore_index=-100) / grad_accum
            loss.backward()
            acc_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if st % 200 == 0 or st == steps:
            cur_lr = sched.get_last_lr()[0]
            avg_loss = acc_loss / (200.0 if st % 200 == 0 else (st % 200))
            elapsed = time.time() - t0
            tok_per_sec = (st * batch_size * grad_accum * 160) / max(0.1, elapsed)
            print(f"[Cycle 3 Step {st:4d}/{steps}] Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | Speed: {tok_per_sec:.0f} tok/s | Tempo: {elapsed:.1f}s", flush=True)
            acc_loss = 0.0

    model.eval()
    model.save_pretrained(ckpt_dir)
    ckpt_file = ckpt_dir / "checkpoint_corpus_sft_v3.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, ckpt_file)

    exp_dir = ROOT / "exports_lumina_226m_candidate"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(exp_dir)
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(exp_dir / "model.safetensors"))
    except Exception:
        pass

    import shutil
    shutil.copy(ROOT / "tokenizer_lumina_next/vocab.json", exp_dir / "vocab.json")
    shutil.copy(ROOT / "tokenizer_lumina_next/merges.txt", exp_dir / "merges.txt")
    print(f"\n[+] Addestramento Cycle 3 completato in {time.time() - t0:.1f}s!", flush=True)
    print(f"[+] Pesi esportati in {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
