"""
train_lumina_next.py - Pipeline di Addestramento Long-Context per Lumina-Next (135M).
Caratteristiche:
- Tokenizer BPE (32.000 vocab)
- RoPE + YaRN
- Contesto configurabile (2.048 -> 8.192 -> 32.768 token)
- Mixed Precision bfloat16 su GPU RTX 5060
- Gradient Checkpointing per azzerare il rischio di OOM
"""

import sys
import os
import time
import math
import json
import random
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modeling_lumina_next import LuminaNextForCausalLM, DEFAULT_CONFIG_135M


def load_dataset_tokens(tokenizer: ByteLevelBPETokenizer, dataset_path: str = "data/lumina_train_dataset_v3_cot.jsonl", max_samples: int = 50000) -> list[torch.Tensor]:
    """Carica i dati di coding e conversazione tokenizzati in tensori ID."""
    print("[*] Caricamento e tokenizzazione del dataset con BPE (32k vocab)...")
    tokenized_samples = []

    ds_file = Path(dataset_path)
    if not ds_file.exists():
        ds_file = Path("data/lumina_train_dataset_v2.jsonl")
    if not ds_file.exists():
        ds_file = Path("data/distilled_dialogues_clean.jsonl")
    if not ds_file.exists():
        ds_file = Path("data/distilled_dialogues.jsonl")
    print(f"[*] File dataset utilizzato: {ds_file}")

    for path in [Path("data/lumina_coding_curriculum.jsonl"), ds_file]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = eval(line) if not line.strip().startswith("{") else json.loads(line)
                    u = d.get("user", "").strip()
                    a = d.get("assistant", "").strip()
                    if u and a:
                        text = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
                        ids = tokenizer.encode(text).ids
                        if len(ids) >= 8:
                            tokenized_samples.append(torch.tensor(ids, dtype=torch.long))
                            if len(tokenized_samples) >= max_samples:
                                break
                except Exception:
                    continue

    print(f"[+] Caricati {len(tokenized_samples):,} campioni tokenizzati con successo.")
    return tokenized_samples


def get_batch(samples: list[torch.Tensor], seq_len: int, batch_size: int, device: torch.device):
    """Estrae un batch di sequenze concatenate di lunghezza seq_len esatta."""
    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        # Concatena campioni fino a raggiungere seq_len + 1
        seq = []
        while len(seq) < seq_len + 1:
            sample = random.choice(samples)
            seq.extend(sample.tolist())
        
        seq = seq[:seq_len + 1]
        t = torch.tensor(seq, dtype=torch.long)
        batch_x.append(t[:-1])
        batch_y.append(t[1:])

    x = torch.stack(batch_x).to(device)
    y = torch.stack(batch_y).to(device)
    return x, y


def train(
    seq_len: int = 2048,
    steps: int = 500,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    lr: float = 4e-4,
    use_checkpointing: bool = True,
):
    print("=" * 75)
    print("      🚀 LUMINA-NEXT 135M: ADDESTRAMENTO LONG-CONTEXT ACCELERATO")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print(f"[*] Hardware       : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"[*] Precisione     : {dtype}")
    print(f"[*] Finestra Seq   : {seq_len:,} token per sequenza")
    print(f"[*] Batch Effettivo: {batch_size * grad_accum_steps} sequenze ({batch_size * grad_accum_steps * seq_len:,} token/step)")
    print(f"[*] Gradient Checkp: {'Attivo (Anti-OOM)' if use_checkpointing else 'Disattivato'}")
    print("=" * 75)

    # 1. Carica Tokenizer BPE
    tok_dir = Path("tokenizer_lumina_next")
    if not tok_dir.exists():
        print("[!] Tokenizer non trovato. Addestramento in corso...")
        import train_bpe_tokenizer
        train_bpe_tokenizer.train_tokenizer()

    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    out_dir = Path(os.environ.get("LUMINA_CHECKPOINT_DIR", "checkpoints_lumina_next"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Inizializza o riprendi Modello Lumina-Next
    config = dict(DEFAULT_CONFIG_135M)
    config["max_position_embeddings"] = max(8192, seq_len)
    if (out_dir / "pytorch_model.bin").exists():
        print(f"[*] Ripresa dell'addestramento dal checkpoint esistente in '{out_dir}'...")
        model = LuminaNextForCausalLM.from_pretrained(out_dir).to(device=device, dtype=dtype)
    else:
        model = LuminaNextForCausalLM(config).to(device=device, dtype=dtype)
    
    if use_checkpointing:
        model.enable_gradient_checkpointing()

    total_params = model.count_parameters()
    print(f"[+] Modello inizializzato: {total_params:,} parametri ({total_params/1e6:.1f}M)")

    # 3. Carica Dati
    dataset = load_dataset_tokens(tokenizer)
    if not dataset:
        print("[!] Nessun dato trovato in data/*.jsonl. Creazione campioni base...")
        dataset = [torch.randint(0, 32000, (seq_len + 1,)) for _ in range(20)]

    # 4. Ottimizzatore
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # 5. Loop di Addestramento
    print(f"\n[*] Avvio {steps} step di addestramento su GPU...")
    print("-" * 75)
    model.train()
    start_time = time.time()
    accum_loss = 0.0
    loss_history = []

    for step in range(1, steps + 1):
        x, y = get_batch(dataset, seq_len=seq_len, batch_size=batch_size, device=device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1))
            loss = loss / grad_accum_steps

        loss.backward()
        accum_loss += loss.item() * grad_accum_steps

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0 or step == steps:
            cur_loss = accum_loss / 25 if step % 25 == 0 else accum_loss
            elapsed = time.time() - start_time
            tok_per_sec = (step * batch_size * seq_len) / elapsed
            vram = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"[Step {step:4d}/{steps}] Loss: {cur_loss:.4f} | Throughput: {tok_per_sec:,.0f} tok/s | VRAM: {vram:.1f} MB | Tempo: {elapsed:.1f}s", flush=True)
            loss_history.append({"step": step, "loss": round(cur_loss, 4), "vram_mb": round(vram, 1), "tok_per_sec": round(tok_per_sec, 0), "elapsed_s": round(elapsed, 1)})
            accum_loss = 0.0

    print("\n" + "=" * 75)
    print("             🎉 ADDESTRAMENTO LUMINA-NEXT COMPLETATO!")
    print("=" * 75)

    # 6. Salva Checkpoint e aggiorna pacchetto HF
    prev_steps = 0
    try:
        prev_data = json.loads(Path("data/training_history_cycle1.json").read_text(encoding="utf-8"))
        prev_steps = prev_data.get("cumulative_steps", 0)
    except Exception:
        pass
    cum_steps = prev_steps + steps

    model.save_pretrained(out_dir)
    versioned_ckpt = out_dir / f"checkpoint_step{cum_steps}.pt"
    torch.save({"state_dict": model.state_dict(), "steps": cum_steps, "config": config, "loss_history": loss_history}, versioned_ckpt)
    print(f"[+] Pesi, configurazione e checkpoint versionato salvati in '{versioned_ckpt}'!")

    hf_dir = Path(os.environ.get("LUMINA_EXPORT_DIR", "hf_export_lumina_next"))
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_dir)
    try:
        from safetensors.torch import save_model
        save_model(model, str(hf_dir / "model.safetensors"))
        print(f"[+] Model safetensors aggiornato in '{hf_dir / 'model.safetensors'}'!")
    except Exception as e:
        print(f"[!] Nota export safetensors: {e}")

    # Salva storico metriche di addestramento
    for h_path in [Path("data/training_history_cycle7.json"), Path("data/training_history_cycle1.json")]:
        with open(h_path, "w", encoding="utf-8") as f:
            json.dump({
                "cycle": 7,
                "cumulative_steps": cum_steps,
                "steps": steps,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "lr": lr,
                "initial_loss": loss_history[0]["loss"] if loss_history else None,
                "final_loss": loss_history[-1]["loss"] if loss_history else None,
                "history": loss_history
            }, f, indent=2)
    print(f"[+] Storico metriche aggiornato per il monitoraggio GitHub (Step cumulativi: {cum_steps})!")

    # 7. Test Generativo Finale
    test_prompt = "<|im_start|>user\nCome creo una funzione somma in Python?<|im_end|>\n<|im_start|>assistant\n"
    test_ids = torch.tensor([tokenizer.encode(test_prompt).ids], device=device)
    print(f"\n[*] Test Generativo (Prompt: 'Come creo una funzione somma in Python?'):")
    gen_out = model.generate(test_ids, max_new_tokens=40, temperature=0.3)
    gen_text = tokenizer.decode(gen_out[0].tolist())
    print(gen_text)
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048, help="Lunghezza sequenza di addestramento (es. 2048 o 4096)")
    parser.add_argument("--steps", type=int, default=500, help="Numero di step")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per step")
    args = parser.parse_args()

    train(seq_len=args.seq_len, steps=args.steps, batch_size=args.batch_size)
