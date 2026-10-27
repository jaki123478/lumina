"""
Addestratore Intensivo di Coding per Lumina AI.
Addestra direttamente i pesi neurali di Lumina sul curriculum di programmazione Python,
algoritmi, gestione errori e Git sulla GPU RTX 5060.
Mostra in tempo reale il codice generato da Lumina durante l'apprendimento!
"""

import sys
import os
import time
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model import CONFIG, Lumina50K, load_model, prompt_tokens, encode
from train import read_examples, batch, evaluate, export


TEST_PROMPTS = [
    "Come filtro i numeri pari da una lista in Python?",
    "Come creo una funzione che somma due numeri in Python?",
    "Cosa significa IndexError in Python e come si risolve?",
    "Come inverto una stringa in Python?",
]


def generate_live(model: Lumina50K, message: str, max_tokens: int = 50) -> str:
    """Genera una risposta live direttamente dal modello Lumina."""
    device = next(model.parameters()).device
    model.eval()
    tokens = prompt_tokens(message, None)
    with torch.no_grad():
        for _ in range(max_tokens):
            if len(tokens) >= CONFIG["context_length"]:
                break
            x = torch.tensor([tokens], device=device)
            logits = model(x)
            next_token = int(torch.argmax(logits[0, -1, :]))
            if next_token == CONFIG["eos_id"]:
                break
            tokens.append(next_token)
    model.train()
    reply_tokens = [t for t in tokens[len(prompt_tokens(message, None)):] if t < 256]
    return bytes(reply_tokens).decode("utf-8", errors="replace").strip()


def train_coding_specialization(steps: int = 800, lr: float = 6e-4, batch_size: int = 16):
    print("=" * 70)
    print("     💻 ADDESTRAMENTO INTENSIVO DI CODING: LUMINA AI")
    print("=" * 70)

    curriculum_path = Path("data/lumina_coding_curriculum.jsonl")
    if not curriculum_path.exists():
        print("[!] File curriculum non trovato. Creazione in corso...")
        import data.build_coding_curriculum as bcc
        bcc.build_curriculum()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo di addestramento: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Carica Modello
    model_path = Path("model.pt")
    if not model_path.exists():
        model_path = Path("model-50k.pt")

    print(f"[*] Caricamento pesi modello da '{model_path}'...")
    model, metadata = load_model(model_path)
    model = model.to(device)

    # 2. Carica Dataset Coding
    print(f"[*] Caricamento curriculum da '{curriculum_path}'...")
    train_ex, val_ex = read_examples(curriculum_path, CONFIG)
    print(f"    Esempi di addestramento: {len(train_ex)} | Validazione: {len(val_ex)}")

    # 3. Test Iniziale Prima dell'Addestramento
    print("\n--- TEST CODING PRIMA DELL'ADDESTRAMENTO ---")
    for q in TEST_PROMPTS[:2]:
        ans = generate_live(model, q)
        print(f"Q: {q}\nA: {ans}\n")

    # 4. Ottimizzatore & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(1337)

    print(f"[*] Avvio {steps} step di addestramento su GPU con AdamW...")
    print("-" * 70)
    start_time = time.time()
    total_loss = 0.0

    model.train()
    for step in range(1, steps + 1):
        x, y = batch(train_ex, batch_size=batch_size, rng=rng)
        x, y = x.to(device), y.to(device)

        # Learning rate warmup and cosine decay
        progress = step / steps
        current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Checkpoint e generazione live ogni 200 step
        if step % 200 == 0 or step == steps:
            avg_loss = total_loss / 200 if step % 200 == 0 else total_loss / (step % 200)
            val_loss = evaluate(model, val_ex)
            elapsed = time.time() - start_time
            print(f"\n[Step {step:4d}/{steps}] Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Tempo: {elapsed:.1f}s | LR: {current_lr:.2e}")
            
            # Mostra cosa ha imparato Lumina in tempo reale
            print(">>> LIVE GENERATION (Cosa risponde adesso Lumina):")
            sample_prompt = TEST_PROMPTS[(step // 200) % len(TEST_PROMPTS)]
            ans = generate_live(model, sample_prompt)
            print(f"    Domanda: \"{sample_prompt}\"")
            print(f"    Lumina : \"{ans}\"")
            print("-" * 70)
            total_loss = 0.0

    # 5. Valutazione Finale Completa
    print("\n======================================================================")
    print("             🎉 ADDESTRAMENTO CODING COMPLETATO CON SUCCESSO!")
    print("======================================================================")
    print("Verifica finale su tutti i prompt di programmazione:\n")
    for i, q in enumerate(TEST_PROMPTS, 1):
        ans = generate_live(model, q)
        print(f"[{i}] {q}")
        print(f"    🤖 Lumina Codice: {ans}\n")

    # 6. Salvataggio Pesi
    print("[*] Salvataggio pesi aggiornati in model.pt...")
    torch.save({"config": model.config, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, "model.pt")

    # Esporta pesi per il browser
    export_path = Path("dist/model/weights.json")
    metadata = {
        "origin": "Lumina Coding Intensive Specialization",
        "steps": steps,
        "final_loss": round(val_loss, 4),
        "timestamp": time.time(),
    }
    export(model, metadata, export_path)
    print(f"[+] Pesi esportati in '{export_path}' per il browser web!")

    # Esporta in ONNX
    try:
        from export_onnx import export_model
        export_model("hf_export_lumina_large", "lumina_2.75m.onnx", "Lumina 2.75M Coding")
    except Exception as e:
        print(f"[!] Nota export ONNX: {e}")

    print("\n[SUCCESS] Lumina ha assimilato la sintassi e la logica di programmazione!")


if __name__ == "__main__":
    train_coding_specialization(steps=800)
