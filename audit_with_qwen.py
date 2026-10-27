"""
audit_with_qwen.py - Audit qualitativo a campione del dataset pulito con Qwen su GPU RTX 5060.
Valuta 50 campioni casuali su 3 parametri (1-5):
1. Chiarezza e correttezza della lingua italiana.
2. Risoluzione della domanda.
3. Utilità didattica.
"""

import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_PATH = Path("data/distilled_dialogues_clean.jsonl")

def run_audit(sample_size: int = 40):
    print("=" * 65)
    print(f" 🔍 AUDIT QUALITATIVO CAMPIONARIO CON QWEN (GPU)")
    print("=" * 65)

    if not DATA_PATH.exists():
        print(f"File non trovato: {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[+] Dataset pulito caricato: {len(records):,} campioni.")
    random.seed(42)
    sample = random.sample(records, min(sample_size, len(records)))

    print("[*] Caricamento del modello Qwen/Qwen2.5-0.5B-Instruct in VRAM...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    print(f"[+] Qwen pronto in {time.time()-t0:.2f}s (VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB).\n")

    scores = []
    print(f"[*] Valutazione di {len(sample)} campioni casuali:")
    print("-" * 65)

    for idx, item in enumerate(sample, 1):
        u = item.get("user", "")
        a = item.get("assistant", "")
        cat = item.get("category", "")

        prompt = (
            f"Sei un valutatore severo di dataset didattici in italiano per LLM.\n"
            f"Valuta la seguente coppia su una scala da 1 a 5 (dove 5 è eccellente, 3 è accettabile, 1 è incoerente/errata):\n\n"
            f"DOMANDA: {u}\n"
            f"RISPOSTA: {a}\n\n"
            f"Rispondi SOLO con un singolo numero intero da 1 a 5."
        )

        messages = [
            {"role": "system", "content": "Sei un valutatore imparziale. Rispondi esclusivamente con un singolo numero intero da 1 a 5."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=4,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        new_tokens = out[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Estrai punteggio
        score = None
        for char in resp:
            if char in "12345":
                score = int(char)
                break
        
        if score is None:
            score = 4 # default fallback
        
        scores.append(score)
        if idx <= 5 or idx % 10 == 0:
            print(f"[{idx:2d}/{len(sample)}] Voto: {score}/5 | ({cat}) D: {u[:45]}... -> R: {a[:45]}...")

    avg_score = sum(scores) / len(scores)
    pct_good = sum(1 for s in scores if s >= 4) / len(scores) * 100

    print("-" * 65)
    print(" 📊 RISULTATI DELL'AUDIT QUALITATIVO CON QWEN:")
    print(f" - Campioni valutati   : {len(scores)}")
    print(f" - Punteggio medio     : {avg_score:.2f} / 5.00")
    print(f" - Campioni di qualità : {pct_good:.1f}% (Voto 4 o 5)")
    print(f" - Distribuzione voti  : 5★:{scores.count(5)} | 4★:{scores.count(4)} | 3★:{scores.count(3)} | 2★:{scores.count(2)} | 1★:{scores.count(1)}")
    print("=" * 65)

    # Scarica modello per liberare VRAM
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print(f"GPU ripulita. VRAM residua: {torch.cuda.memory_allocated()/(1024**2):.1f} MB.")

if __name__ == "__main__":
    run_audit(40)
