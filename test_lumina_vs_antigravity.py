"""
test_lumina_vs_antigravity.py - Sfida di Esame: Antigravity interroga Lumina-Next 135M
Esegue 6 quesiti d'esame multi-dominio su Lumina e registra le risposte esatte e le metriche.
"""

import sys
import os
import time
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from modeling_lumina_next import LuminaNextForCausalLM
from tokenizers import ByteLevelBPETokenizer

QUESTIONS = [
    {
        "id": 1,
        "domain": "Coding & Python",
        "question": "Scrivi una funzione Python chiamata filtra_pari che accetta una lista di numeri interi e restituisce solo i numeri pari usando una list comprehension.",
        "antigravity_ideal": "def filtra_pari(numeri):\n    return [x for x in numeri if x % 2 == 0]"
    },
    {
        "id": 2,
        "domain": "Algoritmi & Computer Science",
        "question": "Qual e' la complessita' computazionale della ricerca binaria in un array ordinato di N elementi e perche'?",
        "antigravity_ideal": "La complessita' e' O(log n) perche' a ogni iterazione l'algoritmo dimezza lo spazio di ricerca."
    },
    {
        "id": 3,
        "domain": "Biologia & Scienze Naturali",
        "question": "Come funziona la fotosintesi clorofilliana nelle piante e quali sostanze produce e rilascia?",
        "antigravity_ideal": "Le piante utilizzano luce solare, acqua e CO2 per produrre glucosio (nutrimento) e rilasciare ossigeno."
    },
    {
        "id": 4,
        "domain": "Geografia Italiana",
        "question": "Qual e' il lago piu' grande d'Italia, quanto e' grande approssimativamente e tra quali regioni si trova?",
        "antigravity_ideal": "Il Lago di Garda, con circa 370 km², situato tra Lombardia, Veneto e Trentino-Alto Adige."
    },
    {
        "id": 5,
        "domain": "Grammatica & Regole della Crusca",
        "question": "Come si scrive correttamente 'qual e'' e perche' non si deve usare l'apostrofo?",
        "antigravity_ideal": "Si scrive 'qual e'' rigorosamente senza apostrofo perche' e' un troncamento e non un'elisione."
    },
    {
        "id": 6,
        "domain": "Onesta' Epistemica & Limiti Futuri",
        "question": "Chi vincera' la finale di Champions League nel 2045?",
        "antigravity_ideal": "Non e' possibile saperlo perche' il 2045 appartiene al futuro e l'evento non e' ancora avvenuto."
    }
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tok_dir = Path("tokenizer_lumina_next")
    tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    
    ckpt_path = "checkpoints_lumina_next/checkpoint_step8000.pt"
    print(f"[*] Caricamento Lumina-Next da {ckpt_path} su {device} ({dtype})...")
    model = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=dtype).eval()
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[+] Pesi step8000 caricati con successo.")

    results = []

    print("\n" + "=" * 75)
    print("      ⚔️ TEST DIRETTO: ANTIGRAVITY INTERROGA LUMINA-NEXT 135M ⚔️")
    print("=" * 75)

    for item in QUESTIONS:
        q_text = item["question"]
        dom = item["domain"]
        qid = item["id"]

        prompt = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n"
        in_ids = tok.encode(prompt).ids
        in_tensor = torch.tensor([in_ids], device=device, dtype=torch.long)

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(in_tensor, max_new_tokens=90, temperature=0.1, top_p=0.9)
        elapsed = time.perf_counter() - t0

        gen_tokens = len(out_ids[0]) - len(in_ids)
        speed = gen_tokens / elapsed if elapsed > 0 else 0

        raw_reply = tok.decode(out_ids[0][len(in_ids):].tolist()).strip()
        if "<|im_end|>" in raw_reply:
            raw_reply = raw_reply.split("<|im_end|>")[0].strip()

        print(f"\n[QUESITO #{qid}] DOMINIO: {dom}")
        print(f"❓ Domanda Antigravity : {q_text}")
        print(f"💡 Risposta Lumina     : {raw_reply}")
        print(f"⏱️  Latenza: {elapsed:.2f}s | Velocità: {speed:.1f} tok/s")
        print("-" * 75)

        results.append({
            "id": qid,
            "domain": dom,
            "question": q_text,
            "lumina_answer": raw_reply,
            "antigravity_ideal": item["antigravity_ideal"],
            "latency_sec": round(elapsed, 3),
            "speed_tok_s": round(speed, 1)
        })

    out_file = Path("test_results_duel.json")
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] Risultati completi salvati in {out_file}")

if __name__ == "__main__":
    main()
