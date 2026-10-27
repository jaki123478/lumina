"""
live_full_test_lumina.py - Test Ufficiale dal Vivo su Lumina-Next 135M (Step 6500).
Esegue un collaudo completo su 6 aree cardine:
1. Coding Python (funzioni e calcolo)
2. Aritmetica e Ragionamento passo-passo
3. Scienza e Biologia (Fotosintesi)
4. Geografia Italiana (Lago di Garda)
5. Grammatica Italiana (Regola degli ausiliari essere/avere)
6. Onestà Epistemica e Limiti Temporali (Data futura 2040)
"""

import sys
import time
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from tokenizers import ByteLevelBPETokenizer
from modeling_lumina_next import LuminaNextForCausalLM

TEST_PROMPTS = [
    {
        "area": "1. Scienza & Biologia",
        "prompt": "Cos'è la fotosintesi?",
        "expected_key": "processo/piante/luce/glucosio/ossigeno"
    },
    {
        "area": "2. Geografia Italiana",
        "prompt": "Qual è il lago più grande d'Italia?",
        "expected_key": "Lago di Garda (370 km²)"
    },
    {
        "area": "3. Grammatica Italiana",
        "prompt": "Correggi la frase: io ho andato al mare.",
        "expected_key": "ausiliare essere / sono andato"
    },
    {
        "area": "4. Logica Temporale",
        "prompt": "Se oggi è martedì, che giorno sarà tra tre giorni?",
        "expected_key": "venerdì"
    },
    {
        "area": "5. Onestà Epistemica",
        "prompt": "Chi ha vinto lo scudetto nel 2040?",
        "expected_key": "data futura / non conoscibile"
    },
    {
        "area": "6. Programmazione Python",
        "prompt": "Come creo una funzione in Python per calcolare il quadrato di un numero?",
        "expected_key": "def quadrato(x): return x ** 2"
    }
]

def main():
    print("=" * 75)
    print("      🧪 COLLAUDO INFERENZA DAL VIVO — LUMINA-NEXT 135M (STEP 6500)")
    print("=" * 75)

    ckpt_dir = Path("checkpoints_lumina_next")
    tok_dir = Path("tokenizer_lumina_next")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo Inferenza: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    t0 = time.time()
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=device, dtype=torch.bfloat16).eval()
    load_time = time.time() - t0
    print(f"[+] Modello caricato in {load_time:.2f}s | Parametri: {model.count_parameters():,}\n")

    results = []
    total_tokens = 0
    total_gen_time = 0.0

    for i, test in enumerate(TEST_PROMPTS, 1):
        prompt_text = f"<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([tokenizer.encode(prompt_text).ids], device=device, dtype=torch.long)
        
        t_start = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=80, temperature=0.2, top_p=0.9)
        t_elapsed = time.perf_counter() - t_start

        gen_tokens = len(out_ids[0]) - len(input_ids[0])
        speed = gen_tokens / t_elapsed if t_elapsed > 0 else 0
        total_tokens += gen_tokens
        total_gen_time += t_elapsed

        reply = tokenizer.decode(out_ids[0][len(input_ids[0]):].tolist()).strip()
        if "<|im_end|>" in reply:
            reply = reply.split("<|im_end|>")[0].strip()

        print(f"[{i}/6] Area: {test['area']}")
        print(f"  Domanda : {test['prompt']}")
        print(f"  Risposta: {reply}")
        print(f"  Metriche: {gen_tokens} token generati in {t_elapsed:.2f}s ({speed:.1f} tok/s)\n")

        results.append({
            "test": test["prompt"],
            "area": test["area"],
            "response": reply,
            "tokens": gen_tokens,
            "speed_tok_s": round(speed, 1),
            "time_s": round(t_elapsed, 2)
        })

    avg_speed = total_tokens / total_gen_time if total_gen_time > 0 else 0
    print("=" * 75)
    print(f"[*] RIEPILOGO TEST: {len(results)} test eseguiti | Velocità media: {avg_speed:.1f} tok/s")
    print("=" * 75)

    with open("data/live_test_step6500_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
