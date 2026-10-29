"""
evaluate_8_questions.py - Benchmark comparativo sulle 8 domande standard di controllo.
Valuta Lumina-Next (o un checkpoint specificato) su:
1. Cos'è la fotosintesi? (2 pt)
2. Qual è il lago più grande d'Italia? (2 pt)
3. Quanto fa 17 per 6? (2 pt)
4. Se oggi è martedì, che giorno sarà tra tre giorni? (2 pt)
5. Correggi la frase: io ho andato al mare. (2 pt)
6. Scrivi una frase di cinque parole sul mare. (2 pt)
7. In Python, come si scrive il comando per stampare la parola ciao? (2 pt)
8. Chi ha vinto lo scudetto nel 2040? (2 pt se ammette di non saperlo, 0 altrimenti)

Punteggio: 2 = Risposta esatta e corretta, 1 = Parziale/imprecisa, 0 = Errata o allucinata.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modeling_lumina_next import LuminaNextForCausalLM

QUESTIONS = [
    {
        "id": 1,
        "prompt": "Cos'è la fotosintesi?",
        "expected": "Processo con cui le piante convertono luce, acqua e anidride carbonica in glucosio e ossigeno.",
        "max_points": 2
    },
    {
        "id": 2,
        "prompt": "Qual è il lago più grande d'Italia?",
        "expected": "Lago di Garda.",
        "max_points": 2
    },
    {
        "id": 3,
        "prompt": "Quanto fa 17 per 6?",
        "expected": "102 (17 * 6 = 102).",
        "max_points": 2
    },
    {
        "id": 4,
        "prompt": "Se oggi è martedì, che giorno sarà tra tre giorni?",
        "expected": "Venerdì.",
        "max_points": 2
    },
    {
        "id": 5,
        "prompt": "Correggi la frase: io ho andato al mare.",
        "expected": "Io sono andato al mare.",
        "max_points": 2
    },
    {
        "id": 6,
        "prompt": "Scrivi una frase di cinque parole sul mare.",
        "expected": "Frase di esattamente 5 parole sul tema mare.",
        "max_points": 2
    },
    {
        "id": 7,
        "prompt": "In Python, come si scrive il comando per stampare la parola ciao?",
        "expected": "print('ciao') o print(\"ciao\").",
        "max_points": 2
    },
    {
        "id": 8,
        "prompt": "Chi ha vinto lo scudetto nel 2040?",
        "expected": "Non si può sapere / il 2040 è una data futura.",
        "max_points": 2
    }
]


def score_response(q_id: int, answer: str) -> tuple[int, str]:
    """Valutazione automatica deterministica oggettiva con supporto CoT scratchpad."""
    # Se presente scratchpad <think>...</think>, estrai la risposta finale ma mantieni il testo completo
    think_part = ""
    final_part = answer
    if "</think>" in answer:
        parts = answer.split("</think>")
        think_part = parts[0].replace("<think>", "").strip()
        final_part = parts[1].strip()
    
    a_full = answer.lower().strip()
    a_final = final_part.lower().strip() if final_part else a_full
    
    if q_id == 1: # Fotosintesi
        if any(w in a_full for w in ["piante", "luce", "energia", "clorofilliana", "anidride", "glucosio"]):
            return 2, "Esatto (riconosce processo biologico/piante/luce)"
        return 0, "Non pertinente"
        
    elif q_id == 2: # Lago più grande
        if "garda" in a_full:
            return 2, "Esatto (Lago di Garda)"
        elif "maggiore" in a_full:
            return 0, "Sbagliato (ha indicato il Lago Maggiore)"
        return 0, "Non riconosce il Lago di Garda"
        
    elif q_id == 3: # 17 * 6
        if "102" in a_full or "102" in a_final:
            return 2, "Esatto (102 calcolato correttamente)"
        return 0, "Calcolo errato"
        
    elif q_id == 4: # Martedì + 3
        if "venerdì" in a_full or "venerdi" in a_full or "venerdì" in a_final:
            return 2, "Esatto (Venerdì)"
        elif "giovedì" in a_full or "sabato" in a_full:
            return 1, "Parziale/Sfasato di un giorno"
        return 0, "Giorno errato"
        
    elif q_id == 5: # io ho andato -> io sono andato
        if "sono andato" in a_full:
            return 2, "Esatto (io sono andato al mare)"
        return 0, "Non corretto"
        
    elif q_id == 6: # 5 parole sul mare
        # Pulisci punteggiatura sulla risposta finale (dopo </think>)
        cleaned = "".join([c if c.isalnum() or c.isspace() else " " for c in a_final]).strip()
        words = cleaned.split()
        if len(words) == 5 and "mare" in a_final:
            return 2, f"Esatto (Esattamente 5 parole: {words})"
        elif abs(len(words) - 5) <= 1 and "mare" in a_final:
            return 1, f"Parziale ({len(words)} parole invece di 5: {words})"
        return 0, f"Non rispetta il vincolo ({len(words)} parole nella risposta finale)"
        
    elif q_id == 7: # print('ciao')
        if "print(" in a_full and "ciao" in a_full:
            return 2, "Esatto (print con ciao)"
        return 0, "Sintassi Python non corretta"
        
    elif q_id == 8: # Scudetto 2040
        if any(w in a_full for w in ["non posso", "futur", "impossibile", "non è ancora", "non si può", "non so", "non è possibile"]):
            return 2, "Esatto (Onestà: riconosce evento futuro sconosciuto)"
        return 0, "Allucinazione su evento futuro"

    return 0, "Non valutato"


def evaluate_model(checkpoint_dir: str = "checkpoints_lumina_next", tag: str = "baseline"):
    # Decoding deterministico: il punteggio deve essere confrontabile tra run.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("=" * 70)
    print(f" 🧪 VALUTAZIONE BENCHMARK 8 DOMANDE: {tag.upper()}")
    print(f" Checkpoint: {checkpoint_dir}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tok_dir = Path("tokenizer_lumina_next")
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    model = LuminaNextForCausalLM.from_pretrained(checkpoint_dir).to(device=device, dtype=dtype)
    model.eval()

    total_score = 0
    max_possible = len(QUESTIONS) * 2
    results = []

    for q in QUESTIONS:
        prompt_text = f"<|im_start|>user\n{q['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([tokenizer.encode(prompt_text).ids], device=device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=90, temperature=0.0, top_p=1.0)
        
        gen_tokens = output_ids[0][len(input_ids[0]):].tolist()
        ans = tokenizer.decode(gen_tokens).strip()
        if "<|im_end|>" in ans:
            ans = ans.split("<|im_end|>")[0].strip()

        score, feedback = score_response(q["id"], ans)
        total_score += score

        res_item = {
            "id": q["id"],
            "prompt": q["prompt"],
            "model_answer": ans,
            "score": score,
            "feedback": feedback
        }
        results.append(res_item)
        print(f"[{q['id']}/8] {q['prompt']}")
        print(f"     -> Risposta: {repr(ans)}")
        print(f"     -> Punteggio: {score}/2 ({feedback})\n")

    print("=" * 70)
    print(f" RISULTATO COMPLESSIVO {tag.upper()}: {total_score} / {max_possible} PUNTI ({total_score/max_possible*100:.1f}%)")
    print("=" * 70)

    output_path = Path("data") / f"benchmark_{tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "tag": tag,
            "checkpoint": checkpoint_dir,
            "total_score": total_score,
            "max_possible": max_possible,
            "pct": round(total_score / max_possible * 100, 2),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] Risultati salvati in: {output_path}")

    del model
    torch.cuda.empty_cache()
    return total_score, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_lumina_next")
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.tag)
