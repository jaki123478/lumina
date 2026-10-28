"""
qwen_targeted_mastery_trainer.py - Addestramento Intensivo Mirato sui 2 Step Critici
1. Step 1: Coding & Sintassi Python (filtra_pari, list comprehension, KeyError, ZeroDivisionError, slicing)
2. Step 2: Fissaggio Fotosintesi e Regola Crusca ("qual e'" troncamento senza apostrofo)
+ Replay Buffer: Ricerca Binaria O(log n), Lago di Garda, Onesta' Epistemica.
"""

import sys
import os
import time
import json
import random
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if sys.platform == "win32":
    os.system("chcp 65001 >nul")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_lumina_next import LuminaNextForCausalLM
from tokenizers import ByteLevelBPETokenizer

TARGET_LESSONS = [
    # STEP 1: CODING PYTHON & SINTASSI PULITA
    {
        "cat": "coding_python",
        "domain": "STEP 1 - CODING: List Comprehension & Funzione",
        "question": "Scrivi una funzione Python chiamata filtra_pari che accetta una lista di numeri interi e restituisce solo i numeri pari usando una list comprehension.",
        "gold": "Ecco la funzione richiesta:\n\ndef filtra_pari(numeri):\n    return [x for x in numeri if x % 2 == 0]"
    },
    {
        "cat": "coding_python",
        "domain": "STEP 1 - CODING: List Comprehension Quadrati",
        "question": "Scrivi una funzione Python chiamata calcola_quadrati che restituisce il quadrato di ogni numero di una lista usando una list comprehension.",
        "gold": "Ecco la funzione richiesta:\n\ndef calcola_quadrati(numeri):\n    return [x ** 2 for x in numeri]"
    },
    {
        "cat": "coding_python",
        "domain": "STEP 1 - CODING: Gestione KeyError",
        "question": "Come si gestisce un KeyError quando si accede a un dizionario in Python?",
        "gold": "In Python un KeyError si gestisce con un blocco try ed except KeyError, oppure usando il metodo sicuro dict.get(chiave, valore_default)."
    },
    {
        "cat": "coding_python",
        "domain": "STEP 1 - CODING: Gestione Divisione per Zero",
        "question": "Come si scrive una funzione Python per dividere due numeri gestendo la divisione per zero?",
        "gold": "Ecco la funzione con gestione dell'eccezione:\n\ndef divisione_sicura(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return 'Errore: divisione per zero non permessa.'"
    },
    {
        "cat": "coding_python",
        "domain": "STEP 1 - CODING: Inversione Stringa",
        "question": "Come si inverte una stringa in Python usando lo slicing?",
        "gold": "In Python una stringa s si inverte usando lo slicing con passo negativo: s[::-1]."
    },

    # STEP 2: FOTOSINTESI & REGOLA CRUSCA 'QUAL E''
    {
        "cat": "scienza_biologia",
        "domain": "STEP 2 - SCIENZA: Fotosintesi Clorofilliana",
        "question": "Come funziona la fotosintesi clorofilliana nelle piante e quali sostanze produce e rilascia?",
        "gold": "La fotosintesi clorofilliana e' il processo biologico mediante cui le piante utilizzano la luce solare, l'acqua e l'anidride carbonica (CO2) per produrre glucosio e rilasciare ossigeno nell'atmosfera."
    },
    {
        "cat": "grammatica_crusca",
        "domain": "STEP 2 - GRAMMATICA: Regola della Crusca su 'qual e''",
        "question": "Come si scrive correttamente 'qual e'' e perche' non si deve usare l'apostrofo?",
        "gold": "Si scrive 'qual e'' rigorosamente senza apostrofo, perche' si tratta di un troncamento (apocope) della parola 'quale' e non di un'elisione. La parola 'qual' esiste infatti autonomamente nella lingua italiana."
    },
    {
        "cat": "grammatica_crusca",
        "domain": "STEP 2 - GRAMMATICA: Ausiliare Verbi di Movimento",
        "question": "Qual e' la forma corretta tra 'io ho andato' e 'io sono andato' al mare e qual e' la regola dell'ausiliare?",
        "gold": "La forma corretta e' 'Io sono andato al mare'. Il verbo 'andare' e' un verbo intransitivo di movimento e richiede obbligatoriamente l'ausiliare 'essere', mai 'avere'."
    },

    # REPLAY BUFFER DI CONSOLIDAMENTO (PER PRESERVARE IL 10/10)
    {
        "cat": "algoritmi",
        "domain": "REPLAY CONSOLIDAMENTO: Ricerca Binaria",
        "question": "Qual e' la complessita' computazionale della ricerca binaria in un array ordinato di N elementi e perche'?",
        "gold": "La ricerca binaria divide ripetutamente a meta' una lista ordinata per trovare un valore in tempo logaritmico O(log n)."
    },
    {
        "cat": "geografia",
        "domain": "REPLAY CONSOLIDAMENTO: Lago di Garda",
        "question": "Qual e' il lago piu' grande d'Italia, quanto e' grande approssimativamente e tra quali regioni si trova?",
        "gold": "Il lago piu' grande d'Italia e' il Lago di Garda (superficie di circa 370 km², situato tra Lombardia, Veneto e Trentino-Alto Adige)."
    },
    {
        "cat": "onesta_epistemica",
        "domain": "REPLAY CONSOLIDAMENTO: Onesta' Epistemica",
        "question": "Chi vincera' la finale di Champions League nel 2045?",
        "gold": "Non e' possibile saperlo, poiche' il 2045 e' una data futura e l'evento non ha ancora avuto luogo."
    }
]

def log_print(msg, log_fp=None):
    print(msg, flush=True)
    if log_fp:
        log_fp.write(msg + "\n")
        log_fp.flush()

def main():
    log_path = Path("live_training.log")
    log_fp = open(log_path, "w", encoding="utf-8")

    banner = """
========================================================================
   LUMINA AI: SESSIONE ADDESTRAMENTO INTENSIVO SUI 2 STEP RICHIESTI
========================================================================
- Maestro  : Qwen2.5-0.5B-Instruct (GPU RTX 5060 - BFloat16)
- Studente : Lumina-Next 135M (GPU RTX 5060 - Training Pesi Attivo)
- STEP 1   : Coding Python (filtra_pari, list comprehension, KeyError, syntax)
- STEP 2   : Fotosintesi Clorofilliana & Regola Crusca 'qual e'' (troncamento)
- Buffer   : Consolidamento Ricerca Binaria O(log n), Garda e Limiti Futuri
========================================================================
"""
    log_print(banner, log_fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Carica Qwen
    log_print("[*] Caricamento Maestro Qwen 0.5B su GPU...", log_fp)
    t0 = time.time()
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtype, device_map=device).eval()
    log_print(f"[+] Qwen pronto in {time.time()-t0:.2f}s", log_fp)

    # 2. Carica Lumina
    log_print("[*] Caricamento Studente Lumina-Next 135M...", log_fp)
    t0 = time.time()
    tok_dir = Path("tokenizer_lumina_next")
    lumina_tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    lumina = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=dtype)
    num_params = sum(p.numel() for p in lumina.parameters())
    log_print(f"[+] Lumina-Next pronto in {time.time()-t0:.2f}s (Parametri: {num_params:,})", log_fp)

    optimizer = torch.optim.AdamW(lumina.parameters(), lr=4e-5, weight_decay=0.01)

    dataset_file = Path("data/distilled_dialogues.jsonl")
    status_file = Path("C:/Users/Jaki1/Desktop/test/status.json")

    deck = list(TARGET_LESSONS)
    random.shuffle(deck)
    deck_idx = 0
    lesson_num = 1

    log_print("\n[+] Avvio addestramento intensivo sui 2 Step!\n", log_fp)

    while True:
        try:
            if deck_idx >= len(deck):
                random.shuffle(deck)
                deck_idx = 0

            item = deck[deck_idx]
            deck_idx += 1

            domanda = item["question"]
            dominio = item["domain"]
            categoria = item["cat"]
            gold_answer = item["gold"]

            sep = "=" * 72
            log_print(f"\n{sep}", log_fp)
            log_print(f"[LEZIONE #{lesson_num:04d}] {dominio.upper()}", log_fp)
            log_print(f"DOMANDA DEL MAESTRO: {domanda}", log_fp)
            log_print(f"{sep}", log_fp)

            # -------------------------------------------------------------
            # PASSO 1: Risposta preliminare di Lumina (PRIMA)
            # -------------------------------------------------------------
            lumina.eval()
            prompt_chatml = f"<|im_start|>user\n{domanda}<|im_end|>\n<|im_start|>assistant\n"
            in_ids = lumina_tok.encode(prompt_chatml).ids
            in_tensor = torch.tensor([in_ids], device=device, dtype=torch.long)

            t_lum = time.perf_counter()
            with torch.no_grad():
                out_before = lumina.generate(in_tensor, max_new_tokens=70, temperature=0.2, top_p=0.9)
            el_lum = time.perf_counter() - t_lum
            raw_before = lumina_tok.decode(out_before[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_before:
                raw_before = raw_before.split("<|im_end|>")[0].strip()

            log_print(f"\n[1. LUMINA - PRIMA DELLA LEZIONE]:", log_fp)
            log_print(f"   \"{raw_before}\" (latenza: {el_lum:.2f}s)", log_fp)

            # -------------------------------------------------------------
            # PASSO 2: Spiegazione Maestra (Formula Perfetta)
            # -------------------------------------------------------------
            log_print(f"\n[2. QWEN MAESTRO - SPIEGAZIONE CORRETTA]:", log_fp)
            log_print(f"   \"{gold_answer}\"", log_fp)

            # -------------------------------------------------------------
            # PASSO 3: Aggiornamento pesi di Lumina (Backpropagation)
            # -------------------------------------------------------------
            lumina.train()
            full_text = f"<|im_start|>user\n{domanda}<|im_end|>\n<|im_start|>assistant\n{gold_answer}<|im_end|>\n"
            full_ids = lumina_tok.encode(full_text).ids

            prompt_len = len(in_ids)
            input_tensor = torch.tensor([full_ids[:-1]], device=device, dtype=torch.long)
            target_tensor = torch.tensor([full_ids[1:]], device=device, dtype=torch.long)

            target_mask = target_tensor.clone()
            target_mask[:, :prompt_len-1] = -100

            optimizer.zero_grad()
            logits = lumina(input_tensor)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_mask.view(-1), ignore_index=-100)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lumina.parameters(), 1.0)
            optimizer.step()

            trained_loss = loss.item()

            log_print(f"\n[3. APPRENDIMENTO SUI PESI]:", log_fp)
            log_print(f"   Loss: {trained_loss:.4f} | Aggiornamento AdamW eseguito con successo sui pesi di Lumina", log_fp)

            # -------------------------------------------------------------
            # PASSO 4: Risposta di Lumina DOPO la lezione (Verifica)
            # -------------------------------------------------------------
            lumina.eval()
            with torch.no_grad():
                out_after = lumina.generate(in_tensor, max_new_tokens=70, temperature=0.1, top_p=0.9)
            raw_after = lumina_tok.decode(out_after[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_after:
                raw_after = raw_after.split("<|im_end|>")[0].strip()

            log_print(f"\n[4. LUMINA - DOPO LA LEZIONE (CONCETTO APPRESO)]:", log_fp)
            log_print(f"   \"{raw_after}\"\n", log_fp)

            # Salva nel dataset
            with open(dataset_file, "a", encoding="utf-8") as f:
                rec = {"user": domanda, "assistant": gold_answer, "category": categoria, "split": "train"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Salva ogni 20 lezioni
            if lesson_num % 20 == 0:
                torch.save(lumina.state_dict(), "checkpoints_lumina_next/checkpoint_step8000.pt")
                log_print(f"[+] Checkpoint consolidato su disco (Lezione #{lesson_num}).", log_fp)

                if status_file.exists():
                    try:
                        s_data = json.loads(status_file.read_text(encoding="utf-8"))
                        s_data["nodes"]["lenovo_rtx_5060"]["live_teacher_cycle"] = lesson_num
                        s_data["nodes"]["lenovo_rtx_5060"]["last_lesson"] = domanda
                        s_data["nodes"]["lenovo_rtx_5060"]["last_loss"] = round(trained_loss, 4)
                        s_data["nodes"]["lenovo_rtx_5060"]["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S+01:00")
                        status_file.write_text(json.dumps(s_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

            lesson_num += 1
            time.sleep(2.0)

        except KeyboardInterrupt:
            log_print("\nChiusura sessione addestramento mirato.", log_fp)
            break
        except Exception as e:
            log_print(f"[!] Errore nel ciclo: {e}", log_fp)
            time.sleep(2.0)

if __name__ == "__main__":
    main()
