"""
qwen_max_power_trainer.py - POTENZA MASSIMA: Qwen2.5-Coder-1.5B-Instruct Insegna a Lumina-Next 135M
Caratteristiche:
- Maestro  : Qwen2.5-Coder-1.5B-Instruct (1.54 Miliardi di parametri, BFloat16 su RTX 5060)
- Studente : Lumina-Next 135M (Backprop in tempo reale sui pesi)
- Ambiti   : Coding Python, Algoritmi, Strutture Dati, SQL, Web/API, Scienze, Geografia, Crusca
- Terminale: Visualizzazione pulita, priva di bug grafici, su live_training.log
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
from qwen_precision import precision_instruction

CURRICULUM_MAX_POWER = [
    # -----------------------------------------------------------------
    # CODING PYTHON & INGEGNERIA SOFTWARE (QWEN CODER SPECIALIST)
    # -----------------------------------------------------------------
    {
        "cat": "coding_python",
        "domain": "Coding: List Comprehension & Funzione",
        "question": "Scrivi una funzione Python chiamata filtra_pari che accetta una lista di numeri interi e restituisce solo i numeri pari usando una list comprehension.",
        "sys": "Sei un senior software engineer. Spiega brevemente la funzione e fornisci il codice Python esatto e pulito."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Calcolo Quadrati con List Comprehension",
        "question": "Scrivi una funzione Python chiamata calcola_quadrati che restituisce il quadrato di ogni numero di una lista usando una list comprehension.",
        "sys": "Sei un programmatore Python esperto. Mostra la sintassi della funzione con return [x**2 for x in lista]."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Gestione Eccezioni KeyError",
        "question": "Come si gestisce un'eccezione KeyError quando si accede a un dizionario in Python?",
        "sys": "Sei un software engineer. Spiega come usare try/except KeyError oppure dict.get(chiave, default) con un esempio pratico."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Gestione Divisione per Zero",
        "question": "Come si scrive una funzione Python per dividere due numeri gestendo la divisione per zero?",
        "sys": "Sei un docente di informatica. Mostra la funzione divisione_sicura(a, b) con blocco try ed except ZeroDivisionError."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Inversione Stringhe Slicing",
        "question": "Come si inverte una stringa in Python usando lo slicing?",
        "sys": "Sei un programmatore Python. Spiega la sintassi dello slicing a passo negativo s[::-1] in modo conciso."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Classi e Costruttore __init__",
        "question": "Come si definisce una classe Persona con costruttore __init__ e attributo nome in Python?",
        "sys": "Sei un programmatore OOP. Spiega il ruolo del costruttore __init__ e del parametro self con codice Python minimale."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: File Context Manager",
        "question": "Perche' in Python si usa 'with open(file) as f' per aprire i file?",
        "sys": "Sei un senior developer. Spiega il context manager e la chiusura automatica garantita delle risorse."
    },

    # -----------------------------------------------------------------
    # ALGORITMI & COMPUTER SCIENCE
    # -----------------------------------------------------------------
    {
        "cat": "algoritmi",
        "domain": "Algoritmi: Ricerca Binaria",
        "question": "Qual e' la complessita' computazionale della ricerca binaria in un array ordinato di N elementi e perche'?",
        "sys": "Sei un docente di algoritmi. Spiega la complessita' logaritmica O(log n) e il dimezzamento dello spazio di ricerca a ogni passo."
    },
    {
        "cat": "algoritmi",
        "domain": "Algoritmi: Pila LIFO vs Coda FIFO",
        "question": "Qual e' la differenza fondamentale tra una Pila (Stack) e una Coda (Queue)?",
        "sys": "Sei un docente di strutture dati. Spiega Last-In First-Out vs First-In First-Out con esempi chiari."
    },
    {
        "cat": "algoritmi",
        "domain": "Algoritmi: Scalabilita' e Ordinamento",
        "question": "Perche' MergeSort O(n log n) e' piu' efficiente di BubbleSort O(n^2) su grandi collezioni di dati?",
        "sys": "Sei un analista di algoritmi. Spiega la differenza di crescita asintotica su grandi volumi di dati."
    },

    # -----------------------------------------------------------------
    # SCIENZE, GEOGRAFIA, GRAMMATICA & LIMITI
    # -----------------------------------------------------------------
    {
        "cat": "scienza_biologia",
        "domain": "Scienze: Fotosintesi Clorofilliana",
        "question": "Come funziona la fotosintesi clorofilliana nelle piante e quali sostanze produce e rilascia?",
        "sys": "Sei un professore di biologia. Spiega il processo che trasforma luce solare, acqua e CO2 in glucosio e rilascia ossigeno."
    },
    {
        "cat": "grammatica_crusca",
        "domain": "Grammatica: Regola Crusca su 'qual e''",
        "question": "Come si scrive correttamente 'qual e'' e perche' non si deve usare l'apostrofo?",
        "sys": "Sei un linguista della Crusca. Spiega che si tratta di un troncamento (apocope) e mai di un'elisione, quindi senza apostrofo."
    },
    {
        "cat": "grammatica_crusca",
        "domain": "Grammatica: Ausiliare Verbi di Movimento",
        "question": "Qual e' la forma corretta tra 'io ho andato' e 'io sono andato' al mare e qual e' la regola?",
        "sys": "Sei un docente di grammatica. Spiega che 'andare' e' un verbo intransitivo di moto e richiede l'ausiliare 'essere'."
    },
    {
        "cat": "geografia",
        "domain": "Geografia: Lago di Garda",
        "question": "Qual e' il lago piu' grande d'Italia, quanto e' grande approssimativamente e tra quali regioni si trova?",
        "sys": "Sei un docente di geografia. Spiega che e' il Lago di Garda (circa 370 km²) situato tra Lombardia, Veneto e Trentino-Alto Adige."
    },
    {
        "cat": "onesta_epistemica",
        "domain": "Etica: Limiti di Conoscenza Futura",
        "question": "Chi vincera' la finale di Champions League nel 2045?",
        "sys": "Sei un assistente trasparente. Dichiara che il 2045 appartiene al futuro e che non e' possibile prevedere l'evento con certezza."
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
   LUMINA AI: SESSIONE POTENZA MASSIMA (QWEN-CODER 1.5B SU RTX 5060)
========================================================================
- Maestro  : Qwen2.5-Coder-1.5B-Instruct (1.54B Parametri - BFloat16)
- Studente : Lumina-Next 135M (Training Pesi Attivo con AdamW)
- Risorse  : GPU NVIDIA GeForce RTX 5060 Laptop (VRAM ~3.1 GB allocati)
- Ruolo    : Eccellenza nel Coding Python, Algoritmi, Scienze e Lingua
========================================================================
"""
    log_print(banner, log_fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Caricamento Qwen-Coder 1.5B
    log_print("[*] Caricamento Maestro Qwen2.5-Coder-1.5B-Instruct su GPU RTX 5060...", log_fp)
    t0 = time.time()
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", dtype=dtype, device_map=device).eval()
    log_print(f"[+] Qwen2.5-Coder pronto in {time.time()-t0:.2f}s (VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB)", log_fp)

    # 2. Caricamento Lumina-Next 135M
    log_print("[*] Caricamento Studente Lumina-Next 135M...", log_fp)
    t0 = time.time()
    tok_dir = Path("tokenizer_lumina_next")
    lumina_tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    lumina = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=dtype)
    ckpt_path = Path("checkpoints_lumina_next/checkpoint_step8000.pt")
    if ckpt_path.exists():
        lumina.load_state_dict(torch.load(ckpt_path, map_location=device))
    num_params = sum(p.numel() for p in lumina.parameters())
    log_print(f"[+] Lumina-Next pronto in {time.time()-t0:.2f}s (Parametri: {num_params:,} - Ripreso da Lezione #1160)", log_fp)

    optimizer = torch.optim.AdamW(lumina.parameters(), lr=3e-5, weight_decay=0.01)

    dataset_file = Path("data/distilled_dialogues.jsonl")
    status_file = Path("C:/Users/Jaki1/Desktop/test/status.json")

    deck = list(CURRICULUM_MAX_POWER)
    random.shuffle(deck)
    deck_idx = 0
    lesson_num = 2402

    log_print("\n[+] Potenza Massima Attiva: Qwen Coder istruisce Lumina in diretta!\n", log_fp)

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
            sys_instruct = item["sys"]

            sep = "=" * 72
            log_print(f"\n{sep}", log_fp)
            log_print(f"[LEZIONE #{lesson_num:04d}] AMBITO: {dominio.upper()}", log_fp)
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
            # PASSO 2: Spiegazione Maestra di Qwen Coder 1.5B
            # -------------------------------------------------------------
            msgs = [
                {"role": "system", "content": precision_instruction() + f"\n{sys_instruct} Spiega in italiano corretto, chiaro e conciso."},
                {"role": "user", "content": f"Spiega al tuo studente la soluzione corretta a questa domanda: '{domanda}'"}
            ]
            q_chat = qwen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            q_in = qwen_tok([q_chat], return_tensors="pt").to(device)

            t_q = time.perf_counter()
            with torch.no_grad():
                q_out = qwen.generate(**q_in, max_new_tokens=100, temperature=0.2, top_p=0.9)
            el_q = time.perf_counter() - t_q
            qwen_answer = qwen_tok.decode(q_out[0][len(q_in.input_ids[0]):], skip_special_tokens=True).strip()

            log_print(f"\n[2. QWEN CODER 1.5B (MAESTRO) - SPIEGAZIONE ECCELLENTE]:", log_fp)
            log_print(f"   \"{qwen_answer}\" (latenza: {el_q:.2f}s)", log_fp)

            # -------------------------------------------------------------
            # PASSO 3: Aggiornamento Pesi Lumina (Backpropagation)
            # -------------------------------------------------------------
            lumina.train()
            full_text = f"<|im_start|>user\n{domanda}<|im_end|>\n<|im_start|>assistant\n{qwen_answer}<|im_end|>\n"
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
            log_print(f"   Loss: {trained_loss:.4f} | Aggiornamento AdamW eseguito sui parametri di Lumina", log_fp)

            # -------------------------------------------------------------
            # PASSO 4: Risposta di Lumina DOPO la lezione
            # -------------------------------------------------------------
            lumina.eval()
            with torch.no_grad():
                out_after = lumina.generate(in_tensor, max_new_tokens=70, temperature=0.2, top_p=0.9)
            raw_after = lumina_tok.decode(out_after[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_after:
                raw_after = raw_after.split("<|im_end|>")[0].strip()

            log_print(f"\n[4. LUMINA - DOPO LA LEZIONE (CONCETTO APPRESO)]:", log_fp)
            log_print(f"   \"{raw_after}\"\n", log_fp)

            # Salva nel dataset distillato
            with open(dataset_file, "a", encoding="utf-8") as f:
                rec = {"user": domanda, "assistant": qwen_answer, "category": categoria, "split": "train"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Salva checkpoint ogni 20 lezioni
            if lesson_num % 20 == 0:
                torch.save(lumina.state_dict(), "checkpoints_lumina_next/checkpoint_step8000.pt")
                log_print(f"[+] Checkpoint consolidato su disco (Lezione #{lesson_num}).", log_fp)

                if status_file.exists():
                    try:
                        s_data = json.loads(status_file.read_text(encoding="utf-8"))
                        s_data["nodes"]["lenovo_rtx_5060"]["live_teacher_cycle"] = lesson_num
                        s_data["nodes"]["lenovo_rtx_5060"]["teacher_model"] = "Qwen2.5-Coder-1.5B-Instruct"
                        s_data["nodes"]["lenovo_rtx_5060"]["last_lesson"] = domanda
                        s_data["nodes"]["lenovo_rtx_5060"]["last_loss"] = round(trained_loss, 4)
                        s_data["nodes"]["lenovo_rtx_5060"]["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S+01:00")
                        status_file.write_text(json.dumps(s_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

            lesson_num += 1
            time.sleep(2.0)

        except KeyboardInterrupt:
            log_print("\nChiusura sessione addestramento.", log_fp)
            break
        except Exception as e:
            log_print(f"[!] Errore nel ciclo: {e}", log_fp)
            time.sleep(2.0)

if __name__ == "__main__":
    from qwen_verified_trainer import main as verified_main
    verified_main()
