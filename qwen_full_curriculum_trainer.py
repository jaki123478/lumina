"""
qwen_full_curriculum_trainer.py - Addestramento Multidisciplinare con Forte Focus su Coding & Computer Science
Ambiti:
1. Coding Python & Software Engineering (50% del programma)
2. Algoritmi e Strutture Dati
3. Web, API e Database SQL
4. Git e Controllo Versione
5. AI e Reti Neurali
6. Storia, Geografia, Scienze, Spazio, Arte, Letteratura ed Etica
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

CURRICULUM_DATA = [
    # -------------------------------------------------------------
    # CODING PYTHON & INGEGNERIA SOFTWARE
    # -------------------------------------------------------------
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Come si gestisce un'eccezione KeyError quando si accede a un dizionario in Python?",
        "sys": "Sei un senior software engineer. Spiega brevemente come gestire un KeyError con try/except o con il metodo .get() mostrando un esempio sintetico di codice Python."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Qual e' la differenza tra una lista mutabile e una tupla immutabile in Python?",
        "sys": "Sei un programmatore Python esperto. Spiega la differenza chiave tra liste e tuple e quando usare ciascuna in modo chiaro e pratico."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Come funziona una list comprehension in Python rispetto a un tradizionale ciclo for?",
        "sys": "Sei un docente di programmazione. Spiega come la list comprehension rende il codice piu' conciso e leggibile mostrando la sintassi di base [x for x in lista]."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Perche' in Python e' buona pratica aprire i file usando l'istruzione 'with open(...) as f'?",
        "sys": "Sei un esperto di Python. Spiega il concetto di context manager e la chiusura automatica del file anche in caso di errore."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Come si definisce una classe con costruttore __init__ e attributo d'istanza in Python?",
        "sys": "Sei un programmatore OOP. Spiega il ruolo del metodo __init__ e del parametro self mostrando una classe minimale Persona."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Cosa sono le f-string in Python e come si usano per interpolare variabili?",
        "sys": "Sei un docente di Python. Spiega la formattazione con f'Testo {variabile}' e perche' e' preferibile rispetto al vecchio operatore % o a .format()."
    },
    {
        "cat": "coding_python",
        "domain": "Coding: Python & Sintassi",
        "question": "Come si gestisce una divisione per zero con un blocco try ed except ZeroDivisionError?",
        "sys": "Sei uno sviluppatore software. Mostra la struttura corretta di try ed except ZeroDivisionError per evitare il crash del programma."
    },

    # -------------------------------------------------------------
    # ALGORITMI & STRUTTURE DATI
    # -------------------------------------------------------------
    {
        "cat": "algoritmi",
        "domain": "Algoritmi e Strutture Dati",
        "question": "Come funziona l'algoritmo di Ricerca Binaria e qual e' la sua complessita' temporale?",
        "sys": "Sei un docente di algoritmi. Spiega la ricerca binaria, la precondizione di array ordinato e la complessita' logaritmica O(log n)."
    },
    {
        "cat": "algoritmi",
        "domain": "Algoritmi e Strutture Dati",
        "question": "Qual e' la differenza tra una Pila (Stack LIFO) e una Coda (Queue FIFO)?",
        "sys": "Sei un docente di informatica. Spiega Last-In First-Out vs First-In First-Out con esempi concreti della vita reale."
    },
    {
        "cat": "algoritmi",
        "domain": "Algoritmi e Strutture Dati",
        "question": "Qual e' la differenza di complessita' tra un algoritmo BubbleSort O(n^2) e un MergeSort O(n log n)?",
        "sys": "Sei un analista di algoritmi. Confronta la scalabilita' su grandi quantita' di dati tra complessita' quadratica e lineare-logaritmica."
    },
    {
        "cat": "algoritmi",
        "domain": "Algoritmi e Strutture Dati",
        "question": "Come si calcola la sequenza di Fibonacci in modo ricorsivo e perche' la memoization la rende piu' efficiente?",
        "sys": "Sei un docente di informatica. Spiega la ricorsione di Fibonacci e come memorizzare i risultati intermedi evita calcoli ripetuti."
    },

    # -------------------------------------------------------------
    # WEB, API, DATABASE & GIT
    # -------------------------------------------------------------
    {
        "cat": "web_database_git",
        "domain": "Web, API e Database",
        "question": "Qual e' la differenza tra una richiesta HTTP di tipo GET e una di tipo POST?",
        "sys": "Sei un backend engineer. Spiega la differenza tra recuperare risorse (GET) e inviare dati con payload nel corpo della richiesta (POST)."
    },
    {
        "cat": "web_database_git",
        "domain": "Web, API e Database",
        "question": "Cosa significano i codici di stato HTTP 200, 404 e 500?",
        "sys": "Sei uno sviluppatore web. Spiega 200 OK (successo), 404 Not Found (risorsa non trovata) e 500 Internal Server Error (errore lato server)."
    },
    {
        "cat": "web_database_git",
        "domain": "Web, API e Database",
        "question": "Cosa e' il formato JSON e come si converte una stringa JSON in dizionario Python?",
        "sys": "Sei un software engineer. Spiega la funzione json.loads() per deserializzare una stringa JSON in strutture Python native."
    },
    {
        "cat": "web_database_git",
        "domain": "Web, API e Database",
        "question": "In un database SQL, qual e' la differenza tra una Primary Key e una Foreign Key?",
        "sys": "Sei un database architect. Spiega l'identificatore univoco di una riga (Primary Key) e il vincolo di relazione verso un'altra tabella (Foreign Key)."
    },
    {
        "cat": "web_database_git",
        "domain": "Controllo Versione: Git",
        "question": "Qual e' la differenza tra il comando 'git commit' e il comando 'git push'?",
        "sys": "Sei un DevOps engineer. Spiega il salvataggio locale della cronologia (commit) e l'invio delle modifiche al server remoto (push)."
    },

    # -------------------------------------------------------------
    # AI E RETI NEURALI
    # -------------------------------------------------------------
    {
        "cat": "intelligenza_artificiale",
        "domain": "AI & Reti Neurali",
        "question": "Come funziona la Backpropagation per aggiornare i pesi di una rete neurale?",
        "sys": "Sei un ricercatore di machine learning. Spiega come l'errore (Loss) viene propagato all'indietro calcolando i gradienti per la discesa del gradiente."
    },
    {
        "cat": "intelligenza_artificiale",
        "domain": "AI & Reti Neurali",
        "question": "Cosa e' l'attenzione multi-testa (Multi-Head Attention) nei modelli Transformer?",
        "sys": "Sei un esperto di LLM. Spiega come l'attenzione consente al modello di focalizzarsi su relazioni tra parole a diverse distanze simultaneamente."
    },

    # -------------------------------------------------------------
    # STORIA, SCIENZA, SPAZIO & CULTURA
    # -------------------------------------------------------------
    {
        "cat": "scienze_spazio",
        "domain": "Scienze & Astronomia",
        "question": "Come funziona la fotosintesi clorofilliana nelle piante?",
        "sys": "Sei un docente di scienze. Spiega come le piante trasformano luce solare, acqua e CO2 in glucosio e rilasciano ossigeno."
    },
    {
        "cat": "scienze_spazio",
        "domain": "Scienze & Astronomia",
        "question": "Cosa e' un buco nero e come si forma nello spazio?",
        "sys": "Sei un astrofisico divulgatore. Spiega il collasso gravitazionale di una stella massiccia e l'orizzonte degli eventi in modo chiaro."
    },
    {
        "cat": "storia_cultura",
        "domain": "Storia & Cultura Generale",
        "question": "Cosa e' stato il Rinascimento italiano e perche' e' famoso nel mondo?",
        "sys": "Sei uno storico. Spiega la rinascita delle arti, delle lettere e delle scienze a Firenze e in Italia tra il XIV e il XVI secolo."
    },
    {
        "cat": "storia_cultura",
        "domain": "Storia & Cultura Generale",
        "question": "Chi era Leonardo da Vinci e quali sono le sue opere e innovazioni principali?",
        "sys": "Sei un divulgatore d'arte. Spiega il genio poliedrico di Leonardo tra pittura (Gioconda, Cenacolo) e studi scientifico-ingegneristici."
    },
    {
        "cat": "lingua_etica",
        "domain": "Lingua Italiana & Etica",
        "question": "Perche' si scrive 'qual e'' senza apostrofo secondo l'Accademia della Crusca?",
        "sys": "Sei un linguista della Crusca. Spiega che si tratta di un troncamento e non di un'elisione, per cui l'apostrofo e' un errore."
    },
    {
        "cat": "lingua_etica",
        "domain": "Lingua Italiana & Etica",
        "question": "Perche' e' fondamentale per un'intelligenza artificiale dichiarare con onesta' i propri limiti di conoscenza?",
        "sys": "Sei un eticista dell'IA. Spiega perche' la trasparenza e l'onesta' epistemica evitano disinformazione e allucinazioni."
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
   LUMINA AI: SESSIONE LIVE MULTIDISCIPLINARE (FOCUS CODING & SCIENZA)
========================================================================
- Maestro  : Qwen2.5-0.5B-Instruct (GPU RTX 5060 - BFloat16)
- Studente : Lumina-Next 135M (GPU RTX 5060 - Training Pesi Attivo)
- Materie  : Python, Algoritmi, Web/API, Database SQL, Git, AI, Storia, Scienze
- Flusso   : 1. Domanda -> 2. Lumina (Prima) -> 3. Qwen Maestro -> 4. Backprop -> 5. Lumina (Dopo)
========================================================================
"""
    log_print(banner, log_fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Caricamento Qwen
    log_print("[*] Caricamento Maestro Qwen 0.5B su GPU...", log_fp)
    t0 = time.time()
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtype, device_map=device).eval()
    log_print(f"[+] Qwen pronto in {time.time()-t0:.2f}s", log_fp)

    # 2. Caricamento Lumina
    log_print("[*] Caricamento Studente Lumina-Next 135M...", log_fp)
    t0 = time.time()
    tok_dir = Path("tokenizer_lumina_next")
    lumina_tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    lumina = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=dtype)
    num_params = sum(p.numel() for p in lumina.parameters())
    log_print(f"[+] Lumina-Next pronto in {time.time()-t0:.2f}s (Parametri: {num_params:,})", log_fp)

    optimizer = torch.optim.AdamW(lumina.parameters(), lr=3.5e-5, weight_decay=0.01)

    dataset_file = Path("data/distilled_dialogues.jsonl")
    status_file = Path("C:/Users/Jaki1/Desktop/test/status.json")

    # Coda non ripetitiva circolare
    deck = list(CURRICULUM_DATA)
    random.shuffle(deck)
    deck_idx = 0
    lesson_num = 1

    log_print("\n[+] Ciclo didattico avviato con successo: lezioni in tempo reale a schermo!\n", log_fp)

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
                out_before = lumina.generate(in_tensor, max_new_tokens=70, temperature=0.3, top_p=0.9)
            el_lum = time.perf_counter() - t_lum
            raw_before = lumina_tok.decode(out_before[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_before:
                raw_before = raw_before.split("<|im_end|>")[0].strip()

            log_print(f"\n[1. LUMINA - PRIMA DELLA LEZIONE]:", log_fp)
            log_print(f"   \"{raw_before}\" (latenza: {el_lum:.2f}s)", log_fp)

            # -------------------------------------------------------------
            # PASSO 2: Spiegazione didattica di Qwen Maestro
            # -------------------------------------------------------------
            msgs = [
                {"role": "system", "content": sys_instruct},
                {"role": "user", "content": f"Spiega in modo pratico, istruttivo e chiaro: {domanda}"}
            ]
            q_chat = qwen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            q_in = qwen_tok([q_chat], return_tensors="pt").to(device)

            t_q = time.perf_counter()
            with torch.no_grad():
                q_out = qwen.generate(**q_in, max_new_tokens=85, temperature=0.2, top_p=0.9)
            el_q = time.perf_counter() - t_q
            qwen_answer = qwen_tok.decode(q_out[0][len(q_in.input_ids[0]):], skip_special_tokens=True).strip()

            log_print(f"\n[2. QWEN MAESTRO - SPIEGAZIONE]:", log_fp)
            log_print(f"   \"{qwen_answer}\" (latenza: {el_q:.2f}s)", log_fp)

            # -------------------------------------------------------------
            # PASSO 3: Aggiornamento pesi di Lumina (Backpropagation)
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

            # Salva nel dataset
            with open(dataset_file, "a", encoding="utf-8") as f:
                rec = {"user": domanda, "assistant": qwen_answer, "category": categoria, "split": "train"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Salva ogni 25 lezioni
            if lesson_num % 25 == 0:
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
            time.sleep(2.5)

        except KeyboardInterrupt:
            log_print("\nChiusura sessione addestramento.", log_fp)
            break
        except Exception as e:
            log_print(f"[!] Errore nel ciclo: {e}", log_fp)
            time.sleep(2.0)

if __name__ == "__main__":
    main()
