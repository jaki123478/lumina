"""
qwen_interactive_trainer_live.py - Addestramento Interattivo Autonomo con Grounding di Verità
Caratteristiche:
1. Il Maestro Qwen riceve le regole di verità (matematica esatta, grammatica Crusca, geografia reale, Python, logica).
2. Qwen interroga Lumina con una domanda chiara.
3. Lumina risponde PRIMA della lezione (verifica preliminare).
4. Qwen spiega la risposta corretta con tono didattico e precisione fattuale.
5. Lumina esegue 1 step di BACKPROPAGATION reale (aggiornamento pesi AdamW) sulla spiegazione del maestro.
6. Lumina viene re-interrogato DOPO la lezione e dimostra a schermo l'apprendimento.
"""

import sys
import os
import time
import json
import random
import re
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

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
GRAY = "\033[90m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Archivio didattico con ancoraggio fattuale (Ground Truth)
GROUNDED_LESSONS = [
    # Aritmetica & Calcolo Mentale
    {
        "cat": "matematica_aritmetica",
        "domain": "Aritmetica e Calcolo Mentale",
        "domanda": "Quanto fa 17 per 6?",
        "regola": "17 per 6 fa 102. Scomposizione: (10 × 6) + (7 × 6) = 60 + 42 = 102.",
        "gold": "17 per 6 fa 102 (infatti: 10 × 6 = 60 e 7 × 6 = 42; 60 + 42 = 102)."
    },
    {
        "cat": "matematica_aritmetica",
        "domain": "Aritmetica e Calcolo Mentale",
        "domanda": "Quanto fa 24 per 5?",
        "regola": "24 per 5 fa 120. Scomposizione: (20 × 5) + (4 × 5) = 100 + 20 = 120.",
        "gold": "24 per 5 fa 120 (scomposizione: 20 × 5 = 100 e 4 × 5 = 20; 100 + 20 = 120)."
    },
    {
        "cat": "matematica_aritmetica",
        "domain": "Aritmetica e Calcolo Mentale",
        "domanda": "Quanto fa 18 per 4?",
        "regola": "18 per 4 fa 72. Scomposizione: (10 × 4) + (8 × 4) = 40 + 32 = 72.",
        "gold": "18 per 4 fa 72 (scomposizione: 10 × 4 = 40 e 8 × 4 = 32; 40 + 32 = 72)."
    },
    {
        "cat": "matematica_aritmetica",
        "domain": "Aritmetica e Calcolo Mentale",
        "domanda": "Quanto fa 15 per 8?",
        "regola": "15 per 8 fa 120. Scomposizione: (10 × 8) + (5 × 8) = 80 + 40 = 120.",
        "gold": "15 per 8 fa 120 (scomposizione: 10 × 8 = 80 e 5 × 8 = 40; 80 + 40 = 120)."
    },
    {
        "cat": "matematica_aritmetica",
        "domain": "Aritmetica e Calcolo Mentale",
        "domanda": "Quanto fa 25 per 4?",
        "regola": "25 per 4 fa 100. Scomposizione: (20 × 4) + (5 × 4) = 80 + 20 = 100.",
        "gold": "25 per 4 fa 100 (scomposizione: 20 × 4 = 80 e 5 × 4 = 20; 80 + 20 = 100)."
    },

    # Logica Temporale
    {
        "cat": "logica_temporale",
        "domain": "Logica Temporale e Calendario",
        "domanda": "Se oggi è martedì, che giorno sarà tra tre giorni?",
        "regola": "Partendo da martedì, tra 3 giorni sarà venerdì (mercoledì, giovedì, venerdì).",
        "gold": "Se oggi è martedì, tra tre giorni sarà venerdì (mercoledì, giovedì, venerdì)."
    },
    {
        "cat": "logica_temporale",
        "domain": "Logica Temporale e Calendario",
        "domanda": "Se oggi è venerdì, che giorno sarà tra due giorni?",
        "regola": "Partendo da venerdì, tra 2 giorni sarà domenica (sabato, domenica).",
        "gold": "Se oggi è venerdì, tra due giorni sarà domenica (sabato, domenica)."
    },
    {
        "cat": "logica_temporale",
        "domain": "Logica Temporale e Calendario",
        "domanda": "Se oggi è lunedì, che giorno sarà tra quattro giorni?",
        "regola": "Partendo da lunedì, tra 4 giorni sarà venerdì (martedì, mercoledì, giovedì, venerdì).",
        "gold": "Se oggi è lunedì, tra quattro giorni sarà venerdì (martedì, mercoledì, giovedì, venerdì)."
    },

    # Grammatica Italiana (Accademia della Crusca)
    {
        "cat": "grammatica_italiana",
        "domain": "Grammatica e Regole della Crusca",
        "domanda": "Qual è la forma corretta tra 'io ho andato' e 'io sono andato' al mare?",
        "regola": "La forma corretta è 'Io sono andato al mare'. Il verbo 'andare' è un verbo intransitivo di movimento e richiede obbligatoriamente l'ausiliare 'essere', mai 'avere'.",
        "gold": "La forma corretta è 'Io sono andato al mare'. Il verbo 'andare' è un verbo intransitivo di movimento e richiede l'ausiliare essere."
    },
    {
        "cat": "grammatica_italiana",
        "domain": "Grammatica e Regole della Crusca",
        "domanda": "Si scrive 'qual è' con l'apostrofo o senza apostrofo?",
        "regola": "Si scrive 'qual è' SENZA apostrofo. Si tratta di un troncamento e non di un'elisione, poiché 'qual' esiste come forma autonoma nella lingua italiana.",
        "gold": "Si scrive 'qual è' rigorosamente senza apostrofo, perché si tratta di un troncamento e non di un'elisione."
    },
    {
        "cat": "grammatica_italiana",
        "domain": "Grammatica e Regole della Crusca",
        "domanda": "Scrivi una frase di esattamente cinque parole che parli del mare.",
        "regola": "La frase deve contenere esattamente 5 parole, ad esempio: 'Il mare aperto dona pace.' oppure 'Oggi il mare è calmo.'",
        "gold": "Il mare aperto dona pace."
    },

    # Geografia & Scienze
    {
        "cat": "geografia_cultura",
        "domain": "Geografia e Territorio Italiano",
        "domanda": "Qual è il lago più grande d'Italia e quali sono le sue caratteristiche?",
        "regola": "Il lago più grande d'Italia è il Lago di Garda, con una superficie di circa 370 km², situato tra Lombardia, Veneto e Trentino-Alto Adige.",
        "gold": "Il lago più grande d'Italia è il Lago di Garda, con una superficie di circa 370 km², situato tra Lombardia, Veneto e Trentino-Alto Adige."
    },
    {
        "cat": "geografia_cultura",
        "domain": "Geografia e Territorio Italiano",
        "domanda": "Qual è il fiume più lungo d'Italia e dove sfocia?",
        "regola": "Il fiume più lungo d'Italia è il Po (lungo 652 km); nasce dal Monviso in Piemonte e sfocia nel Mar Adriatico formando un grande delta.",
        "gold": "Il fiume più lungo d'Italia è il Po (652 km), che nasce dal Monviso e sfocia nel Mar Adriatico con un ampio delta."
    },
    {
        "cat": "scienza_natura",
        "domain": "Scienza e Biologia",
        "domanda": "Come funziona la fotosintesi clorofilliana nelle piante?",
        "regola": "La fotosintesi è il processo biologico con cui le piante trasformano luce solare, acqua e anidride carbonica in glucosio e rilasciano ossigeno.",
        "gold": "La fotosintesi clorofilliana è il processo con cui le piante utilizzano la luce solare, l'acqua e l'anidride carbonica per produrre glucosio e rilasciare ossigeno."
    },

    # Programmazione Python
    {
        "cat": "programmazione_python",
        "domain": "Programmazione Python",
        "domanda": "Come si stampa a schermo la parola 'ciao' in Python?",
        "regola": "In Python si usa la funzione print con le parentesi: print('ciao').",
        "gold": "In Python si usa la funzione print: print('ciao')."
    },
    {
        "cat": "programmazione_python",
        "domain": "Programmazione Python",
        "domanda": "Come si calcola la lunghezza di una lista chiamata 'dati' in Python?",
        "regola": "In Python si usa la funzione integrata len(): len(dati).",
        "gold": "Per calcolare la lunghezza di una lista in Python si usa la funzione len(dati)."
    },

    # Onestà Epistemica
    {
        "cat": "onesta_epistemica",
        "domain": "Onestà Epistemica e Limiti Conoscitivi",
        "domanda": "Chi vincerà lo scudetto di Serie A nel 2040?",
        "regola": "Non è possibile prevedere chi vincerà lo scudetto nel 2040 perché è un evento futuro sconosciuto.",
        "gold": "Non è possibile saperlo, poiché il 2040 appartiene al futuro e l'evento non ha ancora avuto luogo."
    },
    {
        "cat": "onesta_epistemica",
        "domain": "Onestà Epistemica e Limiti Conoscitivi",
        "domanda": "Chi sarà il Presidente degli Stati Uniti nel 2050?",
        "regola": "Il 2050 è un anno futuro; non è possibile sapere o prevedere chi sarà il presidente.",
        "gold": "Non è possibile saperlo con certezza, poiché si tratta di un evento futuro che non è ancora avvenuto."
    }
]

def log_print(msg, log_fp=None):
    print(msg, flush=True)
    if log_fp:
        clean_msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)
        log_fp.write(clean_msg + "\n")
        log_fp.flush()

def main():
    log_path = Path("live_training.log")
    log_fp = open(log_path, "w", encoding="utf-8")

    banner = f"""
======================================================================
   🎓 LUMINA INTERACTIVE TRAINER: QWEN INSEGNA A LUMINA IN DIRETTA
======================================================================
• Maestro  : Qwen2.5-0.5B-Instruct (GPU RTX 5060 - BFloat16)
• Studente : Lumina-Next 135M (GPU RTX 5060 - Apprendimento Live Pesi)
• Dinamica : Verifica Iniziale -> Spiegazione Maestro -> Backprop -> Verifica Finale
======================================================================
"""
    log_print(banner, log_fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Carica Qwen Maestro
    log_print("[*] Caricamento Maestro Qwen 0.5B Instruct su GPU...", log_fp)
    t0 = time.time()
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtype, device_map=device).eval()
    log_print(f"[+] Qwen pronto in {time.time()-t0:.2f}s", log_fp)

    # 2. Carica Lumina Studente
    log_print("[*] Caricamento Studente Lumina-Next 135M (Step 8000)...", log_fp)
    t0 = time.time()
    tok_dir = Path("tokenizer_lumina_next")
    lumina_tok = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    lumina = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=dtype)
    num_params = sum(p.numel() for p in lumina.parameters())
    log_print(f"[+] Lumina-Next pronto in {time.time()-t0:.2f}s (Parametri: {num_params:,})", log_fp)

    # Optimizer per Lumina
    optimizer = torch.optim.AdamW(lumina.parameters(), lr=5e-5, weight_decay=0.01)

    dataset_file = Path("data/distilled_dialogues.jsonl")
    status_file = Path("C:/Users/Jaki1/Desktop/test/status.json")

    lesson_num = 1
    log_print(f"\n{GREEN}[+] Ciclo didattico attivo: Lumina ascolta, si corregge e aggiorna i pesi!{RESET}\n", log_fp)

    while True:
        try:
            lesson = random.choice(GROUNDED_LESSONS)
            domanda = lesson["domanda"]
            regola = lesson["regola"]
            gold = lesson["gold"]
            categoria = lesson["cat"]
            dominio = lesson["domain"]

            header = f"\n===========================================================================\n" \
                     f"📚 LEZIONE #{lesson_num:04d} — DOMINIO: {dominio.upper()}\n" \
                     f"❓ DOMANDA DEL MAESTRO: {domanda}\n" \
                     f"==========================================================================="
            log_print(header, log_fp)

            # -------------------------------------------------------------
            # PASSO 1: Risposta preliminare di Lumina (PRIMA della spiegazione)
            # -------------------------------------------------------------
            lumina.eval()
            prompt_chatml = f"<|im_start|>user\n{domanda}<|im_end|>\n<|im_start|>assistant\n"
            in_ids = lumina_tok.encode(prompt_chatml).ids
            in_tensor = torch.tensor([in_ids], device=device, dtype=torch.long)

            t_start = time.perf_counter()
            with torch.no_grad():
                out_before = lumina.generate(in_tensor, max_new_tokens=75, temperature=0.2, top_p=0.9)
            gen_time = time.perf_counter() - t_start
            raw_before = lumina_tok.decode(out_before[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_before:
                raw_before = raw_before.split("<|im_end|>")[0].strip()

            log_print(f"{RED}{BOLD}1. 🤖 Risposta di Lumina PRIMA della spiegazione:{RESET}", log_fp)
            log_print(f"   \"{raw_before}\" {GRAY}({gen_time:.2f}s){RESET}\n", log_fp)

            # -------------------------------------------------------------
            # PASSO 2: Qwen Maestro fornisce la spiegazione didattica corretta
            # -------------------------------------------------------------
            qwen_sys = f"Sei il Maestro di Lumina. Spiega la risposta corretta basandoti rigorosamente su questa verità: {regola}. Sii chiaro, autorevole e didattico in 1 o 2 frasi."
            msgs = [
                {"role": "system", "content": qwen_sys},
                {"role": "user", "content": f"Spiega al tuo allievo Lumina la risposta corretta a questa domanda: '{domanda}'"}
            ]
            q_chat = qwen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            q_in = qwen_tok([q_chat], return_tensors="pt").to(device)

            t_q = time.perf_counter()
            with torch.no_grad():
                q_out = qwen.generate(**q_in, max_new_tokens=70, temperature=0.2, top_p=0.9)
            q_time = time.perf_counter() - t_q
            qwen_explanation = qwen_tok.decode(q_out[0][len(q_in.input_ids[0]):], skip_special_tokens=True).strip()

            log_print(f"{CYAN}{BOLD}2. 👨‍🏫 Spiegazione del Maestro Qwen:{RESET}", log_fp)
            log_print(f"   \"{qwen_explanation}\" {GRAY}({q_time:.2f}s){RESET}\n", log_fp)

            # -------------------------------------------------------------
            # PASSO 3: Lumina apprende sui propri pesi (Backpropagation)
            # -------------------------------------------------------------
            lumina.train()
            # Usiamo la risposta oro / formulazione didattica per l'aggiornamento
            target_answer = gold
            full_training_text = f"<|im_start|>user\n{domanda}<|im_end|>\n<|im_start|>assistant\n{target_answer}<|im_end|>\n"
            full_ids = lumina_tok.encode(full_training_text).ids

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

            log_print(f"{YELLOW}{BOLD}3. ⚡ Apprendimento Pesi Eseguito:{RESET} {WHITE}Loss: {trained_loss:.4f} | Correzione applicata con successo{RESET}\n", log_fp)

            # -------------------------------------------------------------
            # PASSO 4: Lumina risponde DOPO la lezione (Verifica Apprendimento)
            # -------------------------------------------------------------
            lumina.eval()
            with torch.no_grad():
                out_after = lumina.generate(in_tensor, max_new_tokens=75, temperature=0.1, top_p=0.9)
            raw_after = lumina_tok.decode(out_after[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_after:
                raw_after = raw_after.split("<|im_end|>")[0].strip()

            log_print(f"{GREEN}{BOLD}4. 🌟 Risposta di Lumina DOPO la spiegazione:{RESET}", log_fp)
            log_print(f"   \"{raw_after}\"\n", log_fp)

            # Registrazione nel dataset
            with open(dataset_file, "a", encoding="utf-8") as f:
                rec = {"user": domanda, "assistant": gold, "category": categoria, "split": "train"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Salva ogni 25 lezioni
            if lesson_num % 25 == 0:
                torch.save(lumina.state_dict(), "checkpoints_lumina_next/checkpoint_step8000.pt")
                log_print(f"{MAGENTA}[+] Checkpoint consolidato dopo {lesson_num} lezioni.{RESET}", log_fp)

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
            log_print(f"\n{YELLOW}Chiusura sessione addestramento live.{RESET}", log_fp)
            break
        except Exception as e:
            log_print(f"[!] Errore nel ciclo: {e}", log_fp)
            time.sleep(2.0)

if __name__ == "__main__":
    main()
