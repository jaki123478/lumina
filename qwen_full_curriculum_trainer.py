"""
qwen_full_curriculum_trainer.py - Addestramento Multidisciplinare a 360 Gradi Qwen -> Lumina
Campi coperti:
1. Storia e Grandi Figure Storiche
2. Geografia Italiana e Mondiale
3. Scienze Naturali, Biologia e Corpo Umano
4. Astronomia, Spazio e Cosmo
5. Fisica e Chimica della Vita Quotidiana
6. Informatica, Internet, Algoritmi e AI
7. Arte, Pittura, Architettura e Scultura
8. Letteratura, Poesia e Lingua Italiana
9. Cucina Tradizionale e Cultura Gastronomica
10. Ecologia, Animali e Natura
11. Filosofia, Etica e Pensiero Critico
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

ALL_DOMAINS_LESSONS = [
    # 1. STORIA E GRANDI FIGURE
    {
        "cat": "storia",
        "domain": "Storia e Grandi Figure",
        "question": "Chi era Giulio Cesare e quale ruolo ha avuto nella storia romana?"
    },
    {
        "cat": "storia",
        "domain": "Storia e Grandi Figure",
        "question": "Cosa e' stato il Rinascimento italiano e in quale citta' e' nato?"
    },
    {
        "cat": "storia",
        "domain": "Storia e Grandi Figure",
        "question": "Chi era Leonardo da Vinci e quali sono le sue opere e invenzioni piu' note?"
    },
    {
        "cat": "storia",
        "domain": "Storia e Grandi Figure",
        "question": "Come e' avvenuta l'Unita' d'Italia e chi sono stati i suoi protagonisti?"
    },
    {
        "cat": "storia",
        "domain": "Storia e Grandi Figure",
        "question": "Chi erano gli antichi Etruschi e dove vivevano nella penisola italiana?"
    },

    # 2. GEOGRAFIA ITALIANA E MONDIALE
    {
        "cat": "geografia",
        "domain": "Geografia e Territorio",
        "question": "Qual e' il lago piu' grande d'Italia e in quali regioni si estende?"
    },
    {
        "cat": "geografia",
        "domain": "Geografia e Territorio",
        "question": "Qual e' il fiume piu' lungo d'Italia, da dove nasce e dove sfocia?"
    },
    {
        "cat": "geografia",
        "domain": "Geografia e Territorio",
        "question": "Quali sono i vulcani attivi piu' famosi d'Italia?"
    },
    {
        "cat": "geografia",
        "domain": "Geografia e Territorio",
        "question": "Quali sono le caratteristiche principali della catena montuosa delle Alpi?"
    },
    {
        "cat": "geografia",
        "domain": "Geografia e Territorio",
        "question": "Qual e' la capitale del Giappone e quali sono le sue caratteristiche principali?"
    },

    # 3. SCIENZE NATURALI E BIOLOGIA
    {
        "cat": "scienze_biologia",
        "domain": "Scienze Naturali e Biologia",
        "question": "Come funziona la fotosintesi clorofilliana e perche' e' fondamentale per la vita?"
    },
    {
        "cat": "scienze_biologia",
        "domain": "Scienze Naturali e Biologia",
        "question": "Come funziona il sistema cardiocircolatorio nel corpo umano?"
    },
    {
        "cat": "scienze_biologia",
        "domain": "Scienze Naturali e Biologia",
        "question": "Qual e' la differenza biologica tra virus e batteri?"
    },
    {
        "cat": "scienze_biologia",
        "domain": "Scienze Naturali e Biologia",
        "question": "Come respirano i pesci attraverso le branchie sott'acqua?"
    },
    {
        "cat": "scienze_biologia",
        "domain": "Scienze Naturali e Biologia",
        "question": "A cosa serve il DNA negli esseri viventi?"
    },

    # 4. ASTRONOMIA E SPAZIO
    {
        "cat": "astronomia",
        "domain": "Astronomia e Spazio",
        "question": "Come nascono le stelle all'interno delle nebulose nell'universo?"
    },
    {
        "cat": "astronomia",
        "domain": "Astronomia e Spazio",
        "question": "Quali sono i pianeti del Sistema Solare in ordine dal Sole?"
    },
    {
        "cat": "astronomia",
        "domain": "Astronomia e Spazio",
        "question": "Perche' la Luna mostra sempre la stessa faccia rivolta verso la Terra?"
    },
    {
        "cat": "astronomia",
        "domain": "Astronomia e Spazio",
        "question": "Cosa e' un buco nero e come si forma nello spazio profondo?"
    },

    # 5. FISICA E CHIMICA QUOTIDIANA
    {
        "cat": "fisica_chimica",
        "domain": "Fisica e Chimica",
        "question": "Perche' il cielo appare di colore azzurro durante il giorno?"
    },
    {
        "cat": "fisica_chimica",
        "domain": "Fisica e Chimica",
        "question": "Come fanno gli aerei a volare grazie al principio di portanza?"
    },
    {
        "cat": "fisica_chimica",
        "domain": "Fisica e Chimica",
        "question": "Perche' l'olio galleggia sull'acqua invece di mescolarsi?"
    },
    {
        "cat": "fisica_chimica",
        "domain": "Fisica e Chimica",
        "question": "Quali sono i tre stati principali della materia e come avvengono i passaggi di stato?"
    },

    # 6. INFORMATICA, INTERNET E TECNOLOGIA
    {
        "cat": "informatica",
        "domain": "Informatica e Tecnologia",
        "question": "Cosa e' un algoritmo in informatica e perche' e' cosi' importante?"
    },
    {
        "cat": "informatica",
        "domain": "Informatica e Tecnologia",
        "question": "Come funziona la rete Internet per collegare computer in tutto il mondo?"
    },
    {
        "cat": "informatica",
        "domain": "Informatica e Tecnologia",
        "question": "Qual e' la differenza tra la memoria RAM e il disco rigido (SSD o Hard Disk)?"
    },
    {
        "cat": "informatica",
        "domain": "Informatica e Tecnologia",
        "question": "A cosa serve il linguaggio di programmazione Python e perche' e' tanto diffuso?"
    },
    {
        "cat": "informatica",
        "domain": "Informatica e Tecnologia",
        "question": "Come funziona una rete neurale artificiale nell'intelligenza artificiale?"
    },

    # 7. ARTE, PITTURA E ARCHITETTURA
    {
        "cat": "arte",
        "domain": "Arte e Architettura",
        "question": "Chi ha affrescato la volta della Cappella Sistina a Roma e cosa vi e' raffigurato?"
    },
    {
        "cat": "arte",
        "domain": "Arte e Architettura",
        "question": "Cosa caratterizza lo stile architettonico gotico nelle cattedrali europee?"
    },
    {
        "cat": "arte",
        "domain": "Arte e Architettura",
        "question": "Chi era Caravaggio e quale tecnica di pittura basata sulla luce lo ha reso celebre?"
    },
    {
        "cat": "arte",
        "domain": "Arte e Architettura",
        "question": "Quali sono le caratteristiche principali del Colosseo a Roma?"
    },

    # 8. LETTERATURA E LINGUA ITALIANA
    {
        "cat": "letteratura",
        "domain": "Letteratura e Lingua Italiana",
        "question": "Di cosa parla la Divina Commedia di Dante Alighieri e come e' strutturata?"
    },
    {
        "cat": "letteratura",
        "domain": "Letteratura e Lingua Italiana",
        "question": "Chi era Giacomo Leopardi e quali sono i temi principali della sua poesia 'L'Infinito'?"
    },
    {
        "cat": "letteratura",
        "domain": "Letteratura e Lingua Italiana",
        "question": "Chi ha scritto 'I Promessi Sposi' e perche' questo romanzo e' fondamentale per l'italiano moderno?"
    },
    {
        "cat": "letteratura",
        "domain": "Letteratura e Lingua Italiana",
        "question": "Cosa significa il modo di dire italiano 'mettere le carte in tavola'?"
    },
    {
        "cat": "letteratura",
        "domain": "Letteratura e Lingua Italiana",
        "question": "Perche' si scrive 'qual e'' senza apostrofo secondo l'Accademia della Crusca?"
    },

    # 9. CUCINA E TRADIZIONI GASTRONOMICHE
    {
        "cat": "cucina",
        "domain": "Cucina e Cultura Gastronomica",
        "question": "Quali sono gli ingredienti tradizionali della vera pizza Margherita napoletana?"
    },
    {
        "cat": "cucina",
        "domain": "Cucina e Cultura Gastronomica",
        "question": "Come si prepara un classico risotto allo zafferano alla milanese?"
    },
    {
        "cat": "cucina",
        "domain": "Cucina e Cultura Gastronomica",
        "question": "Quali sono le caratteristiche dell'olio extravergine d'oliva italiano?"
    },
    {
        "cat": "cucina",
        "domain": "Cucina e Cultura Gastronomica",
        "question": "Qual e' l'origine e la ricetta tradizionale del tiramisu'?"
    },

    # 10. ECOLOGIA, ANIMALI E NATURA
    {
        "cat": "ecologia_natura",
        "domain": "Ecologia, Animali e Natura",
        "question": "Perche' le api sono insetti indispensabili per l'equilibrio dell'ecosistema?"
    },
    {
        "cat": "ecologia_natura",
        "domain": "Ecologia, Animali e Natura",
        "question": "Come funziona il ciclo dell'acqua sul nostro pianeta?"
    },
    {
        "cat": "ecologia_natura",
        "domain": "Ecologia, Animali e Natura",
        "question": "Perche' le foreste pluviali vengono chiamate i 'polmoni verdi' della Terra?"
    },
    {
        "cat": "ecologia_natura",
        "domain": "Ecologia, Animali e Natura",
        "question": "Come fanno gli uccelli migratori a orientarsi durante i lunghi viaggi stagionali?"
    },

    # 11. FILOSOFIA, ETICA E RAGIONAMENTO
    {
        "cat": "filosofia_etica",
        "domain": "Filosofia, Etica e Logica",
        "question": "Cosa significa il celebre motto socratico 'so di non sapere'?"
    },
    {
        "cat": "filosofia_etica",
        "domain": "Filosofia, Etica e Logica",
        "question": "Qual e' la differenza tra un fatto oggettivo e un'opinione personale?"
    },
    {
        "cat": "filosofia_etica",
        "domain": "Filosofia, Etica e Logica",
        "question": "Cosa significa pensare in modo critico di fronte alle informazioni che riceviamo?"
    },
    {
        "cat": "filosofia_etica",
        "domain": "Filosofia, Etica e Logica",
        "question": "Perche' e' importante per un'intelligenza artificiale dichiarare i propri limiti di conoscenza?"
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
======================================================================
  [LUMINA LIVE MULTIDISCIPLINARE: QWEN INSEGNA A LUMINA SU TUTTO]
======================================================================
- Maestro  : Qwen2.5-0.5B-Instruct (GPU RTX 5060 - BFloat16)
- Studente : Lumina-Next 135M (GPU RTX 5060 - Apprendimento Pesi Live)
- Ambiti   : Storia, Geografia, Scienze, Astronomia, Tecnologia, Arte,
             Letteratura, Lingua Italiana, Cucina, Ecologia, Filosofia
- Flusso   : Risposta Lumina -> Insegnamento Qwen -> Backprop -> Verifica
======================================================================
"""
    log_print(banner, log_fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Caricamento Qwen
    log_print("[*] Caricamento Maestro Qwen 0.5B Instruct su GPU...", log_fp)
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

    # Coda non ripetitiva (shuffle circolare per evitare doppioni)
    deck = list(ALL_DOMAINS_LESSONS)
    random.shuffle(deck)
    deck_idx = 0
    lesson_num = 1

    log_print("\n[+] Avvio delle lezioni multidisciplinari su tutti i domini dello scibile!\n", log_fp)

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

            separator = "=" * 70
            log_print(f"\n{separator}", log_fp)
            log_print(f"[LEZIONE #{lesson_num:04d}] AMBITO: {dominio.upper()}", log_fp)
            log_print(f"DOMANDA DEL MAESTRO: {domanda}", log_fp)
            log_print(f"{separator}", log_fp)

            # -----------------------------------------------------------------
            # 1. LUMINA RISPONDE PRIMA DELLA LEZIONE
            # -----------------------------------------------------------------
            lumina.eval()
            prompt_chatml = f"<|im_start|>user\n{domanda}<|im_end|>\n<|im_start|>assistant\n"
            in_ids = lumina_tok.encode(prompt_chatml).ids
            in_tensor = torch.tensor([in_ids], device=device, dtype=torch.long)

            t_lum = time.perf_counter()
            with torch.no_grad():
                out_before = lumina.generate(in_tensor, max_new_tokens=75, temperature=0.3, top_p=0.9)
            el_lum = time.perf_counter() - t_lum
            raw_before = lumina_tok.decode(out_before[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_before:
                raw_before = raw_before.split("<|im_end|>")[0].strip()

            log_print("\n[1. LUMINA STUDENTE - PRIMA DELLA LEZIONE]:", log_fp)
            log_print(f"   \"{raw_before}\" (tempo: {el_lum:.2f}s)", log_fp)

            # -----------------------------------------------------------------
            # 2. QWEN MAESTRO INSEGNA E SPIEGA
            # -----------------------------------------------------------------
            sys_instruct = "Sei un professore enciclopedico e divulgatore italiano. Spiega il concetto richiesto in modo chiaro, istruttivo, naturale ed esaustivo in 2 o 3 frasi corrette in italiano."
            msgs = [
                {"role": "system", "content": sys_instruct},
                {"role": "user", "content": f"Spiega in modo didattico e chiaro: {domanda}"}
            ]
            q_chat = qwen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            q_in = qwen_tok([q_chat], return_tensors="pt").to(device)

            t_q = time.perf_counter()
            with torch.no_grad():
                q_out = qwen.generate(**q_in, max_new_tokens=85, temperature=0.3, top_p=0.9)
            el_q = time.perf_counter() - t_q
            qwen_answer = qwen_tok.decode(q_out[0][len(q_in.input_ids[0]):], skip_special_tokens=True).strip()

            log_print("\n[2. QWEN MAESTRO - SPIEGAZIONE DIDATTICA]:", log_fp)
            log_print(f"   \"{qwen_answer}\" (tempo: {el_q:.2f}s)", log_fp)

            # -----------------------------------------------------------------
            # 3. APPRENDIMENTO SUI PESI DI LUMINA (BACKPROPAGATION)
            # -----------------------------------------------------------------
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

            log_print("\n[3. APPRENDIMENTO ATTIVO SUI PESI]:", log_fp)
            log_print(f"   Loss: {trained_loss:.4f} | Aggiornamento gradiente applicato su GPU RTX 5060", log_fp)

            # -----------------------------------------------------------------
            # 4. LUMINA RISPONDE DOPO LA LEZIONE
            # -----------------------------------------------------------------
            lumina.eval()
            with torch.no_grad():
                out_after = lumina.generate(in_tensor, max_new_tokens=75, temperature=0.2, top_p=0.9)
            raw_after = lumina_tok.decode(out_after[0][len(in_ids):].tolist()).strip()
            if "<|im_end|>" in raw_after:
                raw_after = raw_after.split("<|im_end|>")[0].strip()

            log_print("\n[4. LUMINA STUDENTE - DOPO LA LEZIONE (CONCETTO APPRESO)]:", log_fp)
            log_print(f"   \"{raw_after}\"\n", log_fp)

            # Salva nel dataset
            with open(dataset_file, "a", encoding="utf-8") as f:
                rec = {"user": domanda, "assistant": qwen_answer, "category": categoria, "split": "train"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Salva checkpoint ogni 25 lezioni
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
