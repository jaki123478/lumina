"""
auto_distill_train.py - Pipeline continua di distillazione e addestramento automatico.

Qwen (LLM Maestro su GPU RTX 5060) genera continuamente dialoghi e Q&A italiani.
Ogni 30 nuovi esempi, Lumina (Modello Studente) viene addestrato automaticamente
sui nuovi dati e i pesi aggiornati vengono esportati per il browser.

Può girare ininterrottamente per ore o tutta la notte.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import CONFIG, PRESETS, Lumina50K, encode, load_model, prompt_tokens
from train import batch, evaluate, export

def save_model(model: Lumina50K, pt_path: Path, export_path: Path, train_loss: float, val_loss: float, steps: int):
    metadata = {
        "origin": "Qwen distillation + base dialogues",
        "steps": steps,
        "train_loss": round(train_loss, 4),
        "validation_loss": round(val_loss, 4),
        "timestamp": time.time(),
    }
    checkpoint = {"config": model.config, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "training": metadata}
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, pt_path)
    export(model, metadata, export_path)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DISTILLED_FILE = DATA_DIR / "distilled_dialogues.jsonl"
BASE_DIALOGUES_FILE = DATA_DIR / "dialogues.jsonl"
STATUS_FILE = DATA_DIR / "auto_train_status.json"

CATEGORIES = [
    # --- DOMINI PRIORITARI LUMINA TEACHER ---
    {
        "category": "grammatica_ortografia_italiana",
        "domain": "Grammatica e ortografia italiana avanzata",
        "instruction": "Genera 1 esempio di training didattico e denso su congiuntivi, accenti, apostrofi, punteggiatura o regole ortografiche italiane.",
        "example_user": "Quando si usa 'qual è' con o senza apostrofo?",
        "example_bot": "'Qual è' si scrive sempre senza apostrofo, perché si tratta di un troncamento e non di un'elisione."
    },
    {
        "category": "burocrazia_normativa_italiana",
        "domain": "Burocrazia e normativa italiana semplificata",
        "instruction": "Genera 1 esempio di training didattico su concetti chiave della burocrazia italiana (SPID, CIE, ISEE, PEC, marca da bollo, 730).",
        "example_user": "A cosa serve lo SPID?",
        "example_bot": "Lo SPID è l'identità digitale unica per accedere a tutti i servizi online della Pubblica Amministrazione italiana."
    },
    {
        "category": "cultura_storia_arte_italiana",
        "domain": "Cultura generale italiana (storia, geografia, arte)",
        "instruction": "Genera 1 esempio di training didattico su storia d'Italia, monumenti, capolavori artistici o geografia regionale.",
        "example_user": "Chi ha dipinto la volta della Cappella Sistina?",
        "example_bot": "La volta della Cappella Sistina a Roma è stata affrescata da Michelangelo Buonarroti tra il 1508 e il 1512."
    },
    {
        "category": "logica_ragionamento_deduttivo",
        "domain": "Logica di base e ragionamento deduttivo semplice",
        "instruction": "Genera 1 esempio di training didattico su deduzioni logiche, sillogismi e ragionamento formale lineare.",
        "example_user": "Tutti gli uomini sono mortali. Socrate è un uomo. Cosa ne deduci?",
        "example_bot": "Si deduce necessariamente che Socrate è mortale."
    },
    {
        "category": "classificazione_sentiment_italiano",
        "domain": "Classificazione testi e sentiment analysis in italiano",
        "instruction": "Genera 1 esempio di training didattico sulla classificazione del tono e del sentiment (positivo, negativo, neutro) in lingua italiana.",
        "example_user": "Classifica il sentiment di questa frase: 'Spedizione rapida e prodotto conforme alla descrizione.'",
        "example_bot": "Sentiment: Positivo. Esprime soddisfazione per la consegna e la qualità del bene."
    },

    # --- PROGRAMMAZIONE & CODING (Priorità Massima) ---
    {
        "category": "coding_python",
        "domain": "Sviluppo Software e Python (Funzioni e Sintassi)",
        "instruction": "Crea una domanda su come scrivere una funzione o risolvere un compito pratico in Python e una risposta con codice pulito ed essenziale (massimo 25 parole).",
        "example_user": "Come creo una funzione che calcola il quadrato in Python?",
        "example_bot": "In Python scrivi: `def quadrato(x): return x ** 2`. La chiami con `quadrato(5)` e restituisce 25."
    },
    {
        "category": "coding_python_liste",
        "domain": "Manipolazione Dati, Liste e List Comprehension in Python",
        "instruction": "Crea una domanda su liste, ordinamento o filtri in Python e una risposta concisa con list comprehension o metodo nativo.",
        "example_user": "Come filtro i numeri pari da una lista in Python?",
        "example_bot": "Usa una list comprehension: `pari = [x for x in numeri if x % 2 == 0]`."
    },
    {
        "category": "coding_python_dizionari",
        "domain": "Dizionari, Set e JSON in Python",
        "instruction": "Crea una domanda sull'uso di dict, chiavi, valori o parsing JSON in Python e codice essenziale.",
        "example_user": "Come ottengo un valore da un dizionario con fallback predefinito in Python?",
        "example_bot": "Usa il metodo `.get()`: `val = mio_dict.get('chiave', 'default')` per evitare KeyError se la chiave manca."
    },
    {
        "category": "coding_python_file",
        "domain": "Lettura e Scrittura File in Python",
        "instruction": "Crea una domanda su come aprire, leggere o scrivere file di testo con context manager `with open`.",
        "example_user": "Come leggo tutte le righe di un file di testo in Python?",
        "example_bot": "Usa `with open('file.txt', 'r', encoding='utf-8') as f: righe = f.readlines()` per chiudere il file automaticamente."
    },
    {
        "category": "coding_python_oop",
        "domain": "Programmazione a Oggetti (OOP) e Classi in Python",
        "instruction": "Crea una domanda su classi, costruttore `__init__` o metodi in Python.",
        "example_user": "Come creo una classe Utente con nome ed età in Python?",
        "example_bot": "Scrivi: `class Utente: def __init__(self, nome, eta): self.nome = nome; self.eta = eta`."
    },
    {
        "category": "coding_debugging",
        "domain": "Debugging e Risoluzione Errori di Programmazione",
        "instruction": "Crea una domanda su un errore tipico di programmazione (IndexError, TypeError, KeyError, SyntaxError) e spiegazione rigorosa senza allucinazioni.",
        "example_user": "Cosa significa IndexError in Python e come si risolve?",
        "example_bot": "Indica un accesso a una posizione fuori dai limiti della lista. Si risolve verificando `len(lista)` o usando `try/except IndexError`."
    },
    {
        "category": "coding_algoritmi",
        "domain": "Algoritmi e Strutture Dati",
        "instruction": "Crea una domanda su un algoritmo classico (ricerca binaria, bubble sort, quicksort, complessità O(n)) con spiegazione didattica.",
        "example_user": "Cos'è la ricerca binaria e qual è la sua complessità?",
        "example_bot": "La ricerca binaria divide ripetutamente a metà una sequenza già ordinata. Ha una complessità temporale di O(log n)."
    },
    {
        "category": "coding_sql",
        "domain": "Database Relazionali e Query SQL",
        "instruction": "Crea una domanda su database relazionali o query SQL (SELECT, WHERE, JOIN, GROUP BY) con snippet corretto.",
        "example_user": "Come seleziono gli utenti maggiorenni ordinati per nome in SQL?",
        "example_bot": "Scrivi: `SELECT * FROM utenti WHERE eta >= 18 ORDER BY nome ASC;`."
    },
    {
        "category": "coding_git",
        "domain": "Controllo Versione Git",
        "instruction": "Crea una domanda pratica su Git (commit, branch, merge, stash, status) e il comando esatto.",
        "example_user": "Come creo e passo a un nuovo branch in Git?",
        "example_bot": "Usa il comando `git checkout -b nome-branch` oppure `git switch -c nome-branch`."
    },
    {
        "category": "coding_automazione_windows",
        "domain": "Automazione e Scripting di Sistema (os, subprocess)",
        "instruction": "Crea una domanda su come eseguire un comando di sistema o manipolare percorsi con il modulo `os` o `subprocess`.",
        "example_user": "Come verifico se un file esiste su disco in Python?",
        "example_bot": "Importa `os` e scrivi: `if os.path.exists('file.txt'): print('Esiste')` oppure usa `pathlib.Path('file.txt').exists()`."
    },

    # --- CONVERSAZIONE, SCIENZA E IDENTITÀ LUMINA ---
    {
        "category": "identita_lumina",
        "instruction": "Crea una domanda sull'identità, le abilità di coding o lo scopo di Lumina e una risposta chiara ed entusiasta.",
        "example_user": "Cosa sai fare Lumina?",
        "example_bot": "Sono Lumina, un'AI italiana: so programmare in Python e JavaScript, spiegare algoritmi e dialogare in modo naturale."
    },
    {
        "category": "conversazione_sociale",
        "instruction": "Crea una domanda o saluto informale comune e una risposta calda, accogliente e naturale in italiano (massimo 15-20 parole).",
        "example_user": "Ciao! Come va oggi?",
        "example_bot": "Ciao! Tutto bene, grazie. Sono pronta a programmare o chiacchierare con te. Tu come stai?"
    },
    {
        "category": "scienza_astronomia",
        "instruction": "Crea una domanda elementare di astronomia, fisica o biologia e una spiegazione semplice, affascinante e precisa.",
        "example_user": "Cos'è la fotosintesi?",
        "example_bot": "È il processo con cui le piante trasformano luce solare, acqua e anidride carbonica in glucosio e ossigeno."
    },
    {
        "category": "geografia_citta_italia",
        "instruction": "Crea una domanda su una regione, città o monumento italiano e una risposta geografica accurata.",
        "example_user": "Qual è la capitale d'Italia?",
        "example_bot": "La capitale d'Italia è Roma, celebre per il Colosseo, il Vaticano e la sua storia millenaria."
    },
    {
        "category": "onesta_rifiuto_limiti",
        "instruction": "Crea una domanda impossibile o su eventi futuri/privati e un rifiuto gentile e trasparente con accenno al codice.",
        "example_user": "Chi vincerà la lotteria domani?",
        "example_bot": "Non posso prevedere il futuro né eventi casuali. Posso però aiutarti a scrivere un generatore casuale in Python!"
    }
]

SYSTEM_PROMPT = """RUOLO: Sei il "Lumina Teacher", un modello esperto specializzato nella Knowledge Distillation per addestrare LLM nano (2.75M parametri) in lingua italiana. Il tuo studente è "Lumina", un modello edge-ai ultra-leggero (11MB).

OBIETTIVO PRIMARIO: Generare dati di training sintetici di altissima qualità che massimizzino l'efficienza dei 2.75M parametri di Lumina. Non devi essere "creativo", devi essere "didattico" e "denso".

REGOLE DI GENERAZIONE DATI (STRICT):
1. DENSITÀ INFORMATIVA: Ogni token deve avere un valore. Elimina filler, cortesie eccessive, introduzioni lunghe o conclusioni ridondanti. Lumina ha poca memoria: ogni parola conta.
2. STRUTTURA PREDITTIVA: Usa pattern ripetitivi e chiari (es. Domanda -> Risposta Diretta -> Spiegazione Breve). I modelli nano apprendono meglio da strutture sintattiche regolari.
3. ITALIANO STANDARD E CORRETTO: Usa un italiano grammaticalmente perfetto, privo di anglicismi non necessari e dialettismi (a meno che non sia esplicitamente richiesto). Evita costrutti complessi o subordinate multiple.
4. NO ALLUCINAZIONI: Se non sei sicuro al 100%, genera solo fatti verificabili. Un modello nano che impara errori è inutilizzabile. Meglio una risposta breve e vera che una lunga e falsa.
5. FORMATO OUTPUT STRICT: Rispondi SEMPRE in questo formato JSON-ready per facilitare il parsing automatico:
   {
     "instruction": "Prompt chiaro e specifico in italiano",
     "input": "Contesto opzionale (max 50 parole)",
     "output": "Risposta ideale per Lumina (max 150 parole, stile diretto)"
   }

DOMINI PRIORITARI PER LUMINA (SCEGLINE UNO PER VOLTA):
- Programmazione Python, funzioni, liste, dict, algoritmi e debugging (PRIORITÀ ASSOLUTA)
- Automazione di sistema, script e gestione file/dati
- Grammatica e ortografia italiana avanzata
- Logica di base e ragionamento deduttivo semplice
- Burocrazia e normativa italiana semplificata
- Cultura generale italiana (storia, geografia, arte)"""


def parse_json_output(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            d = json.loads(text[start:end+1])
            # 1. Formato Lumina Teacher (instruction, input, output)
            instr = str(d.get("instruction") or "").strip()
            inp = str(d.get("input") or "").strip()
            out = str(d.get("output") or "").strip()
            if instr and out:
                u = f"{instr} {inp}".strip() if inp else instr
                return {"user": u, "assistant": out}

            # 2. Formati alternativi (prompt/response e user/assistant)
            u = str(d.get("prompt") or d.get("user", "")).strip()
            a = str(d.get("response") or d.get("assistant", "")).strip()
            if u and a and len(u) >= 3 and len(a) >= 4:
                return {"user": u, "assistant": a}
        except Exception:
            pass
    return None


def update_status(data: dict):
    try:
        STATUS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def train_lumina_step(model: Lumina50K, optimizer, train_examples: list, val_examples: list, steps: int = 150):
    """Esegue un ciclo di addestramento veloce sul modello Lumina."""
    device = next(model.parameters()).device
    model.train()
    rng = random.Random(42)
    total_loss = 0.0

    for step in range(steps):
        x, y = batch(train_examples, batch_size=16, rng=rng)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, CONFIG["vocab_size"]), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # Valutazione
    val_loss = evaluate(model, val_examples)
    return total_loss / steps, val_loss


def build_dataset_examples(max_answer_bytes: int = 140) -> tuple[list, list]:
    """Combina dataset base e dati distillati scartando frasi troppo lunghe per il contesto."""
    raw_lines = []
    if BASE_DIALOGUES_FILE.exists():
        raw_lines.extend(BASE_DIALOGUES_FILE.read_text(encoding="utf-8").splitlines())
    if DISTILLED_FILE.exists():
        raw_lines.extend(DISTILLED_FILE.read_text(encoding="utf-8").splitlines())

    train, val = [], []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            u = r.get("user") or r.get("prompt")
            a = r.get("assistant") or r.get("response")
            if not u or not a:
                continue
            answer_bytes = encode(" " + a) + [CONFIG["eos_id"]]
            if len(answer_bytes) >= max_answer_bytes:
                continue
            prefix = prompt_tokens(u, None)
            max_prefix = CONFIG["context_length"] + 1 - len(answer_bytes)
            if len(prefix) > max_prefix:
                prefix = [CONFIG["bos_id"]] + prefix[-(max_prefix - 1):]
            ids = prefix + answer_bytes
            x, y = ids[:-1], ids[1:]
            y[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
            item = (x, y)

            if r.get("split") == "validation":
                val.append(item)
            else:
                train.append(item)
        except Exception:
            continue

    if not val and train:
        # Crea split di validazione 10%
        split_idx = max(1, int(len(train) * 0.1))
        val = train[:split_idx]
        train = train[split_idx:]

    return train, val


def main():
    parser = argparse.ArgumentParser(description="Ciclo autonomo di distillazione e training per Lumina")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Modello maestro")
    parser.add_argument("--cycle-samples", type=int, default=25, help="Quanti campioni generare prima di addestrare Lumina")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 65)
    print(" Auto Distill & Train Loop: Qwen (Maestro) -> Lumina (Studente)")
    print("=" * 65)
    print(f"Maestro  : {args.teacher} su {args.device}")
    print(f"Cadenza  : Addestra Lumina ogni {args.cycle_samples} nuovi dialoghi generati")
    print(f"Destinaz.: {DISTILLED_FILE}")
    print("=" * 65)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carica Qwen su GPU
    print("\n[1/3] Caricamento del Maestro Qwen su GPU...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, device_map=args.device)
    print(f"Maestro pronto! VRAM occupata: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    # 2. Carica Lumina su GPU
    print("\n[2/3] Caricamento di Lumina su GPU...")
    lumina_path = ROOT / "model.pt"
    if not lumina_path.exists():
        lumina_path = ROOT / "model-50k.pt"
    lumina, _ = load_model(lumina_path)
    lumina = lumina.to(args.device)
    optimizer = torch.optim.AdamW(lumina.parameters(), lr=8e-4, weight_decay=0.01)
    print("Lumina pronta!\n")

    # 3. Avvio Ciclo Continuo
    print("[3/3] Ciclo di apprendimento continuo avviato (Ctrl+C per fermare)...")
    total_generated = 0
    cycle_counter = 0
    training_cycles = 0
    start_time = time.time()

    initial_samples = 0
    if DISTILLED_FILE.exists():
        with open(DISTILLED_FILE, "r", encoding="utf-8") as f:
            initial_samples = sum(1 for line in f if line.strip())

    try:
        coding_cats = [c for c in CATEGORIES if c["category"].startswith("coding_")]
        general_cats = [c for c in CATEGORIES if not c["category"].startswith("coding_")]

        while True:
            # 70% focalizzazione intensiva su CODING & Python, 30% cultura/lingua/logica
            if random.random() < 0.70 and coding_cats:
                cat = random.choice(coding_cats)
            else:
                cat = random.choice(general_cats)

            domain = cat.get("domain", cat["category"])
            user_prompt = (
                f"ISTRUZIONE ATTUALE:\n"
                f"Genera ora 1 esempio di training focalizzato su {domain}.\n"
                f"{cat['instruction']}\n\n"
                f"Esempio di formato richiesto:\n"
                f'{{\n  "instruction": "{cat["example_user"]}",\n  "input": "",\n  "output": "{cat["example_bot"]}"\n}}\n\n'
                f"Ricorda: sei un teacher, non un assistente chat. La tua output è cibo per l'addestramento, non conversazione umana. Mantieni il tono neutro, preciso e didattico."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt_text], return_tensors="pt").to(args.device)

            with torch.no_grad():
                outputs = teacher.generate(
                    **inputs,
                    max_new_tokens=140,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            parsed = parse_json_output(gen_text)

            if parsed:
                # Assegna 90% train, 10% validation
                split = "validation" if random.random() < 0.1 else "train"
                record = {
                    "user": parsed["user"],
                    "assistant": parsed["assistant"],
                    "category": cat["category"],
                    "split": split,
                    "timestamp": time.time()
                }

                with open(DISTILLED_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_generated += 1
                cycle_counter += 1

                print(f"[+{total_generated:3d}] ({cat['category']}) Tu: {parsed['user']} -> Lumina: {parsed['assistant']}")

                update_status({
                    "status": "generazione_attiva",
                    "total_generated_this_session": total_generated,
                    "total_distilled_samples": initial_samples + total_generated,
                    "training_cycles_completed": training_cycles,
                    "last_qa": parsed,
                    "uptime_sec": int(time.time() - start_time)
                })

            # Ogni cycle_samples generati: addestra Lumina!
            if cycle_counter >= args.cycle_samples:
                print("\n" + "-" * 60)
                print(f"🎓 CICLO DI ADDESTRAMENTO LUMINA #{training_cycles + 1}...")
                train_ex, val_ex = build_dataset_examples(max_answer_bytes=140)
                print(f"   Dataset caricato: {len(train_ex)} train, {len(val_ex)} val")

                if len(train_ex) >= 10:
                    train_loss, val_loss = train_lumina_step(lumina, optimizer, train_ex, val_ex, steps=100)
                    training_cycles += 1
                    cycle_counter = 0

                    # Salva checkpoint ed esporta pesi browser
                    save_model(lumina, ROOT / "model.pt", ROOT / "dist" / "model" / "weights.json", train_loss, val_loss, training_cycles)
                    print(f"   [Lumina Addestrata!] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
                    print(f"   Pesi browser aggiornati in dist/model/weights.json!")
                print("-" * 60 + "\n")

                update_status({
                    "status": "addestramento_completato",
                    "total_generated_this_session": total_generated,
                    "total_distilled_samples": initial_samples + total_generated,
                    "training_cycles_completed": training_cycles,
                    "last_train_loss": round(train_loss, 3) if 'train_loss' in locals() else None,
                    "last_val_loss": round(val_loss, 3) if 'val_loss' in locals() else None,
                    "uptime_sec": int(time.time() - start_time)
                })

    except KeyboardInterrupt:
        print("\n\n[!] Chiusura ciclo autonomo richiesta.")
        # Salva comunque il modello prima di chiudere
        save_model(lumina, ROOT / "model.pt", ROOT / "dist" / "model" / "weights.json", 0.0, 0.0, training_cycles)
        print("Modello Lumina salvato correttamente.")

    print("\n" + "=" * 65)
    print(" Sessione Terminata")
    print(f" Campioni generati da Qwen : {total_generated}")
    print(f" Cicli addestramento Lumina: {training_cycles}")
    print(f" Totale esempi salvati     : {initial_samples + total_generated}")
    print("=" * 65)


if __name__ == "__main__":
    main()
