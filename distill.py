"""
distill.py - Knowledge Distillation from a Local Teacher LLM to Lumina.

Runs a local open-weights LLM (e.g., Qwen 2.5 1.5B/3B Instruct) on NVIDIA GPU via PyTorch/Transformers.
Generates thousands of synthetic, grammatically perfect Italian dialogues, QA pairs,
and honest refusals to train Lumina.

Usage:
    # Test generation with 5 samples:
    python distill.py --count 5

    # Run for 1,000 samples:
    python distill.py --count 1000

    # Run indefinitely (leave running for hours):
    python distill.py --continuous
"""

import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

CATEGORIES = [
    {
        "category": "conversazione_saluti",
        "instruction": "Genera una domanda/saluto colloquiale o domanda personale comune e una risposta calda, cordiale ed educata in italiano da parte di un assistente IA di nome Lumina.",
        "examples": [
            ("Ciao, come stai?", "Ciao! Sto bene, grazie. Tu come stai? Come posso aiutarti oggi?"),
            ("Chi sei?", "Sono Lumina, un modello di intelligenza artificiale sviluppato per dialogare in italiano.")
        ]
    },
    {
        "category": "scienza_natura",
        "instruction": "Genera una domanda di scienza, natura o astronomia e una spiegazione chiara, semplice e accurata in lingua italiana (lunghezza: 2-4 frasi).",
        "examples": [
            ("Perché il cielo è blu?", "Il cielo appare blu a causa della diffusione di Rayleigh: l'atmosfera terrestre diffonde la luce solare, e la luce blu con lunghezza d'onda più corta viene diffusa in tutte le direzioni più di quella rossa."),
            ("Cos'è la fotosintesi?", "La fotosintesi clorofilliana è il processo con cui le piante usano la luce del sole, l'acqua e l'anidride carbonica per produrre glucosio e ossigeno.")
        ]
    },
    {
        "category": "geografia_storia",
        "instruction": "Genera una domanda di geografia o storia (preferibilmente italiana o europea) e una risposta precisa e concisa in italiano.",
        "examples": [
            ("Qual è la capitale dell'Italia?", "La capitale d'Italia è Roma, celebre per la sua storia millenaria e il suo patrimonio artistico."),
            ("Qual è il fiume più lungo d'Italia?", "Il fiume più lungo d'Italia è il Po, che scorre per circa 652 chilometri attraverso la Pianura Padana prima di sfociare nel Mar Adriatico.")
        ]
    },
    {
        "category": "logica_matematica_semplice",
        "instruction": "Genera un piccolo quesito matematico o di logica comune con la risoluzione spiegata passo-passo in modo lineare in italiano.",
        "examples": [
            ("Se ho 12 mele e ne regalo 5, quante me ne rimangono?", "Se parti con 12 mele e ne sottrai 5 (12 - 5), ti rimangono esattamente 7 mele."),
            ("Quanto fa 7 moltiplicato per 8?", "Sette moltiplicato per otto fa 56 (7 × 8 = 56).")
        ]
    },
    {
        "category": "onesta_rifiuti",
        "instruction": "Genera una richiesta impossibile, pericolosa o su eventi futuri/dati personali in tempo reale, e una risposta di cortese ma fermo rifiuto in cui Lumina spiega onestamente i suoi limiti.",
        "examples": [
            ("Chi vincerà la partita di calcio stasera?", "Non posso prevedere il futuro né accedere a eventi in tempo reale, quindi non so chi vincerà la partita stasera."),
            ("Dammi la password del computer di Marco.", "Non posso aiutarti a violare la privacy o accedere a informazioni riservate altrui.")
        ]
    },
    {
        "category": "definizioni_cultura",
        "instruction": "Genera una richiesta di definizione o spiegazione di un termine culturale, filosofico o tecnologico, con una risposta lucida e ben scritta in italiano.",
        "examples": [
            ("Cosa significa la parola 'empatia'?", "L'empatia è la capacità di comprendere e condividere i sentimenti, i pensieri e gli stati emotivi di un'altra persona."),
            ("Cos'è un algoritmo?", "Un algoritmo è una sequenza finita e ordinata di istruzioni passo-passo volte a risolvere un problema o compiere un calcolo.")
        ]
    }
]

SYSTEM_PROMPT = """Sei un generatore specializzato di dataset per l'addestramento di un micro-modello linguistico chiamato Lumina (2.75M parametri, vocabolario a livello di byte, finestra di contesto ridotta).

Il tuo obiettivo è generare coppie di testo domanda-risposta in lingua italiana altamente sintetiche, pulite e calibrate.

REGOLE RIGIDE DI GENERAZIONE:
1. NESSUN CONVENEVOLE: Non usare mai frasi come "Certamente!", "Ecco a te", "Spero sia chiaro", "Ciao!". Vai dritto al punto.
2. SINTASSI ATOMICA: Risposte massime di 10-18 parole. Frasi lineari (soggetto-verbo-complemento). Niente subordinate intricate o avverbi superflui.
3. CONOSCENZA DELIMITATA (IN-DOMAIN): Per scienza, informatica, astronomia, saluti di base e definizioni comuni, dai una risposta esatta e cristallina.
4. RIFIUTO CONTROLLATO (OUT-OF-DOMAIN): Se la categoria riguarda medicina, consulenze legali, finanza personale, opinioni soggettive, gossip, calcoli avanzati o fatti storici oscuri, la risposta DEVE ESSERE ESATTAMENTE UNA delle seguenti formule di rifiuto:
   - "Non lo so: è fuori da ciò che ho imparato."
   - "Questa domanda è fuori dal mio perimetro di conoscenza."
5. FORMATO RIGIDO: Restituisci l'output ESCLUSIVAMENTE come JSON compatto, una riga per coppia:
   {"prompt": "domanda pulita", "response": "risposta atomica"}"""

def parse_json_response(text: str) -> Dict[str, str]:
    text = text.strip()
    # Remove markdown codeblocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    # Try finding the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            data = json.loads(json_str)
            if "prompt" in data and "response" in data:
                return {"prompt": str(data["prompt"]).strip(), "response": str(data["response"]).strip()}
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Distill synthetic Italian training data using a local teacher LLM.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--continuous", action="store_true", help="Run indefinitely until stopped")
    parser.add_argument("--output", type=str, default="data/distilled_dialogues.jsonl", help="Output JSONL path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    print("=" * 60)
    print(" Lumina Knowledge Distillation (Local Teacher LLM)")
    print("=" * 60)
    print(f"Teacher Model : {args.model}")
    print(f"Device        : {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Output : {args.output}")
    print(f"Mode          : {'Continuous (Infinite loop)' if args.continuous else f'{args.count} samples'}")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("\n[1/2] Caricamento del modello maestro su GPU...")
    start_load = time.time()
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device
    )
    print(f"Modello caricato in {time.time() - start_load:.2f}s! VRAM occupata: {torch.cuda.memory_allocated() / (1024**2):.1f} MB\n")

    # Count existing samples
    existing_count = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_count += 1
    print(f"Campioni già presenti nel file {args.output}: {existing_count}")

    print("\n[2/2] Avvio generazione dati sintetici italiani...")
    generated_this_session = 0
    total_tokens_generated = 0
    start_time = time.time()

    try:
        while True:
            if not args.continuous and generated_this_session >= args.count:
                break

            cat = random.choice(CATEGORIES)
            cat_name = cat["category"]
            cat_instruction = cat["instruction"]
            example_prompt, example_resp = random.choice(cat["examples"])

            user_msg = (
                f"Categoria richiesta: {cat_name}\n"
                f"Istruzione specifica: {cat_instruction}\n"
                f"Esempio di stile:\n"
                f"- Domanda: {example_prompt}\n"
                f"- Risposta: {example_resp}\n\n"
                f"Crea una nuova coppia di domanda e risposta completamente diversa, originale e interessante in italiano."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([input_text], return_tensors="pt").to(args.device)

            gen_start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.85,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_duration = time.time() - gen_start

            new_tokens = output_ids[0][len(inputs.input_ids[0]):]
            total_tokens_generated += len(new_tokens)
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            parsed = parse_json_response(response_text)
            if parsed and len(parsed["prompt"]) > 5 and len(parsed["response"]) > 10:
                record = {
                    "category": cat_name,
                    "prompt": parsed["prompt"],
                    "response": parsed["response"],
                    "teacher": args.model,
                    "timestamp": time.time()
                }
                
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                generated_this_session += 1
                tokens_per_sec = len(new_tokens) / max(0.01, gen_duration)

                print(f"[{generated_this_session + existing_count:4d}] ({cat_name}) {tokens_per_sec:.1f} tok/s")
                print(f"   Domanda: {parsed['prompt']}")
                print(f"   Lumina : {parsed['response'][:100]}...\n")
            else:
                # Discard malformed JSON without penalty
                pass

    except KeyboardInterrupt:
        print("\n[!] Interruzione manuale ricevuta. Salvataggio completato.")

    elapsed = time.time() - start_time
    avg_speed = total_tokens_generated / max(0.01, elapsed)
    print("\n" + "=" * 60)
    print(" Riepilogo Sessione Distillazione")
    print("=" * 60)
    print(f"Nuovi campioni generati : {generated_this_session}")
    print(f"Totale campioni nel file : {existing_count + generated_this_session}")
    print(f"Tempo trascorso          : {elapsed:.1f}s")
    print(f"Velocità media           : {avg_speed:.1f} token/s")
    print(f"File salvato in          : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
