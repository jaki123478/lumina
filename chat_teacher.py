"""
chat_teacher.py - Chat interattiva con l'LLM Maestro locale (Qwen).

Permette di dialogare in italiano con Qwen direttamente da terminale o script,
e salvare le migliori risposte nel dataset di Lumina per addestrarlo.

Uso:
    python chat_teacher.py
    python chat_teacher.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import os
import sys
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

DEFAULT_SYSTEM = (
    "Sei un assistente virtuale intelligente, chiaro, educato e cordiale. "
    "Rispondi in lingua italiana naturale, corretta ed esaustiva. "
    "Se ti viene chiesta una cosa impossibile o che viola la privacy o su eventi futuri, "
    "spiega gentilmente i tuoi limiti in modo onesto."
)

def main():
    parser = argparse.ArgumentParser(description="Chatta con Qwen locale e allena Lumina")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Modello da usare")
    parser.add_argument("--dataset", type=str, default="data/distilled_dialogues.jsonl", help="File di salvataggio")
    args = parser.parse_args()

    print("=" * 60)
    print(" Qwen Maestro Locale - Chat Interattiva & Distillazione Lumina")
    print("=" * 60)
    print(f"Modello   : {args.model}")
    print(f"Hardware  : {'CUDA (NVIDIA GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset   : {args.dataset}")
    print("=" * 60)

    print("Caricamento del modello...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)
    print("Modello pronto per chattare!\n")
    print("Comandi speciali:")
    print("  'esci' o 'exit' -> Chiude la chat")
    print("  'salva'         -> Salva l'ultimo scambio domanda-risposta nel dataset di Lumina")
    print("  'stats'         -> Mostra quanti esempi sono salvati nel dataset")
    print("-" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(args.dataset)), exist_ok=True)
    history = []
    last_qa = None

    while True:
        try:
            user_input = input("\nTu > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["esci", "exit", "quit"]:
                print("Chiusura sessione chat.")
                break

            if user_input.lower() == "stats":
                count = 0
                if os.path.exists(args.dataset):
                    with open(args.dataset, "r", encoding="utf-8") as f:
                        count = sum(1 for line in f if line.strip())
                print(f"Campioni salvati nel dataset per Lumina: {count}")
                continue

            if user_input.lower() == "salva":
                if not last_qa:
                    print("Nessuna conversazione recente da salvare.")
                else:
                    record = {
                        "category": "chat_interattiva",
                        "prompt": last_qa["prompt"],
                        "response": last_qa["response"],
                        "teacher": args.model,
                        "timestamp": time.time()
                    }
                    with open(args.dataset, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"Salvato con successo nel dataset! Lumina potra imparare da questo scambio.")
                continue

            messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for h in history[-4:]:
                messages.append(h)
            messages.append({"role": "user", "content": user_input})

            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            dur = time.time() - start
            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            tok_sec = len(new_tokens) / max(0.01, dur)

            print(f"\nQwen > {reply}")
            print(f"       ({len(new_tokens)} token in {dur:.2f}s · {tok_sec:.1f} tok/s)")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            last_qa = {"prompt": user_input, "response": reply}

            # Chiedi comodamente se salvare
            salva_quick = input("       Vuoi salvare questo scambio per Lumina? (s/N): ").strip().lower()
            if salva_quick == "s":
                record = {
                    "category": "chat_interattiva",
                    "prompt": user_input,
                    "response": reply,
                    "teacher": args.model,
                    "timestamp": time.time()
                }
                with open(args.dataset, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print("       [OK] Salvato nel dataset!")

        except KeyboardInterrupt:
            print("\nUscita dalla chat.")
            break

if __name__ == "__main__":
    main()
