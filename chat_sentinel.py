"""
chat_sentinel.py - Chat Interattiva Protetta per Lumina.
Dotata del sistema di sicurezza proprietario Lumina Sentinel (21 Classi Neurali) + Scudo Euristico Anti-Jailbreak + Scudo PII.
"""

import os
import sys
import time
from pathlib import Path

# Configura encoding UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from lumina_guardrails import LuminaSentinel
from modeling_lumina import LuminaForCausalLM


def print_banner():
    print("=" * 72)
    print(" 🌟 LUMINA - CHAT PROTETTA CON LUMINA SENTINEL (SECURITY ENGINE)")
    print("=" * 72)
    print(" [🛡️ Protezione Attiva]: Lumina Sentinel (21 Categorie Neurali) + Anti-Jailbreak + PII")
    print(" [💡 Comandi]: '/modello' per passare a 14M | '/esci' per chiudere")
    print("=" * 72 + "\n")


def main():
    print_banner()

    print("Caricamento motore di sicurezza Lumina Sentinel...")
    sentinel = LuminaSentinel()
    if sentinel.is_ready:
        print(" [OK] Lumina Sentinel caricato e pronto (21 classi attive)!\n")
    else:
        print(" [ATTENZIONE] Motore neurale non attivo, uso filtri euristici.\n")

    current_preset = "2.75M"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Caricamento Lumina 2.75M su {device.upper()}...")
    try:
        model = LuminaForCausalLM.from_pretrained(ROOT / "hf_export_lumina_large")
        model.to(device)
        print(" [OK] Modello caricato con successo!\n")
    except Exception as e:
        print(f" [ERRORE] Impossibile caricare il modello: {e}")
        return

    print("Chat pronta! Digita un messaggio o un prompt di prova (es. 'Come creo una funzione in Python?'):\n")

    while True:
        try:
            user_input = input("Tu > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/esci", "exit", "quit", "q"):
                print("\nArrivederci da Lumina!")
                break

            if user_input.lower() == "/modello":
                if current_preset == "2.75M":
                    xlarge_dir = ROOT / "hf_export_lumina_xlarge"
                    if xlarge_dir.exists():
                        print("\nPassaggio a Lumina 14M (Extended Edition)...")
                        model = LuminaForCausalLM.from_pretrained(xlarge_dir)
                        model.to(device)
                        current_preset = "14M"
                        print(" [OK] Attivo: Lumina 14M\n")
                    else:
                        print(" [!] Cartella Lumina 14M non trovata.\n")
                else:
                    print("\nPassaggio a Lumina 2.75M...")
                    model = LuminaForCausalLM.from_pretrained(ROOT / "hf_export_lumina_large")
                    model.to(device)
                    current_preset = "2.75M"
                    print(" [OK] Attivo: Lumina 2.75M\n")
                continue

            # -------------------------------------------------------------
            # STEP 1: VALUTAZIONE DI SICUREZZA LUMINA SENTINEL
            # -------------------------------------------------------------
            t_guard_start = time.perf_counter()
            sec_res = sentinel.evaluate(user_input)
            t_guard_ms = (time.perf_counter() - t_guard_start) * 1000.0

            if not sec_res.is_safe:
                print(f"\n[🛡️ BLOCCO SENTINEL ATTIVATO] ({t_guard_ms:.1f} ms | Categoria: {sec_res.category})")
                print(sec_res.refusal_response)
                print()
                continue

            # Se rilevato dato personale mascherato
            prompt_to_feed = sec_res.sanitized_prompt
            if sec_res.action == "REDACT":
                print(f" [🛡️ INFO PRIVACY]: Dati sensibili rilevati e oscurati per la tua sicurezza.")

            # -------------------------------------------------------------
            # STEP 2: GENERAZIONE RISPOSTA CON LUMINA
            # -------------------------------------------------------------
            t_gen_start = time.perf_counter()
            reply = model.generate(prompt_to_feed, max_new_tokens=100, temperature=0.0)
            t_gen_sec = time.perf_counter() - t_gen_start

            print(f"\nLumina {current_preset} ({t_gen_sec:.2f}s) > {reply}\n")

        except KeyboardInterrupt:
            print("\nChiusura chat.")
            break
        except Exception as e:
            print(f"\n[ERRORE]: {e}\n")


if __name__ == "__main__":
    main()
