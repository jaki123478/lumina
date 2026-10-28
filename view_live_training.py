"""
view_live_training.py - Visualizzatore Live da Terminale con Codifica UTF-8 Perfetta
Legge live_training.log in tempo reale senza bug di caratteri né glitch grafici.
"""

import sys
import os
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32":
    os.system("chcp 65001 >nul")

log_path = Path("live_training.log")

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 75)
    print("   LUMINA AI: SESSIONE LIVE DI INSEGNAMENTO E TRAINING ATTIVO")
    print("   Maestro: Qwen2.5-0.5B-Instruct  |  Studente: Lumina-Next 135M")
    print("   Hardware: GPU NVIDIA GeForce RTX 5060 Laptop (CUDA BFloat16)")
    print("=" * 75)
    print("In attesa delle lezioni in tempo reale...\n")

    if not log_path.exists():
        log_path.touch()

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        # Mostra le ultime 30 righe esistenti
        lines = f.readlines()
        for l in lines[-35:]:
            print(l, end="")

        while True:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.4)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChiusura visualizzatore live.")
