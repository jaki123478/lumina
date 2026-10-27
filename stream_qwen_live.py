"""
stream_qwen_live.py - Visualizzatore Desktop Terminale in Tempo Reale
Mostra a schermo ogni domanda, risposta e codice che Qwen sta insegnando a Lumina.
"""

import os
import sys
import time
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        os.system("chcp 65001 >nul")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "distilled_dialogues.jsonl"
STATUS_FILE = ROOT / "data" / "auto_train_status.json"

# Colori ANSI
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}   🧠 LUMINA TEACHER LIVE: COSA QWEN STA INSEGNANDO A LUMINA        {RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{GRAY}Monitor in ascolto su GPU NVIDIA RTX 5060... Premi Ctrl+C per uscire.{RESET}\n")

    if not DATA_FILE.exists():
        print(f"{YELLOW}[!] In attesa che il file dati venga generato...{RESET}")

    # Leggi gli ultimi 5 campioni già presenti
    last_pos = 0
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
            for line in lines[-5:]:
                try:
                    d = json.loads(line)
                    _print_dialogue(d)
                except Exception:
                    pass
            last_pos = f.tell()

    # Loop di streaming in tempo reale
    while True:
        try:
            if DATA_FILE.exists():
                with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()

                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            _print_dialogue(d)
                        except Exception:
                            continue

            time.sleep(1.0)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Chiusura monitor live.{RESET}")
            break
        except Exception as e:
            time.sleep(2.0)


def _print_dialogue(d: dict):
    cat = d.get("category", "generale")
    user = d.get("user", "")
    assistant = d.get("assistant", "")
    timestamp = d.get("timestamp", time.time())
    timestr = time.strftime("%H:%M:%S", time.localtime(timestamp))

    cat_color = MAGENTA if "coding" in cat else CYAN
    print(f"{GRAY}[{timestr}]{RESET} {cat_color}{BOLD}[{cat.upper()}]{RESET}")
    print(f"  {YELLOW}{BOLD}Qwen (Domanda/Task):{RESET} {WHITE}{user}{RESET}")
    print(f"  {GREEN}{BOLD}Lumina (Codice/Risposta):{RESET} {GREEN}{assistant}{RESET}")
    print(f"{GRAY}{'-' * 70}{RESET}\n")


if __name__ == "__main__":
    main()
