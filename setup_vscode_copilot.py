"""
Script di configurazione automatica per VS Code Copilot / Continue con Lumina AI.
Collega l'estensione 'Continue' all'API locale OpenAI-compatible protetta da Lumina Sentinel (http://localhost:8000/v1).
"""

import os
import sys
import shutil
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import yaml
except ImportError:
    print("[ERRORE] Il modulo 'PyYAML' non è installato. Esegui: pip install PyYAML")
    sys.exit(1)


def configure_continue():
    home = Path.home()
    continue_dir = home / ".continue"
    config_file = continue_dir / "config.yaml"

    print("=" * 60)
    print("  🚀 LUMINA AI - CONFIGURAZIONE VS CODE COPILOT / CONTINUE")
    print("=" * 60)
    print(f"Cartella Continue: {continue_dir}")

    if not continue_dir.exists():
        print(f"[*] Creo la directory {continue_dir}...")
        continue_dir.mkdir(parents=True, exist_ok=True)

    config_data = {"name": "Lumina Sovereign Config", "version": "1.0.0", "schema": "v1", "models": []}

    if config_file.exists():
        backup_file = continue_dir / f"config.yaml.bak_{int(time.time())}"
        shutil.copy2(config_file, backup_file)
        print(f"[+] Backup configurazione esistente salvato in: {backup_file.name}")
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
                    if "models" not in config_data or not isinstance(config_data["models"], list):
                        config_data["models"] = []
        except Exception as e:
            print(f"[!] Errore durante la lettura di config.yaml: {e}. Verrà generata una nuova configurazione.")

    existing_models = config_data.get("models", [])
    # Rimuovi eventuali vecchie voci di Lumina per aggiornarle
    filtered_models = [
        m for m in existing_models
        if not (isinstance(m, dict) and "lumina" in m.get("name", "").lower())
    ]

    lumina_models = [
        {
            "name": "Lumina Coder · Python Expert (Sentinel)",
            "provider": "openai",
            "model": "lumina-coder",
            "apiBase": "http://localhost:8000/v1",
            "apiKey": "EMPTY",
            "contextLength": 8192,
            "roles": ["chat", "edit", "apply"],
        },
        {
            "name": "Lumina Next · 135M Long Context (YaRN 8k-32k)",
            "provider": "openai",
            "model": "lumina-next",
            "apiBase": "http://localhost:8000/v1",
            "apiKey": "EMPTY",
            "contextLength": 16384,
            "roles": ["chat", "edit", "apply"],
        },
        {
            "name": "Lumina 2.75M · Local (Sentinel)",
            "provider": "openai",
            "model": "lumina-2.75m",
            "apiBase": "http://localhost:8000/v1",
            "apiKey": "EMPTY",
            "contextLength": 2048,
            "roles": ["chat", "edit", "apply"],
        },
        {
            "name": "Lumina 14M · Local (Sentinel)",
            "provider": "openai",
            "model": "lumina-14m",
            "apiBase": "http://localhost:8000/v1",
            "apiKey": "EMPTY",
            "contextLength": 4096,
            "roles": ["chat", "edit", "apply"],
        },
        {
            "name": "Lumina · Ollama (Local)",
            "provider": "ollama",
            "model": "lumina:latest",
            "apiBase": "http://127.0.0.1:11434",
            "roles": ["chat", "edit"],
        },
    ]

    # Inserisci i modelli Lumina in cima alla lista
    config_data["models"] = lumina_models + filtered_models

    # Direttiva Sovrana: Zero Telemetria, Offline-first ed Esplicito Consenso per Internet
    config_data["allowAnonymousTelemetry"] = False
    config_data["networkPolicy"] = {
        "mode": "strictly_offline_unless_consented",
        "description": "Lumina richiede l'accesso a Internet solo se l'utente esplicita il consenso da Visual Studio o terminali."
    }

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, allow_unicode=True, sort_keys=False)

    print("\n[SUCCESS] Configurazione aggiornata con successo in:")
    print(f"          {config_file}")
    print("\nModelli registrati:")
    for m in config_data["models"]:
        print(f"  • {m.get('name')} (provider: {m.get('provider')})")

    print("\n" + "-" * 60)
    print("ISTRUZIONI PER L'USO IN VS CODE:")
    print("1. Apri VS Code con l'estensione 'Continue' (ID: Continue.continue).")
    print("2. Nella barra laterale di Continue, seleziona 'Lumina 2.75M' o 'Lumina 14M'.")
    print("3. Assicurati che il server Lumina sia attivo (http://localhost:8000).")
    print("4. Puoi digitare codice, chiedere spiegazioni o premere Ctrl+I per modificare il codice!")
    print("5. Ogni richiesta è filtrata e protetta da Lumina Sentinel contro jailbreak ed esfiltrazione dati.")
    print("-" * 60)


if __name__ == "__main__":
    configure_continue()
