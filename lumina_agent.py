"""
Lumina Action Agent - Assistente Operativo per Windows con Function Calling.
Interpreta comandi in linguaggio naturale ed esegue azioni concrete sul PC
sotto la protezione in tempo reale di Lumina Sentinel.
"""

import sys
import os
import glob
import time
import shutil
import psutil
import subprocess
import webbrowser
import requests

from lumina_guardrails import LuminaSentinel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class LuminaActionAgent:
    def __init__(self, api_url="http://localhost:8000/v1/chat/completions"):
        self.api_url = api_url
        self.sentinel = LuminaSentinel()
        self.desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")

    def execute_tool(self, action: str, params: dict) -> str:
        """Esecutore di azioni di sistema."""
        try:
            if action == "system_status":
                ram = psutil.virtual_memory()
                disk = psutil.disk_usage("C:")
                cpu = psutil.cpu_percent(interval=0.2)
                return (
                    f"📊 **Stato Risorse di Sistema**:\n"
                    f"• CPU in uso: {cpu}%\n"
                    f"• Memoria RAM: {ram.percent}% usata ({ram.used // (1024**3)} GB su {ram.total // (1024**3)} GB)\n"
                    f"• Disco C: {disk.percent}% occupato ({disk.free // (1024**3)} GB liberi su {disk.total // (1024**3)} GB)"
                )

            elif action == "create_note":
                title = params.get("title", f"Nota_Lumina_{int(time.time())}")
                content = params.get("content", "Nota creata da Lumina AI.")
                if not title.endswith(".txt"):
                    title += ".txt"
                file_path = os.path.join(self.desktop_dir, title)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"✅ Nota creata sul Desktop con successo:\n`{file_path}`"

            elif action == "search_files":
                pattern = params.get("query", "*.*")
                folder = params.get("folder", self.desktop_dir)
                matches = []
                for root, _, files in os.walk(folder):
                    for f in files:
                        if pattern.lower() in f.lower():
                            matches.append(os.path.join(root, f))
                            if len(matches) >= 10:
                                break
                    if len(matches) >= 10:
                        break

                if matches:
                    res = f"🔍 Trovati {len(matches)} file corrispondenti a '{pattern}':\n"
                    for m in matches:
                        res += f"• `{m}`\n"
                    return res
                return f"Nessun file trovato per '{pattern}' in `{folder}`."

            elif action == "open_url":
                url = params.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                webbrowser.open(url)
                return f"🌐 Browser aperto su: {url}"

            elif action == "open_app":
                app = params.get("app", "").lower()
                cmd_map = {
                    "notepad": "notepad.exe",
                    "calcolatrice": "calc.exe",
                    "calc": "calc.exe",
                    "terminale": "cmd.exe",
                    "esplora risorse": "explorer.exe",
                    "explorer": "explorer.exe",
                    "task manager": "taskmgr.exe"
                }
                exec_cmd = cmd_map.get(app, app)
                subprocess.Popen(exec_cmd, shell=True)
                return f"🚀 Applicazione avviata: `{exec_cmd}`"

            return f"Azione '{action}' non riconosciuta."
        except Exception as e:
            return f"❌ Errore durante l'esecuzione dell'azione {action}: {e}"

    def parse_intent(self, text: str):
        """Analizza il testo dell'utente per mappare intenti di azione o domanda."""
        t = text.lower()

        # 1. Risorse di sistema
        if any(k in t for k in ["risorse", "ram", "cpu", "memoria", "disco", "prestazioni", "stato pc"]):
            return "system_status", {}

        # 2. Crea nota
        if any(k in t for k in ["crea nota", "scrivi nota", "promemoria", "salva nota"]):
            # estrai titolo o contenuto se specificato
            parts = text.split("intitolata", 1)
            if len(parts) > 1:
                title = parts[1].strip()
            else:
                title = "Promemoria_Lumina"
            return "create_note", {"title": title, "content": f"Nota generata da Lumina AI:\n{text}"}

        # 3. Cerca file
        if any(k in t for k in ["cerca file", "trova file", "dove si trova", "cerca il file"]):
            words = text.split()
            query = words[-1] if words else "*.txt"
            return "search_files", {"query": query, "folder": "."}

        # 4. Apri URL / Web
        if "apri youtube" in t:
            return "open_url", {"url": "https://www.youtube.com"}
        if "apri google" in t:
            return "open_url", {"url": "https://www.google.it"}
        if "apri wikipedia" in t:
            return "open_url", {"url": "https://it.wikipedia.org"}
        if "apri https://" in t or "apri www." in t:
            words = text.split()
            target_url = [w for w in words if "http" in w or "www." in w][0]
            return "open_url", {"url": target_url}

        # 5. Apri App Windows
        for app_name in ["notepad", "calcolatrice", "calc", "terminale", "esplora risorse", "task manager"]:
            if f"apri {app_name}" in t or f"avvia {app_name}" in t:
                return "open_app", {"app": app_name}

        return None, None

    def ask_lumina(self, prompt: str) -> str:
        is_coding = any(k in prompt.lower() for k in ["python", "codice", "funzione", "def ", "classe", "errore", "script", "programma", "lista", "dict"])
        target_model = "lumina-coder" if is_coding else "lumina-2.75m"
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 180,
            "temperature": 0.3,
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=8)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            return f"[Errore HTTP {res.status_code}]"
        except Exception:
            return "Server Lumina non raggiungibile."

    def process(self, user_input: str) -> str:
        # Controllo Lumina Sentinel
        check = self.sentinel.evaluate(user_input)
        if not check.is_safe:
            return f"🛡️ [Lumina Sentinel]: Bloccato tentativo non autorizzato ({check.category})."

        action, params = self.parse_intent(check.sanitized_prompt)
        if action:
            print(f"⚙️  [Action Engine]: Esecuzione comando '{action}' con parametri {params}...")
            return self.execute_tool(action, params)

        # Altrimenti genera risposta testuale/conversazionale con Lumina
        return self.ask_lumina(check.sanitized_prompt)

    def run_interactive(self):
        print("=" * 65)
        print("       🛠️  LUMINA ACTION AGENT (WINDOWS AUTOMATION)")
        print("=" * 65)
        print("Assistente operativo locale con Function Calling protetto da Sentinel.")
        print("Esempi di comandi operativi:")
        print("  • 'Controlla lo stato delle risorse di sistema (RAM/CPU/Disco)'")
        print("  • 'Crea una nota sul desktop intitolata Promemoria'")
        print("  • 'Apri YouTube' oppure 'Apri Calcolatrice'")
        print("  • 'Cerca i file .py nel progetto'")
        print("  • O fai qualsiasi domanda di programmazione o cultura generale!")
        print("Scrivi 'esci' per terminare.\n")

        while True:
            try:
                cmd = input("Tu > ").strip()
                if not cmd:
                    continue
                if cmd.lower() in ["esci", "exit", "quit", "q"]:
                    print("Arrivederci!")
                    break

                output = self.process(cmd)
                print(f"\n🤖 {output}\n")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    agent = LuminaActionAgent()
    agent.run_interactive()
