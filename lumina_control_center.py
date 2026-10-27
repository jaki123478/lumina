"""
Lumina AI - Sovereign Control Center (Desktop GUI)
Dashboard di controllo completa e nativa in dark mode per l'ecosistema Lumina.
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
import requests
import tkinter as tk
from tkinter import ttk, messagebox

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Palette colori Dark Cyber Modern
BG_COLOR = "#0b0f17"
CARD_BG = "#151c28"
CARD_BORDER = "#222f44"
TEXT_MAIN = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_CYAN = "#06b6d4"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_AMBER = "#f59e0b"
BUTTON_BG = "#1e293b"


class LuminaControlCenter:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("✨ Lumina AI - Sovereign Control Center")
        self.root.geometry("880x840")
        self.root.minsize(820, 770)
        self.root.configure(bg=BG_COLOR)

        self.api_base = "http://localhost:8000"
        self.workspace_dir = os.path.dirname(os.path.abspath(__file__))

        self._build_ui()
        self._start_metrics_updater()

    def _build_ui(self):
        # 1. Header
        header_frame = tk.Frame(self.root, bg=BG_COLOR, padx=20, pady=15)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="✨ LUMINA SOVEREIGN CONTROL CENTER",
            font=("Segoe UI", 18, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_COLOR,
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Piattaforma di controllo locale: Edge-AI 2.75M/14M, Sentinel Security Shield & Desktop Tools",
            font=("Segoe UI", 10),
            fg=TEXT_MUTED,
            bg=BG_COLOR,
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 2. Status Cards Panel (Grid 2x2)
        cards_frame = tk.Frame(self.root, bg=BG_COLOR, padx=20, pady=5)
        cards_frame.pack(fill=tk.X)

        for col in range(2):
            cards_frame.columnconfigure(col, weight=1, uniform="card")

        # Card 1: Server Status
        self.card_server = self._create_card(
            cards_frame, 0, 0, "🌐 SERVER REST / OPENAI API", "Verifica in corso...", ACCENT_BLUE
        )
        # Card 2: Lumina Sentinel
        self.card_sentinel = self._create_card(
            cards_frame, 0, 1, "🛡️ LUMINA SENTINEL (SICUREZZA)", "Attivo • ML + Euristiche + PII", ACCENT_GREEN
        )
        # Card 3: Dataset & Training
        self.card_train = self._create_card(
            cards_frame, 1, 0, "🔥 DISTILLATION & TRAINING", "Lettura dataset in corso...", ACCENT_AMBER
        )
        # Card 4: Local Models
        self.card_models = self._create_card(
            cards_frame, 1, 1, "🧠 ARCHITETTURA MODELLI", "Lumina 2.75M (11MB) • 14M (56MB)", ACCENT_PURPLE
        )

        # 3. Quick Action Launchers
        actions_header = tk.Frame(self.root, bg=BG_COLOR, padx=20, pady=10)
        actions_header.pack(fill=tk.X)
        tk.Label(
            actions_header,
            text="⚡ STRUMENTI & APPLICAZIONI DESKTOP (1-CLICK)",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_MAIN,
            bg=BG_COLOR,
        ).pack(anchor=tk.W)

        btns_frame = tk.Frame(self.root, bg=BG_COLOR, padx=20, pady=5)
        btns_frame.pack(fill=tk.X)
        for c in range(3):
            btns_frame.columnconfigure(c, weight=1)

        self._create_btn(btns_frame, 0, 0, "💬 Chat Protetta (Terminal)", ACCENT_BLUE, self._launch_chat_sentinel)
        self._create_btn(btns_frame, 0, 1, "📄 Parla con Documenti (RAG)", ACCENT_AMBER, self._launch_rag)
        self._create_btn(btns_frame, 0, 2, "🎙️ Assistente Vocale Elsa", ACCENT_GREEN, self._launch_voice)

        self._create_btn(btns_frame, 1, 0, "🛠️ Lumina Action Agent", ACCENT_CYAN, self._launch_agent)
        self._create_btn(btns_frame, 1, 1, "📱 Bot Telegram Sovrano", ACCENT_PURPLE, self._launch_telegram)
        self._create_btn(btns_frame, 1, 2, "⚡ Browser Offline (WASM)", "#0284c7", self._launch_wasm)

        self._create_btn(btns_frame, 2, 0, "💻 Attiva VS Code Copilot", "#334155", self._setup_vscode)
        self._create_btn(btns_frame, 2, 1, "🦙 Registra Modello Ollama", "#334155", self._create_ollama)
        self._create_btn(btns_frame, 2, 2, "🏆 Esegui Benchmark Suite", "#b45309", self._launch_benchmark)

        self._create_btn(btns_frame, 3, 0, "⚡ Lumina Spotlight Popup", "#0e7490", self._launch_spotlight)
        self._create_btn(btns_frame, 3, 1, "🧪 Code Sandbox & Fix", "#047857", self._launch_sandbox)
        self._create_btn(btns_frame, 3, 2, "🔄 Sync Hugging Face", "#6d28d9", self._launch_sync_hf)

        # 4. Integrated Quick Chat / Sentinel Tester
        chat_frame = tk.Frame(self.root, bg=CARD_BG, bd=1, relief=tk.SOLID, padx=15, pady=12)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(15, 15))

        chat_title_bar = tk.Frame(chat_frame, bg=CARD_BG)
        chat_title_bar.pack(fill=tk.X, pady=(0, 5))

        tk.Label(
            chat_title_bar,
            text="💬 TEST RAPIDO LUMINA & SENTINEL (LIVE CONSOLE)",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_CYAN,
            bg=CARD_BG,
        ).pack(side=tk.LEFT)

        self.lbl_chat_badge = tk.Label(
            chat_title_bar,
            text="🛡️ Sentinel Shield: VIGILE",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT_GREEN,
            bg=CARD_BG,
        )
        self.lbl_chat_badge.pack(side=tk.RIGHT)

        # Chat history text area
        self.txt_history = tk.Text(
            chat_frame,
            bg="#0d1117",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            font=("Consolas", 10),
            wrap=tk.WORD,
            bd=0,
            padx=10,
            pady=10,
            height=8,
        )
        self.txt_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.txt_history.insert(
            tk.END,
            "Lumina Sovereign Hub pronto.\nDigita una domanda o metti alla prova la sicurezza di Lumina Sentinel!\n\n",
        )
        self.txt_history.configure(state=tk.DISABLED)

        # Input box and send button
        input_bar = tk.Frame(chat_frame, bg=CARD_BG)
        input_bar.pack(fill=tk.X)

        self.entry_prompt = tk.Entry(
            input_bar,
            font=("Segoe UI", 11),
            bg="#1e293b",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            bd=0,
            relief=tk.FLAT,
        )
        self.entry_prompt.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 10))
        self.entry_prompt.bind("<Return>", lambda e: self._send_quick_prompt())

        btn_send = tk.Button(
            input_bar,
            text="Invia Prompt 🚀",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT_BLUE,
            fg="white",
            activebackground="#2563eb",
            activeforeground="white",
            bd=0,
            padx=16,
            pady=5,
            cursor="hand2",
            command=self._send_quick_prompt,
        )
        btn_send.pack(side=tk.RIGHT)

    def _create_card(self, parent, row, col, title, initial_value, color):
        frame = tk.Frame(parent, bg=CARD_BG, bd=1, relief=tk.SOLID, padx=12, pady=10)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        lbl_t = tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"), fg=color, bg=CARD_BG)
        lbl_t.pack(anchor=tk.W)

        lbl_v = tk.Label(
            frame,
            text=initial_value,
            font=("Segoe UI", 10),
            fg=TEXT_MAIN,
            bg=CARD_BG,
            wraplength=340,
            justify=tk.LEFT,
        )
        lbl_v.pack(anchor=tk.W, pady=(4, 0))
        return lbl_v

    def _create_btn(self, parent, row, col, text, color, command):
        btn = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 9, "bold"),
            bg=BUTTON_BG,
            fg=TEXT_MAIN,
            activebackground=color,
            activeforeground="white",
            bd=1,
            relief=tk.GROOVE,
            padx=10,
            pady=8,
            cursor="hand2",
            command=command,
        )
        btn.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)

    def _start_metrics_updater(self):
        def loop():
            while True:
                self._update_metrics()
                time.sleep(4)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _update_metrics(self):
        # 1. Check Server API
        server_status = "🔴 Offline (serve.py non rilevato)"
        try:
            t0 = time.time()
            res = requests.get(f"{self.api_base}/v1/models", timeout=1.5)
            ms = int((time.time() - t0) * 1000)
            if res.status_code == 200:
                server_status = f"🟢 Online (Port 8000 • {ms}ms latency)"
        except Exception:
            pass

        # 2. Check Dataset count
        dataset_status = "In calcolo..."
        jsonl_path = os.path.join(self.workspace_dir, "hf_export_dataset", "lumina_distilled_dialogues.jsonl")
        try:
            if os.path.exists(jsonl_path):
                # Conta rapida righe
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    cnt = sum(1 for _ in f)
                dataset_status = f"🔥 {cnt:,} Campioni distillati • GPU RTX 5060"
            else:
                dataset_status = "Dataset non trovato"
        except Exception:
            dataset_status = "Pronto all'addestramento"

        # Update GUI on main thread safely
        def apply():
            self.card_server.config(text=server_status)
            self.card_train.config(text=dataset_status)

        self.root.after(0, apply)

    def _send_quick_prompt(self):
        prompt = self.entry_prompt.get().strip()
        if not prompt:
            return

        self.entry_prompt.delete(0, tk.END)
        self._append_chat(f"Tu: {prompt}\n", "#60a5fa")

        def worker():
            try:
                t0 = time.time()
                is_coding = any(k in prompt.lower() for k in ["python", "codice", "funzione", "def ", "classe", "lista", "scrivi", "programma", "errore", "sql", "git"])
                target_model = "lumina-coder" if is_coding else "lumina-2.75m"
                payload = {
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 180,
                }
                res = requests.post(f"{self.api_base}/v1/chat/completions", json=payload, timeout=8)
                latency = int((time.time() - t0) * 1000)

                if res.status_code == 200:
                    data = res.json()
                    ans = data["choices"][0]["message"]["content"]
                    blocked = "Lumina Sentinel" in ans or "bloccata" in ans.lower()

                    def update_ui():
                        if blocked:
                            self.lbl_chat_badge.config(text="🚨 Sentinel: INTERVENTO DI BLOCCO", fg=ACCENT_RED)
                            self._append_chat(f"🛡️ Lumina Sentinel: {ans}\n\n", ACCENT_RED)
                        else:
                            self.lbl_chat_badge.config(text="🛡️ Sentinel: PROMPT SICURO", fg=ACCENT_GREEN)
                            self._append_chat(f"Lumina ({latency}ms): {ans}\n\n", "#34d399")

                    self.root.after(0, update_ui)
                else:
                    self.root.after(0, lambda: self._append_chat(f"[ERRORE HTTP {res.status_code}]\n\n", ACCENT_RED))
            except Exception as e:
                self.root.after(0, lambda: self._append_chat(f"[ERRORE CONNETTORE]: {e}\n\n", ACCENT_RED))

        threading.Thread(target=worker, daemon=True).start()

    def _append_chat(self, msg, color):
        self.txt_history.configure(state=tk.NORMAL)
        tag_name = f"color_{len(msg)}_{int(time.time() * 1000)}"
        self.txt_history.tag_configure(tag_name, foreground=color)
        self.txt_history.insert(tk.END, msg, (tag_name,))
        self.txt_history.see(tk.END)
        self.txt_history.configure(state=tk.DISABLED)

    # Launchers
    def _run_bat(self, bat_name):
        path = os.path.join(self.workspace_dir, bat_name)
        if os.path.exists(path):
            subprocess.Popen(["cmd.exe", "/c", "start", bat_name], cwd=self.workspace_dir, shell=True)
        else:
            messagebox.showerror("File non trovato", f"Non ho trovato il file {bat_name}")

    def _launch_chat_sentinel(self):
        self._run_bat("AVVIA_CHAT_PROTETTA.bat")

    def _launch_rag(self):
        self._run_bat("PARLA_CON_DOCUMENTI.bat")

    def _launch_voice(self):
        self._run_bat("AVVIA_LUMINA_VOCALE.bat")

    def _launch_agent(self):
        self._run_bat("AVVIA_LUMINA_AGENT.bat")

    def _launch_telegram(self):
        self._run_bat("AVVIA_BOT_TELEGRAM.bat")

    def _launch_wasm(self):
        self._run_bat("APRI_LUMINA_BROWSER_OFFLINE.bat")

    def _launch_benchmark(self):
        self._run_bat("ESEGUI_BENCHMARK.bat")

    def _setup_vscode(self):
        self._run_bat("ATTIVA_LUMINA_VSCODE.bat")

    def _create_ollama(self):
        self._run_bat("CREA_MODELLO_OLLAMA.bat")

    def _launch_spotlight(self):
        self._run_bat("AVVIA_LUMINA_SPOTLIGHT.bat")

    def _launch_sandbox(self):
        self._run_bat("AVVIA_CODE_SANDBOX.bat")

    def _launch_sync_hf(self):
        self._run_bat("AVVIA_AUTO_SYNC_HF.bat")

    def _open_browser(self):
        webbrowser.open("http://localhost:8000")


def main():
    root = tk.Tk()
    app = LuminaControlCenter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
