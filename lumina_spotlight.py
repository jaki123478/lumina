"""
Lumina Spotlight - Widget Desktop Fluttuante a Comparsa Rapida per Windows.
Ispirato a Raycast e macOS Spotlight: premi invio per chiedere codice o comandi,
risposta istantanea in dark mode, Esc per chiudere.
"""

import os
import sys
import time
import threading
import requests
import tkinter as tk
from tkinter import font

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BG_COLOR = "#0c1017"
INPUT_BG = "#161f2e"
TEXT_COLOR = "#f0f6fc"
ACCENT_CYAN = "#00f0ff"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
API_URL = "http://localhost:8000/v1/chat/completions"


class LuminaSpotlight:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lumina Spotlight")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True) # Senza bordi finestra tradizionali
        self.root.configure(bg="#21262d")

        # Dimensioni e centratura
        self.width = 680
        self.min_height = 80
        self.max_height = 420
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.x = (screen_w - self.width) // 2
        self.y = int(screen_h * 0.22)

        self.root.geometry(f"{self.width}x{self.min_height}+{self.x}+{self.y}")

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        # Bordo esterno sottile Cyberpunk
        self.outer_frame = tk.Frame(self.root, bg="#30363d", padx=1, pady=1)
        self.outer_frame.pack(fill=tk.BOTH, expand=True)

        self.main_frame = tk.Frame(self.outer_frame, bg=BG_COLOR, padx=14, pady=12)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Barra superiore: Icona + Input + Badge
        top_bar = tk.Frame(self.main_frame, bg=BG_COLOR)
        top_bar.pack(fill=tk.X)

        lbl_icon = tk.Label(top_bar, text="⚡", font=("Segoe UI", 16), fg=ACCENT_CYAN, bg=BG_COLOR)
        lbl_icon.pack(side=tk.LEFT, padx=(0, 10))

        self.entry = tk.Entry(
            top_bar,
            font=("Segoe UI", 13),
            bg=INPUT_BG,
            fg=TEXT_COLOR,
            insertbackground=ACCENT_CYAN,
            bd=0,
            relief=tk.FLAT
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 10))
        self.entry.focus_set()

        self.lbl_badge = tk.Label(top_bar, text="ESC chiudi", font=("Segoe UI", 8), fg="#6e7681", bg=BG_COLOR)
        self.lbl_badge.pack(side=tk.RIGHT)

        # Area risposta (espandibile)
        self.response_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        
        self.txt_output = tk.Text(
            self.response_frame,
            font=("Consolas", 10),
            bg="#0d1117",
            fg=TEXT_COLOR,
            bd=0,
            padx=10,
            pady=10,
            wrap=tk.WORD,
            height=14
        )
        self.txt_output.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.txt_output.configure(state=tk.DISABLED)

    def _bind_events(self):
        self.entry.bind("<Return>", lambda e: self._on_submit())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        # Permetti trascinamento della finestra
        self.main_frame.bind("<Button-1>", self._start_move)
        self.main_frame.bind("<B1-Motion>", self._on_move)

    def _start_move(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_move(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def _expand_window(self):
        self.response_frame.pack(fill=tk.BOTH, expand=True)
        self.root.geometry(f"{self.width}x{self.max_height}+{self.x}+{self.y}")

    def _on_submit(self):
        query = self.entry.get().strip()
        if not query:
            return

        self._expand_window()
        self.lbl_badge.config(text="● Elaborazione...", fg=ACCENT_CYAN)
        self.txt_output.configure(state=tk.NORMAL)
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, "Interrogazione Lumina...\n")
        self.txt_output.configure(state=tk.DISABLED)

        def worker():
            t0 = time.time()
            try:
                # Scegli se indirizzare a Lumina Coder o general
                is_code = any(w in query.lower() for w in ["python", "funzione", "codice", "def ", "classe", "errore", "lista", "dict", "script"])
                target_model = "lumina-coder" if is_code else "lumina-2.75m"

                payload = {
                    "model": target_model,
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 150
                }
                res = requests.post(API_URL, json=payload, timeout=8)
                latency = int((time.time() - t0) * 1000)

                if res.status_code == 200:
                    ans = res.json()["choices"][0]["message"]["content"].strip()
                    def update():
                        self.lbl_badge.config(text=f"✓ Risposta in {latency}ms ({target_model})", fg=ACCENT_GREEN)
                        self.txt_output.configure(state=tk.NORMAL)
                        self.txt_output.delete("1.0", tk.END)
                        self.txt_output.insert(tk.END, ans)
                        self.txt_output.configure(state=tk.DISABLED)
                    self.root.after(0, update)
                else:
                    self.root.after(0, lambda: self._show_error(f"Errore HTTP {res.status_code}"))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(f"Server non raggiungibile: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_error(self, err):
        self.lbl_badge.config(text="Errore", fg=ACCENT_RED)
        self.txt_output.configure(state=tk.NORMAL)
        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, f"[!] {err}")
        self.txt_output.configure(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = LuminaSpotlight()
    app.run()
