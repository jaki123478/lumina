import json, queue, threading, uuid, urllib.request, urllib.error, re, ctypes, os, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Gli eseguibili senza console hanno stdout/stderr a None; alcune librerie UI
# scrivono comunque gli avvisi su questi stream durante l'import.
if getattr(sys, "frozen", False):
    class _NullStream:
        def write(self, _text): return 0
        def flush(self): pass
    if sys.stdout is None: sys.stdout = _NullStream()
    if sys.stderr is None: sys.stderr = _NullStream()

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#090D17"; SIDE = "#0E1526"; CARD = "#151F35"; HOVER = "#202D4A"
USER = "#3B5BDB"; ACCENT = "#6C8CFF"; TEXT = "#E6EAF2"; MUTED = "#8A94A8"
OK = "#4ADE80"; ERR = "#F87171"
DATA_FILE = Path.home() / ".lumina" / "sessions.json"
API_ENDPOINTS = (
    "http://127.0.0.1:8000/v1/chat/completions",
    "http://127.0.0.1:7020/api/chat",
)
ROOT_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODES = ["Generale", "Lumina Next", "Codice", "Immagini", "Genera immagine", "Documenti", "Ricerca web", "Traduttore", "Riassunto"]

def load_sessions():
    try: return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        fallback = Path(__file__).resolve().parents[1] / "data" / "desktop_sessions.json"
        try:
            data = json.loads(fallback.read_text(encoding="utf-8"))
            # Migrazione delle risposte del vecchio prototipo che potevano
            # apparire nella cronologia come risposta di ascolto errata.
            for session in data.values():
                for message in session.get("messages", []):
                    if message.get("content", "").strip().lower() == "sì, lumina mi sento come una ruota.":
                        message["content"] = "Sì, ti sento. Puoi parlare: sono in ascolto."
            return data
        except Exception: return {}

def save_sessions(data):
    try:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        # Alcune build Windows bloccano la cartella profilo; la chat deve
        # comunque restare utilizzabile e salvare localmente accanto all’app.
        fallback = Path(__file__).resolve().parents[1] / "data" / "desktop_sessions.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

class LuminaBackend:
    @staticmethod
    def _read_attachment(path):
        """Legge localmente gli allegati testuali senza inviarli online."""
        if not path:
            return ""
        p = Path(path)
        try:
            suffix = p.suffix.lower()
            if suffix in {".txt", ".md", ".csv", ".json", ".py", ".html", ".css", ".js"}:
                return p.read_text(encoding="utf-8", errors="replace")[:50000]
            if suffix == ".pdf":
                from pypdf import PdfReader
                return "\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)[:50000]
            if suffix == ".docx":
                from docx import Document
                return "\n".join(par.text for par in Document(str(p)).paragraphs)[:50000]
        except Exception:
            return ""
        return ""

    def stream(self, messages, mode, attachment=None):
        base_rules = "Segui esattamente l'ultima richiesta dell'utente. Rispondi solo a ciò che viene chiesto, senza cambiare argomento, inventare fatti o aggiungere attività non richieste. Se la richiesta è ambigua, fai una sola domanda di chiarimento. Mantieni la risposta breve e ordinata."
        system = {
            "Generale": base_rules + " Rispondi in italiano naturale e preciso.",
            "Lumina Next": base_rules + " Sei Lumina Next, assistente locale. Segnala i limiti quando non sei sicura.",
            "Codice": base_rules + " Sei un assistente di coding professionale. Spiega e verifica il codice.",
            "Immagini": base_rules + " Analizza l'immagine allegata solo per ciò che è effettivamente disponibile. Se il motore non può vedere i pixel, dichiaralo chiaramente.",
            "Genera immagine": base_rules + " Prepara un prompt chiaro per il generatore immagini locale.",
            "Documenti": base_rules + " Analizza il documento allegato con attenzione. Usa solo il contenuto disponibile e distingui ciò che è verificato.",
            "Ricerca web": base_rules + " Cerca fonti attuali, cita i collegamenti e distingui fatti verificati da risultati da controllare.",
            "Traduttore": base_rules + " Traduci con accuratezza mantenendo tono e formattazione.",
            "Riassunto": base_rules + " Riassumi in modo fedele e ordinato, senza inventare dati.",
        }[mode]
        context_messages = list(messages)
        attachment_text = self._read_attachment(attachment)
        if attachment_text:
            context_messages.append({"role": "user", "content": "Contenuto dell'allegato locale:\n" + attachment_text})
        elif attachment and Path(attachment).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
            # Il motore testuale non riceve l'immagine online: forniamo solo
            # metadati verificabili e chiediamo di non inventare una descrizione.
            try:
                from PIL import Image
                with Image.open(attachment) as image:
                    image_info = f"Immagine locale allegata: {Path(attachment).name}; formato {image.format}; dimensioni {image.width}x{image.height}. I pixel non sono disponibili al motore testuale."
            except Exception:
                image_info = f"Immagine locale allegata: {Path(attachment).name}. I pixel non sono disponibili al motore testuale."
            context_messages.append({"role": "user", "content": image_info + " Descrivi solo ciò che puoi verificare."})
        payload = {"model": "lumina-next" if mode == "Lumina Next" else ("lumina-coder" if mode == "Codice" else "lumina-assisted"),
                   "messages": [{"role":"system","content":system}] + context_messages,
                   "max_tokens": 700,
                   "offline": mode != "Ricerca web",
                   "web": mode == "Ricerca web",
                   "research": mode == "Ricerca web"}
        if attachment: payload["attachment"] = attachment
        # Risposte di controllo della voce: non devono passare dal modello,
        # altrimenti una frase come "mi senti?" può ricevere una risposta
        # casuale mentre il motore sta caricando.
        last_raw = str(messages[-1].get("content", "")).strip() if messages else ""
        last = last_raw.lower()
        img_cmd = re.match(
            r"^(?:per favore\s+)?(?:genera|crea|disegna|dipingi)\s+(?:un['\u2019]?\s*immagine|una\s+foto|un\s+disegno)(?:\s+(?:di|con|che\s+mostra|su))?\s*(.*)$",
            last_raw,
            flags=re.IGNORECASE,
        )
        if mode == "Genera immagine" or img_cmd:
            from lumina_image_gen import generate_image
            img_prompt = img_cmd.group(1).strip() if (img_cmd and img_cmd.group(1).strip()) else last_raw
            try:
                output = generate_image(img_prompt)
                yield f"Immagine generata localmente da Lumina Image e salvata in:\n{output}"
            except Exception as exc:
                yield f"Generazione immagini non disponibile: {exc}"
            return
        if re.search(r"\b(mi senti|mi ascolti|ci sei)\b", last):
            answer = "Sì, ti sento. Puoi parlare: sono in ascolto."
            for word in answer.split(" "): yield word + " "
            return
        if re.fullmatch(r"(?:ciao|salve|buongiorno|buonasera)[!.? ]*", last):
            answer = "Ciao! Sono Lumina. Come posso aiutarti?"
            for word in answer.split(" "): yield word + " "
            return
        answer = ""
        for endpoint in API_ENDPOINTS:
            try:
                body = payload if endpoint.endswith("/v1/chat/completions") else {
                    "messages": payload["messages"], "max_tokens": payload["max_tokens"]
                }
                req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                              headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(req, timeout=4) as res:
                    data = json.loads(res.read().decode("utf-8"))
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or data.get("reply", "")
                if answer: break
            except Exception:
                continue
        if not answer:
            # Fallback per l'eseguibile avviato da solo: non richiede il server API.
            from lumina_general import generate
            local_messages = [{"role":"system","content":system}] + context_messages
            answer, _meta = generate(local_messages, max_tokens=160)
        if not answer: answer = "Lumina non ha restituito una risposta valida."
        for word in answer.split(" "):
            yield word + " "

class LuminaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Lumina — assistente locale")
        self.geometry("1400x900"); self.minsize(1000, 680); self.configure(fg_color=BG)
        # La modalità zoomata di Tk su Windows può riaprire la finestra
        # minimizzata; una geometria esplicita è più stabile per l'utente.
        self.state("normal")
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.sessions = load_sessions(); self.current = None; self.busy = False; self.cancel_event = threading.Event()
        self.q = queue.Queue(); self.stream_text = ""; self.stream_label = None
        self.bubbles = []; self.attachment = None; self._orb_phase = 0; self.voice_enabled = False
        self.voice_panel = None; self.voice_speaking = False; self.voice_listening = False
        # Precarica la voce in background: la prima risposta non deve aspettare
        # l'inizializzazione del modello Piper.
        threading.Thread(target=self._prewarm_voice, daemon=True).start()
        self._build_sidebar(); self._build_main()
        # Avvia sempre su una composizione pulita; le conversazioni precedenti
        # restano nella sidebar e si possono riaprire senza sporcare il primo
        # impatto dell'app.
        self.new_chat()
        self.deiconify(); self.lift(); self.focus_force()
        # Il focus immediato può essere sovrascritto dal terminale su Windows;
        # ripeterlo dopo il primo ciclo dell'event loop rende l'avvio affidabile.
        self.after(250, self._bring_to_front)
        self.after(40, self._poll); self.after(700, self._refresh_health); self.bind("<Configure>", self._resize)
        self.bind("<Escape>", lambda _event: self.send() if self.busy else None)
        self.bind("<Control-n>", lambda _event: (self.new_chat(), "break")[1])

    def _bring_to_front(self):
        try:
            self.deiconify(); self.lift(); self.attributes("-topmost", True)
            self.after(120, lambda: self.attributes("-topmost", False))
            self.focus_force()
        except tk.TclError:
            pass

    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=SIDE, border_width=1, border_color="#1D2A46")
        side.grid(row=0, column=0, sticky="nsew"); side.grid_propagate(False)
        side.grid_rowconfigure(4, weight=1); side.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(side, text="✦  LUMINA", font=("Segoe UI", 24, "bold"), text_color=TEXT).grid(row=0,column=0,padx=24,pady=(28,0),sticky="w")
        ctk.CTkLabel(side, text="Assistente locale · privato", font=("Segoe UI", 12), text_color="#A8B4CC").grid(row=1,column=0,padx=24,pady=(0,22),sticky="w")
        ctk.CTkButton(side,text="＋  Nuova chat",height=42,corner_radius=11,fg_color=ACCENT,hover_color="#566FD6",text_color=BG,font=("Segoe UI",14,"bold"),command=self.new_chat).grid(row=2,column=0,padx=16,pady=(0,14),sticky="ew")
        workspace = ctk.CTkFrame(side, fg_color="transparent")
        workspace.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="ew")
        workspace.grid_columnconfigure((0, 1, 2), weight=1)
        for col, (label, active) in enumerate((("Chat", True), ("Coding", False), ("Documenti", False))):
            ctk.CTkButton(workspace, text=label, height=32, corner_radius=8,
                          fg_color=HOVER if active else "transparent",
                          hover_color=HOVER, text_color=TEXT if active else MUTED,
                          font=("Segoe UI", 11, "bold" if active else "normal"),
                          command=lambda name=label: self._workspace_changed(name)).grid(row=0, column=col, padx=2, sticky="ew")
        self.session_list=ctk.CTkScrollableFrame(side,fg_color="transparent",label_text="  CHAT RECENTI",label_text_color=MUTED,label_fg_color="transparent",label_font=("Segoe UI",11,"bold"),scrollbar_button_color=HOVER)
        self.session_list.grid(row=4,column=0,padx=12,sticky="nsew")
        ctk.CTkButton(side,text="⌫  Elimina chat",height=34,corner_radius=8,fg_color="transparent",hover_color=HOVER,text_color=MUTED,anchor="w",command=self.delete_chat).grid(row=5,column=0,padx=16,pady=(8,2),sticky="ew")
        ctk.CTkButton(side,text="⇩  Esporta chat",height=34,corner_radius=8,fg_color="transparent",hover_color=HOVER,text_color=MUTED,anchor="w",command=self.export_chat).grid(row=6,column=0,padx=16,pady=(0,2),sticky="ew")
        ctk.CTkButton(side,text="⚙  Impostazioni",height=34,corner_radius=8,fg_color="transparent",hover_color=HOVER,text_color=MUTED,anchor="w",command=self.open_settings).grid(row=7,column=0,padx=16,pady=(0,16),sticky="ew")
        profile=ctk.CTkFrame(side,fg_color="#131C30",corner_radius=14,border_width=1,border_color="#202D4A")
        profile.grid(row=8,column=0,padx=14,pady=(0,16),sticky="ew")
        ctk.CTkLabel(profile,text="J",width=34,height=34,corner_radius=17,fg_color=ACCENT,text_color=BG,font=("Segoe UI",16,"bold")).grid(row=0,column=0,padx=(12,9),pady=11)
        info=ctk.CTkFrame(profile,fg_color="transparent"); info.grid(row=0,column=1,sticky="w",pady=9)
        ctk.CTkLabel(info,text="Utente locale",font=("Segoe UI",12,"bold"),text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(info,text="Dati salvati sul PC",font=("Segoe UI",10),text_color=OK).pack(anchor="w")

    def _build_main(self):
        main=ctk.CTkFrame(self,fg_color=BG,corner_radius=0); main.grid(row=0,column=1,sticky="nsew"); main.grid_columnconfigure(0,weight=1); main.grid_rowconfigure(1,weight=1)
        head=ctk.CTkFrame(main,fg_color="transparent",height=68); head.grid(row=0,column=0,sticky="ew",padx=34,pady=(12,0)); head.grid_columnconfigure(0,weight=1)
        self.title_lbl=ctk.CTkLabel(head,text="Nuova chat",font=("Segoe UI",21,"bold"),text_color=TEXT); self.title_lbl.grid(row=0,column=0,sticky="w")
        ctk.CTkLabel(head,text="● Locale",font=("Segoe UI",12,"bold"),text_color=OK).grid(row=0,column=1,padx=(0,18),sticky="e")
        ctk.CTkLabel(head,text="Modello locale",font=("Segoe UI",12),text_color=MUTED).grid(row=0,column=2,padx=(0,18),sticky="e")
        self.mode=ctk.CTkOptionMenu(head,values=MODES,width=160,height=36,fg_color=CARD,button_color=HOVER,button_hover_color=ACCENT,corner_radius=9,command=self._mode_changed); self.mode.grid(row=0,column=3,sticky="e")
        self.live_head_btn=ctk.CTkButton(head,text="◉  Voce live",width=112,height=36,corner_radius=9,fg_color="#17213A",hover_color=HOVER,text_color=TEXT,command=self.voice_input); self.live_head_btn.grid(row=0,column=4,padx=(10,0),sticky="e")
        ctk.CTkFrame(main,height=1,fg_color="#18223A",corner_radius=0).grid(row=0,column=0,sticky="sew",padx=32,pady=(0,0))
        self.chat=ctk.CTkScrollableFrame(main,fg_color="#0C1220",corner_radius=18,border_width=1,border_color="#182641",scrollbar_button_color=HOVER); self.chat.grid(row=1,column=0,sticky="nsew",padx=26,pady=(10,0)); self.chat.grid_columnconfigure(0,weight=1)
        self.hero = ctk.CTkFrame(self.chat, fg_color="transparent")
        self.hero.grid(row=0, column=0, sticky="nsew", pady=(86, 30))
        self.hero.grid_columnconfigure(0, weight=1)
        self.hero_orb = tk.Canvas(self.hero, width=180, height=180, bg=BG, highlightthickness=0)
        self.hero_orb.grid(row=0, column=0, pady=(0, 18))
        self._draw_orb(0)
        ctk.CTkLabel(self.hero, text="Che cosa vuoi fare?", font=("Segoe UI", 30, "bold"), text_color=TEXT).grid(row=1,column=0,pady=(0,8))
        ctk.CTkLabel(self.hero, text="Scrivi a Lumina, lavora sul codice, allega un’immagine o parla.", font=("Segoe UI", 14), text_color="#A8B4CC").grid(row=2,column=0,pady=(0,26))
        quick=ctk.CTkFrame(self.hero, fg_color="transparent"); quick.grid(row=3,column=0)
        actions=(("⌘", "Coding", "Scrivi e verifica codice", "Aiutami a scrivere e verificare del codice"), ("▧", "Immagini", "Analizza un allegato", "Analizza questa immagine"), ("✦", "Crea immagine", "Generazione locale", "__IMAGE_GEN__"), ("⌁", "Ricerca online", "Informazioni aggiornate", "Cerca sul web informazioni aggiornate"), ("◉", "Voce live", "Parla con Lumina", "__VOICE__"))
        for i,(icon,title,sub,query) in enumerate(actions):
            card=ctk.CTkFrame(quick,fg_color=CARD,corner_radius=13,border_width=1,border_color="#263657",width=188,height=78); card.grid(row=0,column=i,padx=5); card.grid_propagate(False)
            icon_label=ctk.CTkLabel(card,text=icon,font=("Segoe UI",18,"bold"),text_color=ACCENT); icon_label.grid(row=0,column=0,rowspan=2,padx=(12,8),pady=10)
            title_label=ctk.CTkLabel(card,text=title,font=("Segoe UI",12,"bold"),text_color=TEXT,anchor="w"); title_label.grid(row=0,column=1,sticky="w",padx=(0,8),pady=(10,0))
            sub_label=ctk.CTkLabel(card,text=sub,font=("Segoe UI",9),text_color=MUTED,anchor="w"); sub_label.grid(row=1,column=1,sticky="w",padx=(0,8),pady=(0,10))
            for widget in (card, icon_label, title_label, sub_label): widget.bind("<Button-1>",lambda _e,q=query:self._quick(q))
        bar=ctk.CTkFrame(main,fg_color=CARD,corner_radius=18,border_width=1,border_color="#243253"); bar.grid(row=2,column=0,sticky="ew",padx=34,pady=(10,5)); bar.grid_columnconfigure(0,weight=1)
        self.entry=ctk.CTkTextbox(bar,height=62,fg_color="transparent",font=("Segoe UI",15),text_color=MUTED,wrap="word",border_width=0); self.entry.grid(row=0,column=0,sticky="ew",padx=(16,4),pady=7); self.entry.insert("1.0","Scrivi a Lumina…"); self.entry.bind("<FocusIn>",self._clear_placeholder); self.entry.bind("<Return>",self._enter)
        ctk.CTkButton(bar,text="＋",width=42,height=42,corner_radius=11,fg_color=HOVER,hover_color="#26335C",command=self.attach_file).grid(row=0,column=1,padx=3)
        self.voice_btn=ctk.CTkButton(bar,text="🎤",width=46,height=42,corner_radius=11,fg_color=HOVER,hover_color="#26335C",command=self.voice_input); self.voice_btn.grid(row=0,column=2,padx=3)
        self.send_btn=ctk.CTkButton(bar,text="Invia  ➤",width=100,height=42,corner_radius=11,fg_color=ACCENT,hover_color="#566FD6",text_color=BG,font=("Segoe UI",14,"bold"),command=self.send); self.send_btn.grid(row=0,column=3,padx=(3,10))
        self.status=ctk.CTkLabel(main,text="●  Lumina locale pronta",text_color=OK,font=("Segoe UI",12)); self.status.grid(row=3,column=0,sticky="w",padx=32,pady=(0,12))

    def refresh_sessions(self):
        for w in self.session_list.winfo_children(): w.destroy()
        for sid in reversed(list(self.sessions)):
            ctk.CTkButton(self.session_list,text=self.sessions[sid]["title"],anchor="w",height=36,corner_radius=8,fg_color=HOVER if sid==self.current else "transparent",hover_color=HOVER,text_color=TEXT if sid==self.current else MUTED,command=lambda s=sid:self.open_session(s)).pack(fill="x",pady=2)
    def new_chat(self):
        if self.busy:return
        sid=uuid.uuid4().hex[:8]; self.sessions[sid]={"title":"Nuova chat","messages":[]}; self.open_session(sid)
    def open_session(self,sid):
        if self.busy:return
        self.current=sid; self.title_lbl.configure(text=self.sessions[sid]["title"])
        for w in self.chat.winfo_children():w.destroy()
        if not self.sessions[sid]["messages"]:
            self.hero = ctk.CTkFrame(self.chat, fg_color="transparent")
            self.hero.grid(row=0,column=0,sticky="nsew",pady=(86,30)); self.hero.grid_columnconfigure(0,weight=1)
            self.hero_orb=tk.Canvas(self.hero,width=180,height=180,bg=BG,highlightthickness=0); self.hero_orb.grid(row=0,column=0,pady=(0,18)); self._draw_orb(0)
            ctk.CTkLabel(self.hero,text="Che cosa vuoi fare?",font=("Segoe UI",30,"bold"),text_color=TEXT).grid(row=1,column=0,pady=(0,8))
            ctk.CTkLabel(self.hero,text="Scrivi a Lumina, lavora sul codice, allega un’immagine o parla.",font=("Segoe UI",14),text_color="#A8B4CC").grid(row=2,column=0,pady=(0,26))
            quick=ctk.CTkFrame(self.hero,fg_color="transparent"); quick.grid(row=3,column=0)
            actions=(("⌘","Coding","Scrivi e verifica codice","Aiutami a scrivere e verificare del codice"),("▧","Immagini","Analizza un allegato","Analizza questa immagine"),("✦","Crea immagine","Generazione locale","__IMAGE_GEN__"),("⌁","Ricerca online","Informazioni aggiornate","Cerca sul web informazioni aggiornate"),("◉","Voce live","Parla con Lumina","__VOICE__"))
            for i,(icon,title,sub,query) in enumerate(actions):
                card=ctk.CTkFrame(quick,fg_color=CARD,corner_radius=13,border_width=1,border_color="#263657",width=188,height=78); card.grid(row=0,column=i,padx=5); card.grid_propagate(False)
                icon_label=ctk.CTkLabel(card,text=icon,font=("Segoe UI",18,"bold"),text_color=ACCENT); icon_label.grid(row=0,column=0,rowspan=2,padx=(12,8),pady=10)
                title_label=ctk.CTkLabel(card,text=title,font=("Segoe UI",12,"bold"),text_color=TEXT,anchor="w"); title_label.grid(row=0,column=1,sticky="w",padx=(0,8),pady=(10,0))
                sub_label=ctk.CTkLabel(card,text=sub,font=("Segoe UI",9),text_color=MUTED,anchor="w"); sub_label.grid(row=1,column=1,sticky="w",padx=(0,8),pady=(0,10))
                for widget in (card, icon_label, title_label, sub_label): widget.bind("<Button-1>",lambda _e,q=query:self._quick(q))
        self.bubbles=[]
        if not self.sessions[sid]["messages"]:
            self.entry.delete("1.0","end"); self.entry.insert("1.0","Scrivi a Lumina…"); self.entry.configure(text_color=MUTED)
        for m in self.sessions[sid]["messages"]: self.add_bubble(m["role"],m["content"])
        self.refresh_sessions(); self.entry.focus()
    def delete_chat(self):
        if self.busy or not self.current:return
        self.sessions.pop(self.current,None); save_sessions(self.sessions); self.new_chat() if not self.sessions else self.open_session(next(reversed(self.sessions)))

    def open_settings(self):
        win=ctk.CTkToplevel(self); win.title("Lumina · Impostazioni"); win.geometry("390x420"); win.resizable(False,False); win.configure(fg_color=BG); win.transient(self)
        ctk.CTkLabel(win,text="Impostazioni",font=("Segoe UI",24,"bold"),text_color=TEXT).pack(anchor="w",padx=28,pady=(28,4))
        ctk.CTkLabel(win,text="Controlli locali dell’app",font=("Segoe UI",12),text_color=MUTED).pack(anchor="w",padx=28,pady=(0,22))
        card=ctk.CTkFrame(win,fg_color=CARD,corner_radius=14); card.pack(fill="x",padx=22,pady=7)
        ctk.CTkLabel(card,text="Motore",font=("Segoe UI",12,"bold"),text_color=TEXT).pack(anchor="w",padx=18,pady=(16,2)); ctk.CTkLabel(card,text="Lumina locale · dati sul PC",font=("Segoe UI",12),text_color=OK).pack(anchor="w",padx=18,pady=(0,16))
        voice=ctk.CTkSwitch(win,text="Riproduci risposte con la voce",text_color=TEXT,command=lambda:self.voice_input()); voice.pack(anchor="w",padx=30,pady=15); voice.select() if self.voice_enabled else voice.deselect()
        ctk.CTkButton(win,text="Chiudi",width=120,height=36,corner_radius=18,fg_color=HOVER,hover_color=ACCENT,command=win.destroy).pack(pady=22)

    def add_bubble(self,role,text):
        text = re.sub(r"\bqwen(?:\s*2(?:\.5)?(?:-coder)?(?:-instruct)?)?\b", "modello locale", str(text), flags=re.IGNORECASE)
        user=role=="user"; row=ctk.CTkFrame(self.chat,fg_color="transparent"); row.grid(sticky="ew",pady=9,padx=16); row.grid_columnconfigure(0 if not user else 1,weight=1)
        col=ctk.CTkFrame(row,fg_color="transparent"); col.grid(row=0,column=1 if user else 0,sticky="e" if user else "w")
        ctk.CTkLabel(col,text="Tu" if user else "Lumina",font=("Segoe UI",11,"bold"),text_color=MUTED).pack(anchor="e" if user else "w",padx=6)
        bubble=ctk.CTkFrame(col,fg_color=USER if user else CARD,corner_radius=16,border_width=1,border_color="#4B6DE0" if user else "#202C49"); bubble.pack(anchor="e" if user else "w")
        lbl=ctk.CTkLabel(bubble,text=text,justify="left",anchor="w",font=("Segoe UI",15),text_color=TEXT,wraplength=max(300,int(self.chat.winfo_width()*.62))); lbl.pack(padx=18,pady=13); self.bubbles.append(lbl)
        if not user:
            self._maybe_attach_image_preview(bubble, text)
            actions = ctk.CTkFrame(col, fg_color="transparent")
            actions.pack(anchor="w", pady=(2,0))
            ctk.CTkButton(actions,text="Copia",width=54,height=23,corner_radius=6,fg_color="transparent",hover_color=HOVER,text_color=MUTED,font=("Segoe UI",11),command=lambda:self._copy(lbl)).pack(side="left")
            ctk.CTkButton(actions,text="Rigenera",width=66,height=23,corner_radius=6,fg_color="transparent",hover_color=HOVER,text_color=MUTED,font=("Segoe UI",10),command=lambda:self._regenerate(lbl)).pack(side="left", padx=2)
            ctk.CTkButton(actions,text="Utile",width=54,height=23,corner_radius=6,fg_color="transparent",hover_color="#1D4732",text_color="#91D8AD",font=("Segoe UI",10),command=lambda:self._feedback(lbl,True)).pack(side="left", padx=3)
            ctk.CTkButton(actions,text="Da rivedere",width=76,height=23,corner_radius=6,fg_color="transparent",hover_color="#4A2930",text_color="#E6A5AE",font=("Segoe UI",10),command=lambda:self._feedback(lbl,False)).pack(side="left")
        self._scroll(); return lbl
    def _copy(self,lbl): self.clipboard_clear(); self.clipboard_append(lbl.cget("text")); self.set_status("Copiato negli appunti")
    def _regenerate(self, _lbl):
        """Ripete l'ultima richiesta usando la modalità corrente."""
        if self.busy:
            self.set_status("Attendi la risposta in corso", ACCENT)
            return
        messages = self.sessions.get(self.current, {}).get("messages", [])
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        if not last_user:
            self.set_status("Nessuna richiesta da rigenerare", ACCENT)
            return
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", last_user)
        self.entry.configure(text_color=TEXT)
        self.send()
    def export_chat(self):
        if not self.current or not self.sessions.get(self.current, {}).get("messages"):
            self.set_status("Nessuna chat da esportare", ACCENT)
            return
        target = filedialog.asksaveasfilename(title="Esporta conversazione", defaultextension=".md", filetypes=[("Markdown", "*.md"), ("Testo", "*.txt")])
        if not target:
            return
        lines = ["# " + self.sessions[self.current].get("title", "Conversazione Lumina"), ""]
        for item in self.sessions[self.current]["messages"]:
            lines.extend(["**Tu:**" if item.get("role") == "user" else "**Lumina:**", item.get("content", ""), ""])
        Path(target).write_text("\n".join(lines), encoding="utf-8")
        self.set_status("Conversazione esportata", OK)
    def _feedback(self, lbl, approved):
        messages = self.sessions.get(self.current, {}).get("messages", [])
        user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        record = {"user": user_text, "assistant": lbl.cget("text"), "approved": approved, "topic": self.mode.get()}
        # Il feedback deve restare disponibile anche quando il server locale
        # non è avviato: il file viene poi raccolto dal motore al successivo avvio.
        try:
            feedback_file = DATA_FILE.parent / "feedback.jsonl"
            feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with feedback_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        try:
            payload = json.dumps(record).encode()
            req = urllib.request.Request("http://127.0.0.1:7020/api/feedback", data=payload, headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=2).read()
        except Exception:
            pass
        self.set_status("Feedback salvato localmente" if approved else "Segnalazione salvata localmente", OK if approved else ACCENT)
    def _resize(self,e=None):
        for b in self.bubbles:
            try:b.configure(wraplength=max(300,int(self.chat.winfo_width()*.68)))
            except Exception:pass
    def _scroll(self): self.update_idletasks(); self.chat._parent_canvas.yview_moveto(1.0)
    def set_status(self,t,color=OK): self.status.configure(text="●  "+t,text_color=color)
    def _refresh_health(self):
        """Aggiorna lo stato del motore senza bloccare la finestra."""
        def check():
            try:
                with urllib.request.urlopen("http://127.0.0.1:7020/api/status", timeout=1.5) as res:
                    data = json.loads(res.read().decode("utf-8"))
                provider = str(data.get("provider", "locale"))
                self.after(0, lambda: self.set_status("Motore locale pronto · " + provider, OK))
            except Exception:
                self.after(0, lambda: self.set_status("Modalità autonoma locale", ACCENT))
        threading.Thread(target=check, daemon=True).start()
        self.after(5000, self._refresh_health)
    def _mode_changed(self, mode):
        self.set_status("Modalità "+mode+" attiva", ACCENT)
    def _workspace_changed(self, name):
        if name == "Coding":
            self.mode.set("Codice")
            self.set_status("Area Coding pronta", ACCENT)
        elif name == "Documenti":
            self.mode.set("Documenti")
            self.set_status("Allega un documento dal pulsante +", ACCENT)
        else:
            self.mode.set("Generale")
            self.set_status("Chat pronta", OK)
    def _enter(self,e):
        if e.state & 1:return
        self.send(); return "break"
    def _clear_placeholder(self, _event=None):
        if self.entry.get("1.0","end").strip() == "Scrivi a Lumina…":
            self.entry.delete("1.0","end"); self.entry.configure(text_color=TEXT)
    def send(self):
        if self.busy:
            self.cancel_event.set()
            self.set_status("Generazione interrotta", ACCENT)
            return
        text=self.entry.get("1.0","end").strip()
        if not text or text == "Scrivi a Lumina…" or self.busy:return
        self.entry.delete("1.0","end"); sess=self.sessions[self.current]
        if hasattr(self,"hero") and self.hero.winfo_exists(): self.hero.destroy()
        if not sess["messages"]: sess["title"]=text[:32]+("…" if len(text)>32 else ""); self.title_lbl.configure(text=sess["title"])
        sess["messages"].append({"role":"user","content":text}); self.add_bubble("user",text)
        self.stream_text=""; self.stream_label=self.add_bubble("assistant","▍"); self.busy=True; self.cancel_event.clear(); self.send_btn.configure(state="normal", text="Ferma"); self.set_status("Lumina sta scrivendo…",ACCENT); self.refresh_sessions()
        threading.Thread(target=self._worker,args=(list(sess["messages"]),self.mode.get(),self.attachment),daemon=True).start(); self.attachment=None

    def _quick(self, label):
        if label == "__VOICE__":
            self.voice_input(); return
        if label == "__IMAGE_GEN__":
            self.mode.set("Genera immagine")
            self.entry.delete("1.0", "end")
            self.entry.insert("1.0", "Crea un'immagine di ")
            self.entry.configure(text_color=TEXT)
            self.entry.focus()
            self.set_status("Modalità Genera immagine pronta", ACCENT)
            return
        self.entry.insert("1.0", {"✦  Spiegami un concetto":"Spiegami un concetto in modo semplice", "⌘  Aiutami col codice":"Aiutami a scrivere e verificare del codice", "◌  Cerca sul web":"Cerca sul web informazioni aggiornate"}.get(label, "")); self.entry.focus()

    def _draw_orb(self, phase):
        if not hasattr(self,"hero_orb") or not self.hero_orb.winfo_exists(): return
        self.hero_orb.delete("all"); cx=cy=90
        for radius, color in ((72,"#131E46"),(60,"#1C2F70"),(48,"#3656B5"),(36,"#6C8CFF"),(24,"#B9C7FF")):
            wobble=(phase % 8) * 1.2 if radius < 60 else 0
            self.hero_orb.create_oval(cx-radius-wobble,cy-radius-wobble,cx+radius+wobble,cy+radius+wobble,fill=color,outline="")
        self.after(120, lambda:self._draw_orb((phase+1)%8))
    def _worker(self,hist,mode,attachment):
        try:
            for t in LuminaBackend().stream(hist,mode,attachment):
                if self.cancel_event.is_set(): break
                self.q.put(("tok",t))
            self.q.put(("done",None))
        except Exception as e:self.q.put(("err",str(e)))
    def _poll(self):
        changed=False
        try:
            while True:
                kind,val=self.q.get_nowait()
                if kind=="tok":self.stream_text+=val; changed=True
                elif kind=="done":self._finish()
                elif kind=="err":self._finish(str(val))
        except queue.Empty:pass
        if changed and self.stream_label:self.stream_label.configure(text=self.stream_text+" ▍"); self._scroll()
        self.after(40,self._poll)
    def _maybe_attach_image_preview(self, bubble_frame, text):
        try:
            m = re.search(r"([A-Za-z]:[\\/][^\r\n\"'<>|]+?\.(?:png|jpg|jpeg|webp))", str(text), flags=re.IGNORECASE)
            if not m:
                return
            img_path = Path(m.group(1).strip())
            if not img_path.exists():
                return
            from PIL import Image
            with Image.open(img_path) as raw_im:
                im = raw_im.convert("RGB").copy()
            ctk_img = ctk.CTkImage(light_image=im, dark_image=im, size=(288, 288))
            if not hasattr(self, "_img_refs"):
                self._img_refs = []
            self._img_refs.append(ctk_img)
            ctk.CTkLabel(bubble_frame, text="", image=ctk_img).pack(padx=16, pady=(0, 14))
        except Exception:
            pass
    def _finish(self,error=None):
        if error:self.stream_text="Errore: "+error; self.set_status("Errore di collegamento",ERR)
        else:self.set_status("Lumina locale pronta")
        if self.stream_label:
            self.stream_label.configure(text=self.stream_text)
            self._maybe_attach_image_preview(self.stream_label.master, self.stream_text)
        self.sessions[self.current]["messages"].append({"role":"assistant","content":self.stream_text}); save_sessions(self.sessions)
        if self.voice_enabled and not error:
            threading.Thread(target=self._speak, args=(self.stream_text,), daemon=True).start()
        self.busy=False; self.stream_label=None; self.send_btn.configure(state="normal", text="Invia  ➤"); self.entry.focus(); self._scroll()
    def attach_file(self):
        p=filedialog.askopenfilename(title="Allega un file",filetypes=[("Documenti","*.txt *.md *.pdf *.docx *.csv"),("Immagini","*.png *.jpg *.jpeg *.webp"),("Tutti i file","*.*")])
        if p:
            self.attachment=p
            if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                self.mode.set("Immagini")
                self.set_status("Immagine pronta: invia per analizzarla localmente", ACCENT)
            else:
                self.set_status("Allegato: "+Path(p).name,ACCENT)
    def voice_input(self):
        self.voice_enabled = not self.voice_enabled
        self.voice_listening = self.voice_enabled
        self.voice_btn.configure(fg_color=ACCENT if self.voice_enabled else HOVER, text="🔊" if self.voice_enabled else "🎤")
        if hasattr(self, "live_head_btn"):
            self.live_head_btn.configure(fg_color=ACCENT if self.voice_enabled else "#17213A", text="●  Voce attiva" if self.voice_enabled else "◉  Voce live", text_color=BG if self.voice_enabled else TEXT)
        if not self.voice_enabled:
            self._close_voice_panel(); self.set_status("Voce disattivata", OK); return
        self._show_voice_panel()
        self.set_status("Ascolto… parla ora", ACCENT)
        threading.Thread(target=self._listen, daemon=True).start()

    def _prewarm_voice(self):
        try:
            import sys
            sys.path.insert(0, str(ROOT_DIR))
            from lumina_voice_neural import load_voice
            load_voice()
        except Exception:
            pass

    def _show_voice_panel(self):
        if self.voice_panel and self.voice_panel.winfo_exists(): return
        self.voice_panel = ctk.CTkToplevel(self); self.voice_panel.title("Lumina · Voce live"); self.voice_panel.geometry("430x500"); self.voice_panel.resizable(False,False); self.voice_panel.configure(fg_color=BG); self.voice_panel.transient(self)
        ctk.CTkLabel(self.voice_panel,text="Voce live",font=("Segoe UI",24,"bold"),text_color=TEXT).pack(pady=(35,4))
        ctk.CTkLabel(self.voice_panel,text="Lumina ascolta solo quando attivi il microfono",font=("Segoe UI",12),text_color=MUTED).pack()
        self.voice_canvas=tk.Canvas(self.voice_panel,width=260,height=260,bg=BG,highlightthickness=0); self.voice_canvas.pack(pady=18); self._animate_voice_panel(0)
        self.voice_transcript=ctk.CTkLabel(self.voice_panel,text="",font=("Segoe UI",11),text_color=MUTED,wraplength=340,justify="center"); self.voice_transcript.pack(pady=(0,4))
        self.voice_status=ctk.CTkLabel(self.voice_panel,text="Ascolto…",font=("Segoe UI",15,"bold"),text_color=ACCENT); self.voice_status.pack(pady=5)
        ctk.CTkButton(self.voice_panel,text="Chiudi",width=120,height=38,corner_radius=18,fg_color=CARD,hover_color=HOVER,text_color=TEXT,command=self.voice_input).pack(pady=16)
    def _close_voice_panel(self):
        if self.voice_panel and self.voice_panel.winfo_exists(): self.voice_panel.destroy()
        self.voice_panel=None
    def _animate_voice_panel(self, phase):
        if not self.voice_panel or not self.voice_panel.winfo_exists(): return
        self.voice_canvas.delete("all"); c=130
        palette = (("#241437","#4B247F","#7B3FB8","#C06CFF","#F0C7FF") if self.voice_speaking else ("#111B3B","#1A2C66","#304DAB","#6C8CFF","#C6D1FF"))
        for r,color in zip((105,88,70,52,35), palette):
            wobble=((phase%10)-5)*1.8 if r<90 else 0
            self.voice_canvas.create_oval(c-r-wobble,c-r-wobble,c+r+wobble,c+r+wobble,fill=color,outline="")
        self.voice_panel.after(90,lambda:self._animate_voice_panel((phase+1)%10))

    def _listen(self):
        if not self.voice_enabled or self.voice_speaking: return
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
            import queue as audio_queue, time as time_module
            # Il modello italiano distribuito con Lumina vive nella cartella
            # voices/vosk-it; il vecchio percorso artifacts/vosk_italiano faceva
            # fallire il ciclo live prima ancora di aprire il microfono.
            model_path = ROOT_DIR / "voices" / "vosk-it"
            if not model_path.exists(): raise FileNotFoundError("modello italiano per il microfono non trovato")
            rec = KaldiRecognizer(Model(str(model_path)), 16000)
            # Migliora il risultato finale: Vosk restituisce anche i confini
            # delle parole e non chiude la frase mentre l'utente sta ancora
            # articolando l'ultima parola.
            try:
                rec.SetWords(True)
                rec.SetPartialWords(True)
                rec.SetMaxAlternatives(3)
            except AttributeError:
                pass
            self.after(0, lambda: self.voice_status.configure(text="Registrazione in corso…", text_color=ACCENT) if self.voice_panel and self.voice_panel.winfo_exists() else None)
            # Acquisizione a blocchi: non tronca la frase a 6 secondi e
            # termina automaticamente dopo un breve silenzio.
            audio = audio_queue.Queue()
            started = False
            last_voice = time_module.monotonic()
            def callback(indata, frames, _time, status):
                # Pre-enfasi leggera per il microfono integrato: riduce il
                # ronzio costante e rende le consonanti più leggibili per Vosk.
                try:
                    import numpy as np
                    samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
                    samples -= samples.mean()
                    rms = float(np.sqrt(np.mean(samples * samples)) + 1.0)
                    gain = min(2.0, max(1.0, 900.0 / rms))
                    samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
                    audio.put(samples.tobytes())
                except Exception:
                    audio.put(bytes(indata))
            with sd.RawInputStream(samplerate=16000, blocksize=1600, dtype="int16", channels=1, callback=callback):
                deadline = time_module.monotonic() + 12.0
                while time_module.monotonic() < deadline:
                    try: chunk = audio.get(timeout=0.25)
                    except audio_queue.Empty: continue
                    rec.AcceptWaveform(chunk)
                    partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if partial:
                        started = True; last_voice = time_module.monotonic()
                        self.after(0, lambda p=partial: self.voice_transcript.configure(text="Tu: "+p) if self.voice_panel and self.voice_panel.winfo_exists() else None)
                    elif started and time_module.monotonic() - last_voice > 0.9:
                        break
            final_data = json.loads(rec.FinalResult())
            alternatives = final_data.get("alternatives") or []
            if alternatives:
                # Prefer the hypothesis with the highest acoustic confidence.
                best = max(alternatives, key=lambda item: float(item.get("confidence", 0.0)))
                result = str(best.get("text", "")).strip()
            else:
                result = str(final_data.get("text", "")).strip()
            if not result:
                def no_voice():
                    if self.voice_panel and self.voice_panel.winfo_exists():
                        self.voice_status.configure(text="Non ho sentito nulla", text_color=ERR)
                        self.voice_transcript.configure(text="Parla vicino al microfono e riprova.")
                        self.after(900, lambda: self.voice_status.configure(text="Ascolto…", text_color=ACCENT) if self.voice_panel and self.voice_panel.winfo_exists() else None)
                self.after(0, no_voice)
                if self.voice_enabled:
                    threading.Timer(0.45, self._listen).start()
                return
            def deliver_voice():
                if self.voice_panel and self.voice_panel.winfo_exists(): self.voice_status.configure(text="Lumina sta rispondendo…", text_color=ACCENT)
                if self.voice_panel and self.voice_panel.winfo_exists(): self.voice_transcript.configure(text="Tu: "+result)
                self.entry.delete("1.0","end"); self.entry.insert("1.0",result); self.entry.configure(text_color=TEXT); self.set_status("Messaggio vocale acquisito", OK); self.send()
            self.after(0, deliver_voice)
        except Exception as exc:
            def retry():
                if self.voice_enabled and self.voice_panel and self.voice_panel.winfo_exists():
                    self.set_status("Errore microfono: "+str(exc), ERR)
                    self.voice_status.configure(text="Microfono non disponibile", text_color=ERR)
                    self.voice_transcript.configure(text="Controlla il microfono e premi Voce live per riprovare.")
            self.after(0, retry)
            # Un errore transitorio del dispositivo non deve lasciare la voce
            # in uno stato morto: riprova solo se l'utente ha ancora la voce
            # attiva e dopo aver lasciato tempo al driver di liberarsi.
            if self.voice_enabled:
                threading.Timer(1.5, self._listen).start()

    def _speak(self, text):
        try:
            import sys
            sys.path.insert(0, str(ROOT_DIR))
            self.voice_speaking = True
            self.after(0, lambda: self.voice_status.configure(text="Lumina sta parlando…", text_color="#C06CFF") if self.voice_panel and self.voice_panel.winfo_exists() else None)
            self.after(0, lambda: self.voice_transcript.configure(text="Lumina: "+text[:160]) if self.voice_panel and self.voice_panel.winfo_exists() else None)
            from lumina_voice_neural import speak
            speak(text, name="lumina_desktop_reply", do_play=True)
            self.voice_speaking = False
            if self.voice_enabled and self.voice_panel and self.voice_panel.winfo_exists():
                self.after(0, lambda: self.voice_status.configure(text="Parla quando vuoi…", text_color=OK))
                # Lascia spegnere l'audio degli altoparlanti prima di
                # riaprire il microfono, evitando che Lumina si riascolti.
                threading.Timer(0.35, self._listen).start()
        except Exception as exc:
            self.voice_speaking = False
            self.after(0, lambda: (self.set_status("Voce non disponibile: "+str(exc), ERR), self.voice_status.configure(text="Errore voce", text_color=ERR) if self.voice_panel and self.voice_panel.winfo_exists() else None))

if __name__ == "__main__":
    # La build corrente non deve essere bloccata da un vecchio prototipo rimasto
    # aperto: durante lo sviluppo è preferibile aprire la finestra nuova.
    _mutex = None
    LuminaApp().mainloop()
