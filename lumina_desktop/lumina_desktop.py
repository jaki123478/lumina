"""IDE desktop indipendente, realizzato con la libreria standard di Python."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess, threading
import json, urllib.request

class Workspace(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Lumina Desktop"); self.geometry("1280x780"); self.minsize(980,600)
        self.project = Path.cwd(); self._build_menu(); self._build_layout(); self.refresh_files()

    def _build_menu(self):
        bar=tk.Menu(self); filemenu=tk.Menu(bar,tearoff=False); filemenu.add_command(label="Apri cartella",command=self.choose_project); filemenu.add_command(label="Salva",command=self.save_file); filemenu.add_separator(); filemenu.add_command(label="Esci",command=self.destroy); bar.add_cascade(label="File",menu=filemenu); self.config(menu=bar)

    def _build_layout(self):
        self.grid_columnconfigure(1,weight=1); self.grid_rowconfigure(0,weight=1)
        left=ttk.Frame(self); left.grid(row=0,column=0,sticky="nsew"); ttk.Label(left,text="PROGETTO").pack(anchor="w",padx=10,pady=8); self.tree=ttk.Treeview(left,show="tree"); self.tree.pack(fill="both",expand=True,padx=6,pady=4); self.tree.bind("<<TreeviewSelect>>",self.open_selected)
        center=ttk.Frame(self); center.grid(row=0,column=1,sticky="nsew"); center.grid_rowconfigure(1,weight=1); center.grid_rowconfigure(2,weight=0); center.grid_columnconfigure(0,weight=1)
        self.tabs=ttk.Label(center,text="Nessun file aperto"); self.tabs.grid(row=0,column=0,sticky="ew",padx=8,pady=8); self.editor=tk.Text(center,undo=True,wrap="none",font=("Cascadia Code",11),background="#101522",foreground="#e6ebff",insertbackground="white"); self.editor.grid(row=1,column=0,sticky="nsew",padx=8); self.editor.bind("<Control-s>",lambda e:self.save_file())
        ttk.Label(center,text="TERMINALE").grid(row=2,column=0,sticky="w",padx=8,pady=(8,0)); self.terminal=tk.Text(center,height=8,background="#0b0f18",foreground="#9ee6b3",font=("Cascadia Mono",10)); self.terminal.grid(row=3,column=0,sticky="nsew",padx=8); self.terminal.insert("end","Lumina Desktop pronto.\n"); tr=ttk.Frame(center); tr.grid(row=4,column=0,sticky="ew",padx=8,pady=(0,8)); self.cmd=ttk.Entry(tr); self.cmd.pack(side="left",fill="x",expand=True); self.cmd.bind("<Return>",lambda e:self.run_command()); ttk.Button(tr,text="Esegui",command=self.run_command).pack(side="right",padx=4)
        right=ttk.Frame(self,width=310); right.grid(row=0,column=2,sticky="nsew"); hdr=ttk.Frame(right); hdr.pack(fill="x",padx=8); ttk.Label(hdr,text="ASSISTENTE AI").pack(side="left",pady=8); self.provider=ttk.Combobox(hdr,values=["Lumina locale","OpenAI-compatible"],state="readonly",width=18); self.provider.current(0); self.provider.pack(side="right"); self.chat=tk.Text(right,width=36,wrap="word",state="disabled",background="#171d30",foreground="#e6ebff"); self.chat.pack(fill="both",expand=True,padx=8); row=ttk.Frame(right); row.pack(fill="x",padx=8,pady=8); self.prompt=ttk.Entry(row); self.prompt.pack(side="left",fill="x",expand=True); self.prompt.bind("<Return>",lambda e:self.ask()); ttk.Button(row,text="Invia",command=self.ask).pack(side="right",padx=(6,0))

    def refresh_files(self):
        self.tree.delete(*self.tree.get_children()); root=self.tree.insert("","end",text=self.project.name,open=True,iid=str(self.project))
        for p in sorted(self.project.iterdir(),key=lambda x:(x.is_file(),x.name.lower())):
            if p.name.startswith("."): continue
            self.tree.insert(root,"end",text=p.name,iid=str(p))

    def choose_project(self):
        p=filedialog.askdirectory(initialdir=self.project)
        if p: self.project=Path(p); self.refresh_files()

    def open_selected(self,_=None):
        sel=self.tree.selection()
        if not sel: return
        p=Path(sel[0])
        if p.is_file() and p.stat().st_size<2_000_000:
            try:self.editor.delete("1.0","end"); self.editor.insert("1.0",p.read_text(encoding="utf-8")); self.tabs.config(text=p.name); self.current=p
            except UnicodeDecodeError: messagebox.showwarning("File binario","Questo file non è testo.")

    def save_file(self):
        p=getattr(self,"current",None)
        if p: p.write_text(self.editor.get("1.0","end-1c"),encoding="utf-8"); self.terminal.insert("end",f"Salvato: {p.name}\n")

    def run_command(self):
        cmd=self.cmd.get().strip(); self.cmd.delete(0,"end")
        if not cmd:return
        self.terminal.insert("end",f"> {cmd}\n")
        try: out=subprocess.run(cmd,shell=True,cwd=self.project,capture_output=True,text=True,timeout=30); self.terminal.insert("end",(out.stdout+out.stderr)[:6000]+"\n")
        except Exception as e:self.terminal.insert("end",f"Errore: {e}\n")

    def ask(self):
        q=self.prompt.get().strip()
        if not q:return
        self.prompt.delete(0,"end"); self.chat.configure(state="normal"); self.chat.insert("end",f"Tu: {q}\nAssistente: "); self.chat.configure(state="disabled"); self.chat.see("end")
        def call():
            try:
                body=json.dumps({"model":"lumina-assisted","messages":[{"role":"user","content":q}],"max_tokens":200}).encode(); req=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}); data=json.load(urllib.request.urlopen(req,timeout=60)); ans=data["choices"][0]["message"]["content"]
            except Exception as e: ans=f"Provider non disponibile: {e}"
            self.after(0,lambda:self._append_chat(ans))
        threading.Thread(target=call,daemon=True).start()

    def _append_chat(self,text):
        self.chat.configure(state="normal"); self.chat.insert("end",text+"\n\n"); self.chat.configure(state="disabled"); self.chat.see("end")

if __name__ == "__main__": Workspace().mainloop()
