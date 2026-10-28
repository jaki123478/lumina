import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json, urllib.request, threading, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
class LuminaChat(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('Lumina'); self.geometry('1180x760'); self.configure(bg='#10131c'); self.history=[]; self._style(); self._ui()
    def _style(self):
        s=ttk.Style(self); s.theme_use('clam'); s.configure('TFrame',background='#10131c'); s.configure('TLabel',background='#10131c',foreground='#dfe5ff'); s.configure('TButton',background='#252c42',foreground='#eef2ff',padding=8); s.configure('TEntry',fieldbackground='#1b2131',foreground='#eef2ff')
    def _ui(self):
        self.grid_columnconfigure(1,weight=1); self.grid_rowconfigure(0,weight=1)
        side=ttk.Frame(self,width=220); side.grid(row=0,column=0,sticky='nsew'); ttk.Label(side,text='LUMINA',font=('Segoe UI',18,'bold')).pack(anchor='w',padx=20,pady=(24,4)); ttk.Label(side,text='Assistente locale',foreground='#8c98bd').pack(anchor='w',padx=20,pady=(0,24)); ttk.Button(side,text='+ Nuova chat',command=self.clear).pack(fill='x',padx=14); ttk.Label(side,text='SESSIONI',foreground='#7e8ab0').pack(anchor='w',padx=20,pady=(30,8)); self.sessions=tk.Listbox(side,bg='#151a29',fg='#dfe5ff',bd=0,highlightthickness=0); self.sessions.pack(fill='both',expand=True,padx=14,pady=4)
        center=ttk.Frame(self); center.grid(row=0,column=1,sticky='nsew'); center.grid_columnconfigure(0,weight=1); center.grid_rowconfigure(1,weight=1); top=ttk.Frame(center); top.grid(row=0,column=0,sticky='ew',padx=30,pady=18); ttk.Label(top,text='Chat',font=('Segoe UI',16,'bold')).pack(side='left'); self.mode=ttk.Combobox(top,values=['Generale','Coding','Ricerca web'],state='readonly',width=16); self.mode.current(0); self.mode.pack(side='right'); self.messages=tk.Text(center,wrap='word',state='disabled',bg='#10131c',fg='#e9edff',font=('Segoe UI',12),bd=0,padx=20,pady=20); self.messages.grid(row=1,column=0,sticky='nsew',padx=18); self.orb=tk.Canvas(center,width=70,height=70,bg='#10131c',highlightthickness=0); self.orb.grid(row=2,column=0,pady=(4,0)); self._orb_phase=0; bottom=ttk.Frame(center); bottom.grid(row=3,column=0,sticky='ew',padx=30,pady=12); bottom.grid_columnconfigure(0,weight=1); self.input=ttk.Entry(bottom); self.input.grid(row=0,column=0,sticky='ew'); self.input.bind('<Return>',lambda e:self.send()); ttk.Button(bottom,text='🎙 Voce',command=self.voice).grid(row=0,column=1,padx=6); ttk.Button(bottom,text='Invia',command=self.send).grid(row=0,column=2); self.status=ttk.Label(center,text='Lumina locale pronta',foreground='#72d49b'); self.status.grid(row=4,column=0,sticky='w',padx=30,pady=(0,12))
    def clear(self): self.messages.configure(state='normal'); self.messages.delete('1.0','end'); self.messages.configure(state='disabled'); self.history=[]
    def add(self,who,text): self.messages.configure(state='normal'); self.messages.insert('end',f'{who}\n{text}\n\n'); self.messages.configure(state='disabled'); self.messages.see('end')
    def send(self):
        q=self.input.get().strip()
        if not q:return
        self.input.delete(0,'end'); self.add('Tu',q); self.status.configure(text='Lumina sta elaborando…')
        def run():
            try:
                mode=self.mode.get(); model='lumina-coder' if mode=='Coding' else 'lumina-assisted'; body=json.dumps({'model':model,'messages':[{'role':'user','content':q}],'max_tokens':400,'web':mode=='Ricerca web','research':mode=='Ricerca web'}).encode(); req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=body,headers={'Content-Type':'application/json'}); d=json.load(urllib.request.urlopen(req,timeout=90)); a=d['choices'][0]['message']['content']
            except Exception as e:a=f'Non riesco a collegarmi a Lumina: {e}'
            self.after(0,lambda:(self.add('Lumina',a),self.status.configure(text='Lumina locale pronta')))
        threading.Thread(target=run,daemon=True).start()
    def voice(self): self.add('Voce','Lumina sta parlando…'); self.status.configure(text='Voce attiva'); self._animate_orb(0)
    def _animate_orb(self,n):
        if n>45: self.orb.delete('all'); self.status.configure(text='Lumina locale pronta'); return
        self.orb.delete('all'); r=13+(n%8)*2; self.orb.create_oval(35-r,35-r,35+r,35+r,fill='#6887ff',outline='#b4c2ff',width=2); self.after(90,lambda:self._animate_orb(n+1))
if __name__=='__main__': LuminaChat().mainloop()
