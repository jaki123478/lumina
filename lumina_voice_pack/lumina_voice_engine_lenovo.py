"""
lumina_voice_engine_lenovo.py
Modulo interattivo vocale di Lumina per il PC Principale Lenovo RTX 5060.
Sviluppato e pacchettizzato dal nodo AMD per il master node, integrato con il modello Step 8000.
Supporta:
1. PyTorch Nativo su NVIDIA GeForce RTX 5060 (CUDA bfloat16) — Velocità max ~45-55 tok/s
2. ONNX Runtime (DML / CPU)
3. Sintesi Vocale Offline Windows (Microsoft Elsa it-IT / SAPI5)
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_TEST_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SCRIPT_DIR.parent

SYNTH_SCRIPT = SCRIPT_DIR / "synthesize_speech.ps1"
AUDIO_DIR = SCRIPT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def clean_for_speech(text: str) -> str:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from lumina_voice_neural import clean
    return clean(text)


def speak_response(text: str, filename_base: str):
    clean_text = clean_for_speech(text)
    if not clean_text:
        clean_text = "Risposta completata."
    wav_file = AUDIO_DIR / f"{filename_base}.wav"
    try:
        from lumina_voice_neural import synthesize, play
        synthesize(clean_text, wav_file)
        play(wav_file)
        return str(wav_file)
    except Exception as e:
        print(f"[!] Errore sintesi vocale: {e}")
        return None


class LuminaPyTorchEngine:
    def __init__(self):
        import torch
        from tokenizers import ByteLevelBPETokenizer
        
        # Aggiungi cartella progetto a sys.path per importare modeling_lumina_next
        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))
        from modeling_lumina_next import LuminaNextForCausalLM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        tok_dir = PROJECT_DIR / "tokenizer_lumina_next"
        if not tok_dir.exists():
            tok_dir = BASE_TEST_DIR / "lenovo_releases"

        self.tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

        from lumina_next_runtime import active_model_dir
        ckpt_dir = active_model_dir()
        print(f"[*] Caricamento Lumina-Next Step 8000 su {self.device} ({self.dtype})...")
        self.model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=self.device, dtype=self.dtype).eval()
        self.history = []
        print(f"[+] Modello pronto in memoria GPU! Parametri: {self.model.count_parameters():,}")

    def generate(self, prompt: str, max_new_tokens: int = 80) -> tuple[str, float, int]:
        import torch
        from lumina_next_runtime import prompt_ids, stop_sequences, response_text
        messages = self.history + [{'role':'user','content':prompt}]
        input_ids = torch.tensor([prompt_ids(self.tokenizer, messages)], device=self.device, dtype=torch.long)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0,
                                          eos_token_id=1, stop_token_sequences=stop_sequences(self.tokenizer))
        elapsed = time.perf_counter() - t0

        gen_tokens = len(out_ids[0]) - len(input_ids[0])
        speed = gen_tokens / elapsed if elapsed > 0 else 0
        reply = response_text(self.tokenizer, out_ids[0][len(input_ids[0]):].tolist())
        self.history = (messages + [{'role':'assistant','content':reply}])[-12:]
        return reply, speed, gen_tokens


class LuminaAssistedEngine:
    def __init__(self):
        import requests
        self.session=requests.Session();self.session.trust_env=False
        self.history=[]

    def generate(self,prompt,max_new_tokens=160):
        started=time.perf_counter()
        messages=self.history+[{'role':'user','content':prompt}]
        r=self.session.post('http://127.0.0.1:8000/v1/chat/completions',json={
            'model':'lumina-assisted','messages':messages,'max_tokens':max_new_tokens,'temperature':0},timeout=180)
        r.raise_for_status();data=r.json();reply=data['choices'][0]['message']['content']
        self.history=(messages+[{'role':'assistant','content':reply}])[-12:]
        count=data.get('backend',{}).get('completion_tokens',0)
        return reply,count/max(time.perf_counter()-started,.001),count


def main():
    print("=" * 70)
    print("      🎙️ LUMINA VOICE INTERACTIVE — PC LENOVO RTX 5060")
    print("      Supporto Qwen locale | Voce: Piper Paola (it-IT)")
    print("      Digita la tua domanda: Lumina risponderà a schermo e a voce!")
    print("      Digita 'esci' per uscire.")
    print("=" * 70 + "\n")

    engine = LuminaPyTorchEngine() if '--native' in sys.argv else LuminaAssistedEngine()
    print("\n[+] Console vocale pronta all'uso!\n")

    idx = 1
    while True:
        try:
            prompt = input("Lenovo Utente > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("esci", "quit", "exit"):
                print("Chiusura Lumina Voice Console.")
                break

            print("\n🤖 Lumina sta pensando...")
            resp, speed, tokens = engine.generate(prompt)
            print(f"Risposta: {resp}")
            print(f"⚡ [Prestazioni]: {tokens} token a {speed:.1f} tok/s")

            print("🔊 Riproduzione vocale in corso...")
            speak_response(resp, f"lenovo_voice_interaction_{int(time.time())}_{idx}")
            idx += 1
            print("-" * 70)
        except (KeyboardInterrupt, EOFError):
            print("\nSessione terminata.")
            break


if __name__ == "__main__":
    main()
