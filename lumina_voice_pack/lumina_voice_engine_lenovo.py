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
PROJECT_DIR = Path(r"C:\Users\Jaki1\Desktop\ai lumina")

SYNTH_SCRIPT = SCRIPT_DIR / "synthesize_speech.ps1"
AUDIO_DIR = SCRIPT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def clean_for_speech(text: str) -> str:
    t = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|pad|>", "")
    t = t.replace("<think>", "").replace("</think>", "")
    lines = [line.strip() for line in t.splitlines() if line.strip()]
    cleaned = " ".join(lines)
    if len(cleaned) > 400:
        cleaned = cleaned[:397] + "..."
    return cleaned


def speak_response(text: str, filename_base: str):
    clean_text = clean_for_speech(text)
    if not clean_text:
        clean_text = "Risposta completata."
    wav_file = AUDIO_DIR / f"{filename_base}.wav"
    cmd = [
        "powershell", "-ExecutionPolicy", "Bypass",
        "-File", str(SYNTH_SCRIPT),
        "-Text", clean_text,
        "-OutputFile", str(wav_file),
        "-SpeakLive"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
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

        ckpt_dir = PROJECT_DIR / "checkpoints_lumina_next"
        print(f"[*] Caricamento Lumina-Next Step 8000 su {self.device} ({self.dtype})...")
        self.model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).to(device=self.device, dtype=self.dtype).eval()
        print(f"[+] Modello pronto in memoria GPU! Parametri: {self.model.count_parameters():,}")

    def generate(self, prompt: str, max_new_tokens: int = 80) -> tuple[str, float, int]:
        import torch
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([self.tokenizer.encode(formatted).ids], device=self.device, dtype=torch.long)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.2, top_p=0.9)
        elapsed = time.perf_counter() - t0

        gen_tokens = len(out_ids[0]) - len(input_ids[0])
        speed = gen_tokens / elapsed if elapsed > 0 else 0
        reply = self.tokenizer.decode(out_ids[0][len(input_ids[0]):].tolist()).strip()
        if "<|im_end|>" in reply:
            reply = reply.split("<|im_end|>")[0].strip()
        return reply, speed, gen_tokens


def main():
    print("=" * 70)
    print("      🎙️ LUMINA VOICE INTERACTIVE — PC LENOVO RTX 5060")
    print("      Modello: Step 8000 (100% Benchmark Score) | Voce: Elsa (it-IT)")
    print("      Digita la tua domanda: Lumina risponderà a schermo e a voce!")
    print("      Digita 'esci' per uscire.")
    print("=" * 70 + "\n")

    engine = LuminaPyTorchEngine()
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
