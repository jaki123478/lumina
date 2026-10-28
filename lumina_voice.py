"""
Assistente Vocale per Lumina AI.
Ascolta l'utente dal microfono in italiano, filtra con Lumina Sentinel,
interroga Lumina LLM e risponde con Piper Paola in locale.
"""

import sys
import os
import time
import requests
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import sounddevice as sd
except ImportError:
    sd = None

from lumina_guardrails import LuminaSentinel


class LuminaVoiceAssistant:
    def __init__(self, api_url="http://localhost:8000/v1/chat/completions", model="lumina-assisted"):
        self.api_url = api_url
        self.model = model
        self.history = []
        self.sentinel = LuminaSentinel()
        self.tts_engine = None
        self._init_tts()

    def _init_tts(self):
        try:
            from lumina_voice_neural import load_voice
            self.tts_engine = load_voice()
        except Exception as e:
            print(f"[!] Errore inizializzazione motore vocale: {e}")
            self.tts_engine = None

    def speak(self, text: str):
        """Pronuncia la frase tramite il motore TTS locale."""
        if not text:
            return
        from lumina_voice_neural import clean, speak
        clean_text = clean(text)
        print(f"\n🔊 [Lumina Voce]: {clean_text}")
        if self.tts_engine:
            try:
                speak(clean_text)
            except Exception as e:
                print(f"[!] Errore riproduzione vocale: {e}")

    def record_microphone(self, duration=5, sample_rate=16000):
        """Registra audio dal microfono usando sounddevice."""
        if not sd:
            print("[!] sounddevice non disponibile.")
            return None

        print(f"\n🎙️  Ascolto in corso per {duration} secondi... (parla adesso)")
        try:
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            # Indicatore visivo durante la registrazione
            for _ in range(duration * 2):
                time.sleep(0.5)
                print(".", end="", flush=True)
            sd.wait()
            print(" [Fatto]")
            return recording, sample_rate
        except Exception as e:
            print(f"\n[!] Errore durante la registrazione audio: {e}")
            return None

    def transcribe(self, recording, sample_rate):
        """Riconoscimento Vosk solo da modello locale, senza fallback di rete."""
        if recording is None:
            return None
        try:
            from pathlib import Path
            import json
            from vosk import Model, KaldiRecognizer
            folder = Path(__file__).resolve().parent / 'voices' / 'vosk-it'
            if not folder.is_dir():
                raise FileNotFoundError('Modello Vosk italiano locale mancante')
            if not hasattr(self, '_stt_model'):
                self._stt_model = Model(str(folder))
            recognizer = KaldiRecognizer(self._stt_model, sample_rate)
            recognizer.AcceptWaveform(recording.astype('<i2').tobytes())
            return json.loads(recognizer.FinalResult()).get('text', '').strip() or None
        except (ImportError, FileNotFoundError):
            print('[!] Trascrizione offline non configurata: usa t per digitare. Nessun audio inviato in rete.')
            return None
        except Exception as e:
            print(f"[!] Errore trascrizione: {e}")
            return None

    def ask_lumina(self, user_text: str) -> str:
        """Invia il prompt a Lumina (protetto da Lumina Sentinel)."""
        # 1. Filtro di sicurezza Sentinel
        check = self.sentinel.evaluate(user_text)
        if not check.is_safe:
            block_msg = check.refusal_response or f"Richiesta bloccata da Lumina Sentinel ({check.category})."
            return block_msg

        prompt = check.sanitized_prompt

        # 2. Chiamata API locale
        payload = {
            "model": self.model,
            "messages": self.history + [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0.3
        }

        try:
            res = requests.post(self.api_url, json=payload, timeout=90)
            if res.status_code == 200:
                data = res.json()
                reply = data["choices"][0]["message"]["content"]
                self.history = (payload['messages'] + [{'role':'assistant','content':reply}])[-12:]
                return reply.strip()
            else:
                return f"Errore server Lumina (HTTP {res.status_code})."
        except Exception as e:
            return f"Impossibile collegarsi al server Lumina locale su {self.api_url}."

    def run_interactive(self):
        print("=" * 65)
        print("       🎙️  LUMINA VOICE ASSISTANT - ASSISTENTE VOCALE ITALIANO")
        print("=" * 65)
        print("• Riconoscimento vocale: Vosk italiano, se installato in locale")
        print("• Sintesi vocale (TTS): Piper Paola, locale")
        print("• Protezione: Lumina Sentinel (ML + Heuristic + Anti-PII)")
        print("• Server: " + self.api_url)
        print("=" * 65)

        self.speak("Ciao! Sono Lumina, il tuo assistente vocale. Come posso aiutarti?")

        while True:
            print("\n-------------------------------------------------------------")
            print("Premi [INVIO] per parlare al microfono, scrivi 't' per digitare, o 'esci' per uscire:")
            choice = input("Comando > ").strip().lower()

            if choice in ["esci", "exit", "quit", "q"]:
                self.speak("Arrivederci! Buona giornata.")
                break

            user_query = None

            if choice == "t":
                user_query = input("Tu (scritto) > ").strip()
            else:
                # Modalità vocale con registrazione microfono
                rec_result = self.record_microphone(duration=5)
                if rec_result:
                    recording, sample_rate = rec_result
                    print("Trascrizione vocale in corso...")
                    user_query = self.transcribe(recording, sample_rate)

            if not user_query:
                continue

            print(f"\n👤 Tu: \"{user_query}\"")
            print("🧠 Lumina elabora...")
            answer = self.ask_lumina(user_query)
            self.speak(answer)


if __name__ == "__main__":
    assistant = LuminaVoiceAssistant()
    assistant.run_interactive()
