"""
Lumina Telegram Bot - Assistente AI Sovrano & Vocale su Telegram.
Permette di chattare con Lumina dal telefono (anche con note vocali),
protetto in tempo reale da Lumina Sentinel.
"""

import sys
import os
import time
import requests
import pyttsx3
from lumina_guardrails import LuminaSentinel

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONFIG_FILE = "telegram_token.txt"
API_URL = "http://localhost:8000/v1/chat/completions"


class LuminaTelegramBot:
    def __init__(self, token: str):
        self.token = token.strip()
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.sentinel = LuminaSentinel()
        self.tts_engine = None
        self._init_tts()

    def _init_tts(self):
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty("voices")
            for v in voices:
                if "it-it" in v.id.lower() or "elsa" in v.name.lower():
                    self.tts_engine.setProperty("voice", v.id)
                    break
            self.tts_engine.setProperty("rate", 175)
        except Exception:
            self.tts_engine = None

    def verify_bot(self):
        try:
            res = requests.get(f"{self.base_url}/getMe", timeout=10)
            data = res.json()
            if data.get("ok"):
                bot_user = data["result"]["username"]
                print(f"[+] Bot Telegram verificato con successo! Username: @{bot_user}")
                return True
            else:
                print(f"[!] Errore token Telegram: {data.get('description')}")
                return False
        except Exception as e:
            print(f"[!] Errore connessione all'API di Telegram: {e}")
            return False

    def send_message(self, chat_id, text):
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            res = requests.post(url, json=payload, timeout=10)
            if not res.ok:
                # Se markdown fallisce, invia come testo semplice
                payload.pop("parse_mode", None)
                requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"[!] Errore invio messaggio a chat {chat_id}: {e}")

    def send_voice(self, chat_id, text):
        """Genera audio con TTS e lo invia come nota vocale."""
        if not self.tts_engine:
            return
        audio_path = f"temp_voice_{chat_id}.wav"
        try:
            clean = text.replace("*", "").replace("#", "").replace("`", "").strip()
            self.tts_engine.save_to_file(clean, audio_path)
            self.tts_engine.runAndWait()
            # Aspetta che il file sia scritto
            time.sleep(0.5)

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 100:
                with open(audio_path, "rb") as audio_file:
                    requests.post(
                        f"{self.base_url}/sendAudio",
                        data={"chat_id": chat_id, "title": "Risposta Lumina", "performer": "Lumina AI"},
                        files={"audio": audio_file},
                        timeout=15,
                    )
        except Exception as e:
            print(f"[!] Errore sintesi/invio vocale: {e}")
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

    def ask_lumina(self, prompt: str) -> str:
        # Controllo Lumina Sentinel
        check = self.sentinel.evaluate(prompt)
        if not check.is_safe:
            return (
                f"🛡️ *Intervento di sicurezza Lumina Sentinel*\n"
                f"Richiesta bloccata: {check.refusal_response or check.category}."
            )

        sanitized = check.sanitized_prompt
        payload = {
            "model": "lumina-2.75m",
            "messages": [{"role": "user", "content": sanitized}],
            "max_tokens": 120,
            "temperature": 0.3,
        }
        try:
            res = requests.post(API_URL, json=payload, timeout=10)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            return f"[Errore Server Lumina: HTTP {res.status_code}]"
        except Exception:
            return "Server Lumina locale non raggiungibile. Verifica che `serve.py` sia attivo."

    def run(self):
        print("=" * 60)
        print("  🤖 LUMINA TELEGRAM BOT ATTIVO")
        print("=" * 60)
        print("In ascolto dei messaggi da Telegram in tempo reale...")
        print("Premi Ctrl+C per fermare il bot.\n")

        offset = None
        while True:
            try:
                params = {"timeout": 30}
                if offset:
                    params["offset"] = offset

                res = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=35)
                if res.status_code == 200:
                    data = res.json()
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        msg = update.get("message")
                        if not msg:
                            continue

                        chat_id = msg["chat"]["id"]
                        user_name = msg.get("from", {}).get("first_name", "Utente")
                        text = msg.get("text")

                        if text:
                            print(f"📩 [{user_name}]: {text}")
                            if text == "/start":
                                reply = (
                                    f"Ciao {user_name}! 👋 Sono *Lumina*, il tuo assistente AI sovrano locale.\n\n"
                                    f"• Sono ospitato sul tuo PC personale.\n"
                                    f"• Le tue conversazioni sono protette da *Lumina Sentinel*.\n"
                                    f"• Puoi farmi domande su codice, testi, burocrazia o cultura generale!"
                                )
                                self.send_message(chat_id, reply)
                                continue

                            # Risposta con Lumina
                            reply = self.ask_lumina(text)
                            print(f"🤖 [Lumina -> {user_name}]: {reply}")
                            self.send_message(chat_id, reply)

                            # Se il messaggio dell'utente chiede voce o audio, invia anche vocale
                            if any(k in text.lower() for k in ["parla", "voce", "audio", "dimmelo", "ascolta"]):
                                self.send_voice(chat_id, reply)

                time.sleep(0.5)
            except requests.exceptions.RequestException:
                time.sleep(3)
            except KeyboardInterrupt:
                print("\n[+] Arresto bot Telegram completato.")
                break
            except Exception as e:
                print(f"[!] Errore loop bot: {e}")
                time.sleep(2)


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token and os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()

    if not token:
        print("=" * 60)
        print("  🔑 CONFIGURAZIONE BOT TELEGRAM LUMINA")
        print("=" * 60)
        print("Per usare Lumina da Telegram:")
        print("1. Apri Telegram e cerca @BotFather")
        print("2. Scrivi /newbot e segui le istruzioni per dare un nome al tuo bot")
        print("3. Incolla qui sotto il TOKEN che BotFather ti fornisce:")
        print("------------------------------------------------------------")
        token = input("Inserisci Token Telegram: ").strip()
        if token:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(token)
            print(f"[+] Token salvato in '{CONFIG_FILE}' per i futuri avvii automatici.")
        else:
            print("[!] Token non inserito. Uscita.")
            sys.exit(1)

    bot = LuminaTelegramBot(token)
    if bot.verify_bot():
        bot.run()


if __name__ == "__main__":
    main()
