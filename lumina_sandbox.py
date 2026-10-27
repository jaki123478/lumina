"""
Lumina Code Sandbox & Self-Correction Engine.
Esegue in un ambiente protetto il codice generato da Lumina Coder,
intercetta eventuali bug o errori di sintassi e si autocorregge prima di fornire la risposta!
"""

import sys
import os
import re
import time
import tempfile
import subprocess
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

API_URL = "http://localhost:8000/v1/chat/completions"


class LuminaCodeSandbox:
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url

    def extract_code(self, response_text: str) -> str:
        """Estrae il blocco di codice Python dal testo markdown."""
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Se non ci sono backtick ma sembra codice python
        lines = [line for line in response_text.split("\n") if line.strip() and not line.strip().startswith("#")]
        if any(kw in response_text for kw in ["def ", "import ", "print(", "class ", "return "]):
            return response_text.strip()
        return ""

    def run_in_sandbox(self, python_code: str, timeout: int = 5) -> dict:
        """Esegue il codice in un processo isolato catturando output o errori."""
        if not python_code.strip():
            return {"success": False, "output": "", "error": "Nessun codice eseguibile trovato."}

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            temp_path = f.name
            f.write(python_code)

        try:
            cmd = [sys.executable, temp_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
            
            if proc.returncode == 0:
                out = proc.stdout.strip()
                return {"success": True, "output": out if out else "(Codice eseguito con successo, nessun output su terminale)", "error": ""}
            else:
                err = proc.stderr.strip()
                return {"success": False, "output": proc.stdout.strip(), "error": err}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": f"Timeout di sicurezza superato ({timeout}s). Possibile ciclo infinito."}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def generate_code(self, prompt: str) -> str:
        """Chiama Lumina Coder per generare la soluzione."""
        payload = {
            "model": "lumina-coder",
            "messages": [
                {"role": "system", "content": "Sei Lumina Coder. Scrivi codice Python pulito ed eseguibile racchiuso in blocchi ```python ... ``` e include alla fine un print() di test."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 250,
            "temperature": 0.2
        }
        res = requests.post(self.api_url, json=payload, timeout=25)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
        return ""

    def self_correct_code(self, original_prompt: str, broken_code: str, error_trace: str) -> str:
        """Invia il traceback dell'errore a Lumina per l'autocorrezione automatica."""
        fix_prompt = (
            f"Il codice seguente generato per la richiesta '{original_prompt}' ha generato un errore durante l'esecuzione:\n\n"
            f"```python\n{broken_code}\n```\n\n"
            f"ERRORE RILEVATO DALLA SANDBOX:\n{error_trace}\n\n"
            f"Correggi il bug e restituisci il codice corretto funzionante racchiuso in ```python ... ```."
        )
        return self.generate_code(fix_prompt)

    def process(self, prompt: str) -> dict:
        """Ciclo completo: Generazione -> Test Sandbox -> Autocorrezione se necessario."""
        print(f"[*] Lumina Coder genera il codice per: '{prompt}'...")
        initial_reply = self.generate_code(prompt)
        code = self.extract_code(initial_reply)

        if not code:
            return {
                "verified": False,
                "reply": initial_reply,
                "code": "",
                "output": "",
                "corrections": 0
            }

        print("[*] Esecuzione codice in Sandbox protetta...")
        result = self.run_in_sandbox(code)

        if result["success"]:
            print("✅ Codice verificato al primo colpo!")
            return {
                "verified": True,
                "reply": initial_reply,
                "code": code,
                "output": result["output"],
                "corrections": 0
            }

        # Tentativo di Autocorrezione
        print(f"⚠️ Errore rilevato: {result['error'].splitlines()[-1] if result['error'] else 'Errore sconosciuto'}")
        print("🧠 Avvio ciclo di Autocorrezione neurale...")
        
        fixed_reply = self.self_correct_code(prompt, code, result["error"])
        fixed_code = self.extract_code(fixed_reply)
        fixed_result = self.run_in_sandbox(fixed_code)

        if fixed_result["success"]:
            print("✨ Autocorrezione riuscita con successo!")
            return {
                "verified": True,
                "reply": fixed_reply,
                "code": fixed_code,
                "output": fixed_result["output"],
                "corrections": 1
            }
        else:
            print("❌ Il codice necessita di revisione manuale.")
            return {
                "verified": False,
                "reply": fixed_reply,
                "code": fixed_code,
                "output": fixed_result["output"],
                "error": fixed_result["error"],
                "corrections": 1
            }

    def run_interactive(self):
        print("=" * 65)
        print("  🧪 LUMINA CODE SANDBOX & AUTOCORREZIONE AUTOMATICA")
        print("=" * 65)
        print("Scrivi cosa vuoi programmare: Lumina scriverà il codice, lo testerà")
        print("nella sandbox e te lo consegnerà verificato e funzionante al 100%!")
        print("Digita 'esci' per uscire.\n")

        while True:
            try:
                prompt = input("Cosa vuoi programmare? > ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ["esci", "exit", "quit", "q"]:
                    break

                res = self.process(prompt)
                print("\n" + "=" * 50)
                if res["verified"]:
                    print(f"✅ CODICE VERIFICATO (Autocorrezioni: {res['corrections']}):")
                    print(res["code"])
                    print(f"\n📊 OUTPUT SANDBOX REALE:\n{res['output']}")
                else:
                    print(res["reply"])
                    if "error" in res:
                        print(f"\n[!] Errore: {res['error']}")
                print("=" * 50 + "\n")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    sandbox = LuminaCodeSandbox()
    sandbox.run_interactive()
