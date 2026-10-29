"""Monitor leggero: Gemma domanda, Lumina risponde, Gemma corregge."""
from pathlib import Path
import json
import requests
import subprocess, sys, time
from llama_cpp import Llama

MODEL = Path('models/gemma-3-4b-it-clean/gemma-3-4b-it-Q4_K_M.gguf')
QUESTIONS = (['Qual è la capitale del Portogallo?', 'Scrivi una funzione Python somma(a, b) corretta.'] * 10)

def main():
    gemma = Llama(model_path=str(MODEL), n_ctx=512, n_gpu_layers=2, verbose=False)
    server_proc = None
    def ensure_server():
        nonlocal server_proc
        try:
            requests.get('http://127.0.0.1:8000/v1/models', timeout=2)
            return
        except Exception:
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            server_proc = subprocess.Popen([sys.executable, 'serve.py', '--port', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, creationflags=flags)
            for _ in range(20):
                try:
                    requests.get('http://127.0.0.1:8000/v1/models', timeout=1)
                    return
                except Exception: time.sleep(0.25)
    consecutive_failures = 0
    for q in QUESTIONS:
        ensure_server()
        lumina = '[server non disponibile]'
        for attempt in range(3):
            try:
                r = requests.post(
                    'http://127.0.0.1:8000/v1/chat/completions',
                    json={'model':'lumina','messages':[{'role':'user','content':q}],
                          'max_tokens':100,'temperature':0.1},
                    timeout=30,
                )
                r.raise_for_status()
                lumina = r.json()['choices'][0]['message']['content'].strip()
                consecutive_failures = 0
                break
            except Exception:
                consecutive_failures += 1
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        if lumina == '[server non disponibile]' and consecutive_failures >= 3:
            print('[STOP] Server non disponibile dopo 3 tentativi: ciclo interrotto in sicurezza.', flush=True)
            break
        judge_prompt = f'Valuta questa risposta di Lumina alla domanda "{q}". Indica CORRETTA o ERRATA e, se errata, dai la correzione esatta. Risposta Lumina: {lumina}'
        g = gemma.create_chat_completion(messages=[{'role':'user','content':judge_prompt}], max_tokens=120, temperature=0.1)
        correction = g['choices'][0]['message']['content'].strip()
        print(f'\nDOMANDA DI GEMMA: {q}\nRISPOSTA DI LUMINA: {lumina}\nCORREZIONE DI GEMMA: {correction}\nESITO: valutazione completata', flush=True)

if __name__ == '__main__':
    main()
