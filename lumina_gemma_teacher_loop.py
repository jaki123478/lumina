"""Loop controllato Lumina -> Gemma: raccoglie esempi verificabili senza modificare pesi."""
from pathlib import Path
import argparse
import json, time, requests, os, subprocess, sys
from llama_cpp import Llama

MODEL = Path('models/gemma-3-4b-it-clean/gemma-3-4b-it-Q4_K_M.gguf')
LOG = Path('data/lumina_gemma_live.log')
OUT = Path('data/lumina_gemma_approved.jsonl')
SERVER_URL = os.environ.get('LUMINA_SERVER_URL', 'http://127.0.0.1:8000')

def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f'{SERVER_URL}/v1/models', timeout=2)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cycles', type=int, default=1)
    ap.add_argument('--train-steps', type=int, default=0,
                    help='micro-step opzionali dopo esempi accettati; 0 = sola raccolta')
    args = ap.parse_args()
    print(f'Controllo server locale: {SERVER_URL}', flush=True)
    if not wait_for_server():
        print('[STOP] Server locale non raggiungibile: nessun esempio e nessun peso modificato.', flush=True)
        return 2
    gemma = Llama(model_path=str(MODEL), n_ctx=512, n_gpu_layers=4, verbose=False)
    prompts = ['Scrivi una funzione Python somma(a, b) corretta.', 'Qual è la capitale del Portogallo?']
    LOG.parent.mkdir(exist_ok=True)
    for cycle in range(1, args.cycles + 1):
      print(f'=== CICLO {cycle}/{args.cycles} ===', flush=True)
      for p in prompts:
        try:
            lumina = requests.post(f'{SERVER_URL}/v1/chat/completions', json={'model':'lumina','messages':[{'role':'user','content':p}],'max_tokens':80,'temperature':0.1}, timeout=30).json()['choices'][0]['message']['content']
        except Exception as e:
            lumina = f'[errore locale: {e}]'
        g = gemma.create_chat_completion(messages=[{'role':'user','content':p}], max_tokens=100, temperature=0.1)
        answer = g['choices'][0]['message']['content'].strip()
        if 'somma' in p.lower():
            accepted = 'def somma' in answer and 'return a + b' in answer
        elif 'portogallo' in p.lower():
            accepted = 'lisbona' in answer.lower()
        else:
            accepted = bool(answer)
        accepted = accepted and 'errore locale' not in answer.lower() and 'errore locale' not in lumina.lower()
        event = {'prompt':p,'lumina':lumina,'gemma':answer,'accepted':accepted,'time':time.time()}
        with LOG.open('a',encoding='utf-8') as f: f.write(json.dumps(event,ensure_ascii=False)+'\n')
        if accepted:
            with OUT.open('a',encoding='utf-8') as f: f.write(json.dumps({'user':p,'assistant':answer,'kind':'teacher_verified'},ensure_ascii=False)+'\n')
        print(f'\nDOMANDA DI LUMINA: {p}', flush=True)
        print(f'RISPOSTA DI LUMINA: {lumina}', flush=True)
        print(f'RISPOSTA DI GEMMA: {answer}', flush=True)
        print(f'VERIFICA: {"ACCETTATA" if accepted else "SCARTATA"}', flush=True)
        if accepted and args.train_steps > 0:
            env = os.environ.copy()
            env['LUMINA_EXTRA_DATASET'] = str(OUT)
            env['LUMINA_INIT_CHECKPOINT'] = env.get('LUMINA_INIT_CHECKPOINT', 'checkpoints_lumina_next')
            env['LUMINA_CHECKPOINT_DIR'] = env.get('LUMINA_CHECKPOINT_DIR', 'checkpoints_lumina_teacher_loop')
            env['LUMINA_EXPORT_DIR'] = env.get('LUMINA_EXPORT_DIR', 'exports_lumina_teacher_loop')
            seq_len = os.environ.get('LUMINA_TRAIN_SEQ_LEN', '32')
            cmd = [sys.executable, 'train_lumina_next.py', '--steps', str(args.train_steps),
                   '--seq-len', seq_len, '--batch-size', '1', '--grad-accum', '1', '--lr', '0.000001']
            print(f'TRAINING MICRO-CICLO: {args.train_steps} step su checkpoint separato', flush=True)
            try:
                result = subprocess.run(cmd, env=env, check=False, timeout=120)
            except subprocess.TimeoutExpired:
                print('[STOP] Micro-ciclo oltre 120 secondi: arrestato per proteggere il PC.', flush=True)
                return
            if result.returncode != 0:
                print('[STOP] Micro-ciclo fallito: checkpoint stabile invariato.', flush=True)
                return

if __name__ == '__main__': main()
