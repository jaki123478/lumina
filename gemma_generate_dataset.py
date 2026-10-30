"""Genera un piccolo dataset didattico locale usando Gemma GGUF."""
from pathlib import Path
import json
from llama_cpp import Llama

MODEL = Path('models/gemma-3-4b-it-clean/gemma-3-4b-it-Q4_K_M.gguf')
OUT = Path('data/gemma_light_examples.jsonl')
PROMPTS = [
    'Crea un esempio breve su una regola grammaticale italiana, con domanda e risposta corrette. Non aggiungere testo extra.',
    'Crea un esempio breve di coding Python con una funzione semplice, codice funzionante e spiegazione di una frase. Non aggiungere testo extra.',
    'Crea un esempio di calcolo aritmetico elementare, mostrando il risultato verificabile in una riga. Non aggiungere testo extra.',
]

def main():
    if not MODEL.exists():
        raise SystemExit(f'Modello non trovato: {MODEL}')
    llm = Llama(model_path=str(MODEL), n_ctx=512, n_gpu_layers=4, verbose=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in PROMPTS:
        r = llm.create_chat_completion(messages=[{'role':'user','content':p}], max_tokens=100, temperature=0.10)
        text = r['choices'][0]['message']['content'].strip()
        rows.append({'source':'gemma', 'prompt':p, 'response':text})
    OUT.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')
    print(f'CREATED {OUT} ({len(rows)} esempi)')

if __name__ == '__main__':
    main()
