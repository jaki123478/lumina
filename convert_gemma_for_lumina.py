"""Converte esempi Gemma filtrati nel formato conversazionale Lumina."""
from pathlib import Path
import json

SRC = Path('data/gemma_light_examples.jsonl')
OUT = Path('data/gemma_lumina_candidate.jsonl')

def main():
    rows = []
    for line in SRC.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get('prompt', '').strip()
        answer = item.get('response', '').strip()
        if not prompt or not answer or len(answer) > 1200:
            continue
        rows.append({'user': prompt, 'assistant': answer, 'split': 'train', 'kind': 'gemma_verified'})
    OUT.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'CONVERTED {len(rows)} examples -> {OUT}')

if __name__ == '__main__':
    main()
