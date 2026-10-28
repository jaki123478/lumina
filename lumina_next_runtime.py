"""Formato condiviso per i checkpoint Next esistenti, senza accesso alla rete."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def active_model_dir():
    config = ROOT / 'config' / 'lumina_active_model.json'
    if config.exists():
        folder = (ROOT / json.loads(config.read_text(encoding='utf-8'))['path']).resolve()
        if not folder.is_relative_to(ROOT):
            raise ValueError('Il modello attivo deve essere nella cartella Lumina')
        return folder
    return ROOT / 'hf_export_lumina_next'


def prompt_ids(tokenizer, messages, budget=2048):
    """Mantiene scambi completi recenti; rifiuta l'ultimo turno troppo lungo."""
    turns = [(m['role'], str(m.get('content', ''))) for m in messages
             if m.get('role') in ('system', 'user', 'assistant')]
    if not turns or turns[-1][0] != 'user':
        raise ValueError('La conversazione deve terminare con un messaggio utente')
    def encode(items):
        text = ''.join(f'<|im_start|>{r}\n{t}<|im_end|>\n' for r, t in items)
        return tokenizer.encode(text + '<|im_start|>assistant\n').ids
    ids = encode(turns)
    while len(ids) > budget and len(turns) > 1:
        # Non lasciare una risposta orfana del suo messaggio utente.
        turns.pop(0)
        while len(turns) > 1 and turns[0][0] == 'assistant':
            turns.pop(0)
        ids = encode(turns)
    if len(ids) > budget:
        raise ValueError('Ultimo messaggio troppo lungo per il contesto del modello')
    return ids


def stop_sequences(tokenizer):
    # I checkpoint precedenti hanno imparato anche i marcatori come byte BPE.
    return [tokenizer.encode(s).ids for s in ('<|im_end|>', '<|im_start|>')]


def response_text(tokenizer, ids):
    text = tokenizer.decode(ids)
    for marker in ('<|im_end|>', '<|im_start|>', '<|pad|>'):
        text = text.split(marker, 1)[0]
    return text.strip()
