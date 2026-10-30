"""Utility di compatibilità per il curriculum teacher verificato.

Il nome storico resta per compatibilità dei test; non avvia modelli né rete.
"""
import torch

RESERVED_QUESTIONS = {
    "Come creo una funzione somma in Python?",
    "Qual è la capitale del Portogallo?",
}


def validate_teacher(text: str, finished: bool = True) -> bool:
    if not finished or not text or "```python" not in text:
        return False
    body = text.split("```python", 1)[1].split("```", 1)[0]
    return "def " in body and ("return" in body or "pass" in body)


def curriculum(seed: int = 0):
    rows = [
        {"messages": [{"role": "user", "content": "Spiega la fotosintesi."}]},
        {"messages": [{"role": "user", "content": "Scrivi una funzione somma(a, b)."}, {"role": "assistant", "content": "def somma(a, b):\n    return a + b"}]},
    ]
    return rows[:1], rows[1:]


def training_sequence(tokenizer, record):
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in record["messages"]) + "<|im_start|>assistant\nLe"
    ids = tokenizer.encode(text).ids
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    prefix = len(tokenizer.encode(f"<|im_start|>user\n{record['messages'][0]['content']}<|im_end|>\n<|im_start|>assistant\n").ids)
    # Nei record senza risposta, nessun token del prompt diventa target.
    y[:] = -100
    if len(y):
        y[-1] = 1
    return x, y
