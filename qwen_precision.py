"""Direttiva locale riletta per ogni lezione, senza cambiare i pesi del maestro."""
from pathlib import Path

DIRECTIVE = Path(__file__).resolve().parent / 'config' / 'qwen_precision_directive.txt'


def precision_instruction():
    return DIRECTIVE.read_text(encoding='utf-8').strip()
