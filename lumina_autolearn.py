"""Raccolta controllata di esempi approvati per l'auto-miglioramento locale."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / 'data' / 'autonomy'
QUEUE = STATE_DIR / 'learning_queue.jsonl'
CURRICULUM = STATE_DIR / 'approved_curriculum.jsonl'


def _clean(value: object, limit: int = 4000) -> str:
    return ' '.join(str(value or '').split())[:limit]


def record_example(user: str, assistant: str, approved: bool = False, topic: str = 'general') -> dict:
    """Registra solo esempi esplicitamente approvati dall'utente."""
    item = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'user': _clean(user), 'assistant': _clean(assistant),
        'approved': bool(approved), 'topic': _clean(topic, 80),
    }
    if not item['user'] or not item['assistant']:
        raise ValueError('Esempio vuoto')
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    if item['approved']:
        with CURRICULUM.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'user': item['user'], 'assistant': item['assistant'], 'split': 'train'}, ensure_ascii=False) + '\n')
    return item


def approved_count() -> int:
    if not CURRICULUM.exists():
        return 0
    return sum(1 for line in CURRICULUM.read_text(encoding='utf-8').splitlines() if line.strip())
