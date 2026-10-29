"""Supervisore locale sicuro per Lumina.

Controlla se il server risponde e scrive solo stato/log nella directory
data/autonomy. Non esegue codice ricevuto dalla chat e non modifica pesi o
file del progetto.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "data" / "autonomy"
HEARTBEAT = STATE_DIR / "heartbeat.json"
EVENT_LOG = STATE_DIR / "events.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def check_server(url: str = "http://127.0.0.1:7020/api/status", timeout: float = 3.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"online": True, "http": response.status, "payload": payload}
    except Exception as exc:
        return {"online": False, "error": str(exc)}


def record(state: dict) -> dict:
    event = {"timestamp": _now(), **state}
    _atomic_write(HEARTBEAT, json.dumps(event, ensure_ascii=False, indent=2) + "\n")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def run_once() -> dict:
    return record(check_server())


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisore locale di Lumina")
    parser.add_argument("--interval", type=float, default=10.0, help="secondi tra i controlli")
    parser.add_argument("--once", action="store_true", help="esegue un solo controllo")
    args = parser.parse_args()
    if args.once:
        print(json.dumps(run_once(), ensure_ascii=False))
        return
    while True:
        event = run_once()
        print(json.dumps(event, ensure_ascii=False), flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
