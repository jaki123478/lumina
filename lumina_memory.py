"""
Lumina Memory Bank - Memoria a Lungo Termine Persistente per Lumina AI.
Archiviazione locale privata su SQLite (data/lumina_memory.db).
Ricorda profilo utente, preferenze, progetti in corso e frammenti di codice.
"""

import os
import sys
import time
import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_DIR = Path("data")
DB_PATH = DB_DIR / "lumina_memory.db"


class LuminaMemory:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Inizializza le tabelle del database se non esistono."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Profilo utente e preferenze (Chiave-Valore)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            # 2. Ricordi episodici (Fatti, conversazioni salienti)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    fact TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            # 3. Snippet di codice preferiti o salvati
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_vault (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    language TEXT NOT NULL,
                    code TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    # --- GESTIONE PROFILO UTENTE ---
    def set_profile_fact(self, key: str, value: str):
        """Salva o aggiorna un fatto nel profilo utente (es. 'nome', 'progetto_corrente')."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_profile (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key.lower().strip(), value.strip(), time.time()))
            conn.commit()

    def get_profile_fact(self, key: str) -> Optional[str]:
        """Recupera un fatto dal profilo utente."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM user_profile WHERE key = ?", (key.lower().strip(),))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_profile(self) -> Dict[str, str]:
        """Restituisce l'intero profilo utente."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM user_profile")
            return {k: v for k, v in cursor.fetchall()}

    # --- RICORDI EPISODICI & FATTI ---
    def remember_fact(self, fact: str, topic: str = "generale"):
        """Archivia un ricordo o informazione specifica."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodic_memory (topic, fact, created_at)
                VALUES (?, ?, ?)
            """, (topic.strip(), fact.strip(), time.time()))
            conn.commit()

    def recall_facts(self, query: str = "", limit: int = 5) -> List[str]:
        """Cerca fatti pertinenti memorizzati in passato."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if query:
                cursor.execute("""
                    SELECT fact FROM episodic_memory 
                    WHERE fact LIKE ? OR topic LIKE ?
                    ORDER BY id DESC LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
            else:
                cursor.execute("SELECT fact FROM episodic_memory ORDER BY id DESC LIMIT ?", (limit,))
            return [row[0] for row in cursor.fetchall()]

    # --- ARCHIVIO CODICE (CODE VAULT) ---
    def save_code_snippet(self, title: str, code: str, language: str = "python"):
        """Salva uno snippet di codice utile nella cassaforte."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO code_vault (title, language, code, created_at)
                VALUES (?, ?, ?, ?)
            """, (title.strip(), language.lower().strip(), code.strip(), time.time()))
            conn.commit()

    def search_code_snippets(self, query: str) -> List[Dict[str, str]]:
        """Cerca codice memorizzato per titolo o contenuto."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT title, language, code FROM code_vault
                WHERE title LIKE ? OR code LIKE ?
                ORDER BY id DESC LIMIT 5
            """, (f"%{query}%", f"%{query}%"))
            return [{"title": r[0], "language": r[1], "code": r[2]} for r in cursor.fetchall()]

    # --- AUTO-APPRENDIMENTO DAI MESSAGGI ---
    def auto_extract_memories(self, user_msg: str):
        """Estrae automaticamente preferenze o identità dai messaggi dell'utente."""
        msg = user_msg.lower()

        # Rilevamento Nome
        name_match = re.search(r"\b(mi chiamo|il mio nome è|sono)\s+([A-Z][a-z]+)\b", user_msg)
        if name_match:
            name = name_match.group(2)
            self.set_profile_fact("nome_utente", name)

        # Rilevamento Linguaggio Preferito
        if "programmo in python" in msg or "preferisco python" in msg:
            self.set_profile_fact("linguaggio_preferito", "Python")
        elif "programmo in javascript" in msg or "preferisco javascript" in msg:
            self.set_profile_fact("linguaggio_preferito", "JavaScript")

        # Rilevamento Progetti
        proj_match = re.search(r"\b(sto lavorando a|progetto)\s+([a-zA-Z0-9_\-\s]{3,20})\b", user_msg, re.IGNORECASE)
        if proj_match:
            self.set_profile_fact("progetto_attivo", proj_match.group(2).strip())

    def get_memory_context_string(self) -> str:
        """Genera una stringa sintetica da iniettare nel prompt per personalizzare la risposta."""
        profile = self.get_all_profile()
        if not profile:
            return ""
        items = [f"{k}: {v}" for k, v in profile.items()]
        return f"[Memoria Utente: {', '.join(items)}]"


if __name__ == "__main__":
    mem = LuminaMemory()
    print("=== TEST LUMINA MEMORY BANK ===")
    mem.set_profile_fact("nome_sviluppatore", "Jacopo")
    mem.set_profile_fact("progetto", "Lumina AI Sovereign Copilot")
    mem.set_profile_fact("gpu", "NVIDIA RTX 5060 Laptop GPU")
    mem.remember_fact("Jacopo preferisce codice Python sintetico con type hints.")
    
    print("Profilo memorizzato:")
    for k, v in mem.get_all_profile().items():
        print(f"  • {k}: {v}")

    print("\nFatti ricordati:")
    for f in mem.recall_facts():
        print(f"  • {f}")

    print("\nStringa di contesto per il modello:")
    print(mem.get_memory_context_string())
