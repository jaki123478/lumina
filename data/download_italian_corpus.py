"""Download rich, diverse, open Italian texts for Lumina pretraining.

Sources:
1. Italian Classic Literature (Manzoni, Pirandello, Boccaccio, Svevo, Verga, Salgari, Deledda, Leopardi...)
2. Italian Wikipedia essential knowledge articles across Science, Geography, History, Tech, AI, Arts.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_texts"
LIT_DIR = RAW_DIR / "literature"
WIKI_DIR = RAW_DIR / "wikipedia"

HEADERS = {"User-Agent": "LuminaBot/2.0 (contact: info@lumina.ai)"}

WIKI_TOPICS = [
    # Scienza e Natura
    "Intelligenza artificiale", "Apprendimento automatico", "Rete neurale artificiale", "Transformer (architettura)",
    "Python (linguaggio di programmazione)", "JavaScript", "Algoritmo", "Computer",
    "Sole", "Terra", "Luna", "Sistema solare", "Stella", "Galassia", "Acqua", "Atmosfera terrestre",
    "Gatto", "Cane", "Mammiferi", "Biologia", "Genetica", "Evoluzione", "Fisica quantistica",
    # Geografia
    "Italia", "Roma", "Milano", "Napoli", "Torino", "Firenze", "Venezia",
    "Francia", "Parigi", "Spagna", "Madrid", "Germania", "Berlino", "Regno Unito", "Londra",
    "Stati Uniti d'America", "Giappone", "Tokyo", "Cina", "Pechino",
    "Mar Mediterraneo", "Monte Bianco", "Alpi", "Tevere", "Po (fiume)",
    # Storia e Cultura
    "Antica Roma", "Rinascimento", "Impero romano", "Rivoluzione francese", "Prima guerra mondiale", "Seconda guerra mondiale",
    "Dante Alighieri", "Divina Commedia", "Leonardo da Vinci", "Alessandro Manzoni", "I promessi sposi",
    "Galileo Galilei", "Filosofia", "Letteratura italiana", "Storia d'Italia",
]


def download_literature() -> None:
    LIT_DIR.mkdir(parents=True, exist_ok=True)
    api_url = "https://huggingface.co/api/datasets/alberto98fx/Letteratura-Italiana"
    try:
        res = requests.get(api_url, timeout=15)
        res.raise_for_status()
        siblings = [f["rfilename"] for f in res.json().get("siblings", []) if f["rfilename"].endswith(".txt")]
    except Exception as e:
        print(f"Error querying literature API: {e}")
        return

    print(f"Downloading {len(siblings)} Italian literature books...")
    for filename in tqdm(siblings, desc="Literature"):
        dest = LIT_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        url = f"https://huggingface.co/datasets/alberto98fx/Letteratura-Italiana/resolve/main/{urllib.parse.quote(filename)}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 500:
                dest.write_text(r.text, encoding="utf-8", errors="replace")
        except Exception as err:
            print(f"Skipping {filename}: {err}")


def download_wikipedia() -> None:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(WIKI_TOPICS)} core Italian Wikipedia articles...")
    for title in tqdm(WIKI_TOPICS, desc="Wikipedia"):
        safe_name = re.sub(r'[\/:*?"<>| ]', "_", title) + ".txt"
        dest = WIKI_DIR / safe_name
        if dest.exists() and dest.stat().st_size > 500:
            continue
        url = f"https://it.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(title)}&format=json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for _, page in pages.items():
                    extract = page.get("extract", "")
                    if extract:
                        dest.write_text(f"{title}\n\n{extract}", encoding="utf-8")
        except Exception as err:
            print(f"Skipping wiki article {title}: {err}")


if __name__ == "__main__":
    download_wikipedia()
    download_literature()
    total_files = list(RAW_DIR.glob("**/*.txt"))
    total_bytes = sum(f.stat().st_size for f in total_files)
    print(f"\nCorpus downloaded successfully: {len(total_files)} files, {total_bytes / (1024 * 1024):.2f} MB of pure Italian text.")
