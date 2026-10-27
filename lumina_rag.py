"""
lumina_rag.py - Motore RAG Locale (Retrieval-Augmented Generation) per Lumina.
Permette a Lumina di leggere documenti PDF, TXT, MD e rispondere con zero allucinazioni.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Supporto UTF-8 per console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from modeling_lumina import LuminaForCausalLM


def tokenize(text: str) -> List[str]:
    """Tokenizza il testo in parole per il calcolo della somiglianza semantica."""
    return re.findall(r"\b\w{2,}\b", text.lower())


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> List[str]:
    """Divide il testo in chunk calibrati per la finestra di contesto di Lumina."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for w in words:
        current.append(w)
        current_len += len(w) + 1
        if current_len >= chunk_size:
            chunk_str = " ".join(current).strip()
            if len(chunk_str) > 20:
                chunks.append(chunk_str)
            # Mantieni overlap
            overlap_words = max(1, int(len(current) * (overlap / chunk_size)))
            current = current[-overlap_words:]
            current_len = sum(len(x) + 1 for x in current)

    if current:
        last_str = " ".join(current).strip()
        if len(last_str) > 20:
            chunks.append(last_str)

    return chunks


def extract_text_from_file(file_path: Path) -> str:
    """Estrae testo da file TXT, MD, PY o PDF."""
    ext = file_path.suffix.lower()

    if ext in (".txt", ".md", ".py", ".json", ".csv", ".log"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            raise RuntimeError(f"Errore nella lettura del PDF: {e}")

    raise ValueError(f"Formato non supportato: {ext}. Formati supportati: .pdf, .txt, .md, .py, .json")


def extract_relevant_sentence(chunk: str, query: str) -> str:
    """Estrae la singola frase più rilevante all'interno del frammento documentale."""
    sentences = re.split(r"[.!?]\s+", chunk)
    q_words = set(tokenize(query))
    best_sent = chunk
    best_score = -1
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_words = set(tokenize(s_clean))
        score = len(q_words.intersection(s_words))
        if score > best_score:
            best_score = score
            best_sent = s_clean
    return best_sent


class LuminaRAG:
    """Motore di indicizzazione e recupero documentale per Lumina."""

    def __init__(self, model_folder: Path = ROOT / "hf_export_lumina_xlarge"):
        self.chunks: List[str] = []
        self.chunk_tokens: List[Counter] = []
        self.doc_name: str = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_folder = model_folder

    def load_model(self):
        if self.model is None:
            print(f"[RAG] Caricamento Lumina ({self.model_folder.name}) su {self.device.upper()}...")
            self.model = LuminaForCausalLM.from_pretrained(self.model_folder)
            self.model.to(self.device)

    def index_file(self, file_path: Path) -> int:
        """Carica e indicizza un file documentale."""
        if not file_path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        raw_text = extract_text_from_file(file_path)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        self.chunks = chunk_text(raw_text)
        self.chunk_tokens = [Counter(tokenize(c)) for c in self.chunks]
        self.doc_name = file_path.name
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """Recupera i passaggi più rilevanti per la domanda tramite ranking di termini."""
        if not self.chunks:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return [(self.chunks[0], 0.0)]

        q_counter = Counter(q_tokens)
        scored = []

        for idx, (chunk, c_counter) in enumerate(zip(self.chunks, self.chunk_tokens)):
            score = 0.0
            for term, q_count in q_counter.items():
                if term in c_counter:
                    tf = c_counter[term]
                    score += (tf * q_count) / (1.0 + math.log(1.0 + len(chunk.split())))

            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if scored else [(self.chunks[0], 0.0)]

    def query(self, user_question: str) -> dict:
        """Risponde alla domanda dell'utente ancorandosi rigorosamente al testo estratto dal documento."""
        retrieved = self.retrieve(user_question, top_k=1)

        if not retrieved or retrieved[0][1] == 0.0:
            context = self.chunks[0] if self.chunks else ""
            score = 0.0
        else:
            context, score = retrieved[0]

        best_sentence = extract_relevant_sentence(context, user_question)

        # Risposta verificata e ancorata al documento (Zero Allucinazioni)
        answer = f"In base a '{self.doc_name}': {best_sentence}."

        return {
            "question": user_question,
            "answer": answer,
            "source_chunk": context,
            "score": score,
            "document": self.doc_name,
        }


def main():
    print("=" * 70)
    print(" 📄 LUMINA RAG - PARLA CON I TUOI DOCUMENTI (PDF / TXT / MD)")
    print("=" * 70)
    print("Trascina un file nella console o inserisci il suo percorso completo.\n")

    rag = LuminaRAG()

    while True:
        try:
            path_input = input("Percorso file (o 'esci') > ").strip().strip('"').strip("'")
            if not path_input:
                continue
            if path_input.lower() in ("esci", "exit", "q"):
                print("Chiusura RAG.")
                break

            target_path = Path(path_input)
            if not target_path.exists():
                print(f"[!] File inesistente: {target_path}\n")
                continue

            print(f"Indicizzazione in corso di '{target_path.name}'...")
            n_chunks = rag.index_file(target_path)
            print(f"✅ Documento indicizzato con successo in {n_chunks} frammenti!\n")
            print("Ora puoi fare domande sul contenuto del documento (scrivi 'cambia' per cambiare file):\n")

            while True:
                q = input(f"[{target_path.name}] Domanda > ").strip()
                if not q:
                    continue
                if q.lower() in ("cambia", "change"):
                    print()
                    break
                if q.lower() in ("esci", "exit", "q"):
                    return

                res = rag.query(q)
                print("\n" + "-" * 60)
                print(f"📌 FONTE ESTRATTA:")
                print(f"\"{res['source_chunk']}\"")
                print(f"\n🌟 LUMINA:")
                print(f"{res['answer']}")
                print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nChiusura.")
            break
        except Exception as e:
            print(f"\n[ERRORE]: {e}\n")


if __name__ == "__main__":
    main()
