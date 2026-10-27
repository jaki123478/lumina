"""
train_bpe_tokenizer.py - Addestramento del Tokenizer BPE (32.000 Vocab) per Lumina-Next.
Addestrato congiuntamente su codice Python, SQL, Git, documentazione e lingua italiana.
"""

import os
import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tokenizers import ByteLevelBPETokenizer


def build_corpus_text(corpus_file: Path):
    """Estrae tutto il testo pulito dai dataset JSONL e dai file di codice."""
    print("[*] Estrazione testi di coding e italiano per l'addestramento...")
    lines_written = 0

    with open(corpus_file, "w", encoding="utf-8") as out:
        # 1. Dataset JSONL
        for jsonl_path in Path("data").glob("*.jsonl"):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                            u = d.get("user", "")
                            a = d.get("assistant", "")
                            if u:
                                out.write(u + "\n")
                                lines_written += 1
                            if a:
                                out.write(a + "\n")
                                lines_written += 1
                        except Exception:
                            continue
            except Exception:
                continue

        # 2. File sorgente Python del workspace come campioni di codice puro
        for py_file in Path(".").glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                out.write(content + "\n")
                lines_written += len(content.splitlines())
            except Exception:
                continue

    print(f"[+] Corpus compilato con {lines_written:,} righe di testo e codice.")


def train_tokenizer(vocab_size: int = 32000):
    corpus_file = Path("data") / "raw_corpus_tokenizer.txt"
    build_corpus_text(corpus_file)

    print(f"\n[*] Avvio addestramento Byte-Level BPE (Vocabolario: {vocab_size:,} token)...")
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[str(corpus_file)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|im_start|>",
            "<|im_end|>",
            "<|pad|>",
            "<|unk|>",
            "<|code_start|>",
            "<|code_end|>"
        ]
    )

    out_dir = Path("tokenizer_lumina_next")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out_dir))

    # Salva anche tokenizer_config.json
    config = {
        "model_type": "lumina_bpe",
        "vocab_size": vocab_size,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>"
    }
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[SUCCESS] Tokenizer salvato in '{out_dir}' (vocab.json + merges.txt)!")

    # Rimuovi file temporaneo
    if corpus_file.exists():
        try:
            corpus_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    train_tokenizer()
