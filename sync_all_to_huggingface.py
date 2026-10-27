"""
sync_all_to_huggingface.py - Sincronizza tutti gli asset Lumina su Hugging Face:
1. Modello: jacopof/lumina-2.75m (safetensors + GGUF + PyTorch + Scheda tecnica)
2. Modello: jacopof/lumina-14m (safetensors + GGUF + Scheda tecnica)
3. Demo Space: jacopof/lumina-chat (Web app live con inferenza browser)
4. Dataset: jacopof/lumina-italian-distilled-qa (Dataset di distillazione italiana + coding)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent

def main():
    api = HfApi()
    user = api.whoami()["name"]
    print("=" * 65)
    print(f" Sincronizzazione Completa Ecosistema Lumina -> Hugging Face ({user})")
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. MODELLO: jacopof/lumina-2.75m
    # ---------------------------------------------------------
    model_2m_repo = f"{user}/lumina-2.75m"
    model_2m_folder = ROOT / "hf_export_lumina_large"
    # Copia modello ONNX e benchmark se presenti
    if (ROOT / "lumina_2.75m.onnx").exists():
        shutil.copy2(ROOT / "lumina_2.75m.onnx", model_2m_folder / "lumina_2.75m.onnx")
    if (ROOT / "BENCHMARK_REPORT.md").exists():
        shutil.copy2(ROOT / "BENCHMARK_REPORT.md", model_2m_folder / "BENCHMARK_REPORT.md")

    print(f"\n[1/4] Sincronizzazione Lumina 2.75M -> https://huggingface.co/{model_2m_repo} ...")
    create_repo(repo_id=model_2m_repo, repo_type="model", exist_ok=True)
    
    api.upload_folder(
        folder_path=str(model_2m_folder),
        repo_id=model_2m_repo,
        repo_type="model",
        commit_message="Update Lumina 2.75M: safetensors, GGUF, benchmarks and documentation",
        ignore_patterns=["*.pyc", "__pycache__"]
    )
    print(f"      [OK] Lumina 2.75M aggiornato!")

    # ---------------------------------------------------------
    # 2. MODELLO: jacopof/lumina-14m
    # ---------------------------------------------------------
    model_14m_repo = f"{user}/lumina-14m"
    model_14m_folder = ROOT / "hf_export_lumina_xlarge"
    if (ROOT / "lumina_14m.onnx").exists():
        shutil.copy2(ROOT / "lumina_14m.onnx", model_14m_folder / "lumina_14m.onnx")

    print(f"\n[2/5] Sincronizzazione Lumina 14M -> https://huggingface.co/{model_14m_repo} ...")
    create_repo(repo_id=model_14m_repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=str(model_14m_folder),
        repo_id=model_14m_repo,
        repo_type="model",
        commit_message="Update Lumina 14M: safetensors, GGUF, benchmarks and documentation",
        ignore_patterns=["*.pyc", "__pycache__"]
    )
    print(f"      [OK] Lumina 14M aggiornato!")

    # ---------------------------------------------------------
    # 3. MODELLO: jacopof/lumina-next-135m (Long Context 8k-32k)
    # ---------------------------------------------------------
    model_next_repo = f"{user}/lumina-next-135m"
    model_next_folder = ROOT / "hf_export_lumina_next"
    print(f"\n[3/5] Sincronizzazione Lumina-Next 135M -> https://huggingface.co/{model_next_repo} ...")
    create_repo(repo_id=model_next_repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=str(model_next_folder),
        repo_id=model_next_repo,
        repo_type="model",
        commit_message="Publish Lumina-Next 135M: RoPE, YaRN (8k-32k context), GQA, SwiGLU, BPE Tokenizer",
        ignore_patterns=["*.pyc", "__pycache__"]
    )
    print(f"      [OK] Lumina-Next 135M pubblicato con successo!")

    # ---------------------------------------------------------
    # 4. DEMO SPACE: jacopof/lumina-chat
    # ---------------------------------------------------------
    space_repo = f"{user}/lumina-chat"
    dist_folder = ROOT / "dist"
    print(f"\n[4/5] Sincronizzazione Demo Web Space -> https://huggingface.co/spaces/{space_repo} ...")
    create_repo(repo_id=space_repo, repo_type="space", space_sdk="static", exist_ok=True)

    api.upload_folder(
        folder_path=str(dist_folder),
        repo_id=space_repo,
        repo_type="space",
        commit_message="Update Lumina Web Demo: Latest neural weights and monitor",
        ignore_patterns=["*.pyc", "__pycache__", "*xlarge*"]
    )
    print(f"      [OK] Demo Web Space aggiornato!")

    # ---------------------------------------------------------
    # 5. DATASET: jacopof/lumina-italian-distilled-qa
    # ---------------------------------------------------------
    dataset_repo = f"{user}/lumina-italian-distilled-qa"
    dataset_dir = ROOT / "hf_export_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5/5] Sincronizzazione Dataset -> https://huggingface.co/datasets/{dataset_repo} ...")

    # Scrivi dataset pulito e validato
    src_data = ROOT / "data" / "distilled_dialogues.jsonl"
    dest_data = dataset_dir / "lumina_distilled_dialogues.jsonl"
    clean_count = 0
    if src_data.exists():
        with open(src_data, "r", encoding="utf-8") as f_in, open(dest_data, "w", encoding="utf-8") as f_out:
            for line in f_in:
                s = line.strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                    u = d.get("user", "").strip()
                    a = d.get("assistant", "").strip()
                    if u and a:
                        rec = {
                            "user": u,
                            "assistant": a,
                            "category": d.get("category", "dialogo_generale"),
                            "split": d.get("split", "train"),
                            "timestamp": float(d.get("timestamp", 1793016345.0)),
                        }
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        clean_count += 1
                except Exception:
                    continue

    # Converti anche in Parquet per anteprima istantanea su Hugging Face
    try:
        import pyarrow.json as paj
        import pyarrow.parquet as pq
        table = paj.read_json(str(dest_data))
        pq.write_table(table, str(dataset_dir / "train.parquet"))
        print(f"      [OK] Dataset Parquet generato ({clean_count:,} righe)!")
    except Exception as e:
        print(f"      [!] Nota Parquet: {e}")

    dataset_readme = f"""---
license: mit
language:
- it
task_categories:
- question-answering
- text-generation
tags:
- synthetic
- distillation
- italian
- coding
- lumina
size_categories:
- 10K<n<100K
---

# 📚 Lumina Italian Distillation & Coding Dataset

Questo dataset contiene oltre **11.000 coppie di dialogo, domande e risposte (Q&A) e compiti di programmazione in lingua italiana**, generate attraverso una pipeline di **Knowledge Distillation autonoma su GPU NVIDIA GeForce RTX 5060** per addestrare la famiglia di modelli [Lumina 2.75M](https://huggingface.co/{model_2m_repo}) e [Lumina 14M](https://huggingface.co/{model_14m_repo}).

## 🎯 Domini Inclusi

1. **Coding & Ingegneria del Software**:
   - Python: funzioni, dizionari, list comprehension, calcolo.
   - JavaScript moderno: DOM, async/await, eventi, array.
   - Algoritmi e complessità temporale ($O(n), O(\\log n)$).
   - Debugging e spiegazione errori (`IndexError`, `TypeError`, `SyntaxError`).
   - Git & Best Practices (`commit`, `branch`, principio DRY).
   - SQL e API REST (GET vs POST, JSON).
2. **Cultura e Geografia Italiana**:
   - Regioni, capoluoghi, monumenti (Roma, Firenze, Venezia, Napoli).
   - Grandi personaggi storici (Leonardo da Vinci, Dante, Rinascimento).
   - Tradizioni e cucina tipica regionale.
3. **Scienza e Natura**:
   - Astronomia, fotosintesi, gravità, biologia ed ecosistemi.
4. **Onestà e Rifiuti Anti-Allucinazione**:
   - Riconoscimento dei limiti su eventi futuri e dati privati, con risposte sincere ed educate.

## 💾 Struttura Dati

Ogni riga nel file `lumina_distilled_dialogues.jsonl` è un oggetto JSON con il seguente formato:

```json
{{
  "user": "Come creo una funzione in Python?",
  "assistant": "In Python puoi definire una funzione con la parola chiave 'def', seguita dal nome e dalle parentesi. Ad esempio: def saluta(): return 'Ciao!'",
  "category": "coding_python",
  "split": "train",
  "timestamp": 1793016345.9
}}
```

## 🚀 Utilizzo in Python

```python
import json

with open("lumina_distilled_dialogues.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        print("Domanda:", item["user"])
        print("Risposta:", item["assistant"])
        break
```

## 📜 Licenza
Rilasciato sotto licenza libera **MIT**. Creato da [Jacopo Forti (`jacopof`)](https://huggingface.co/{user}).
"""
    (dataset_dir / "README.md").write_text(dataset_readme, encoding="utf-8")

    print(f"\n[5/5] Upload file Dataset -> https://huggingface.co/datasets/{dataset_repo} ...")
    create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
    
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Publish updated Italian Distilled & Coding dataset for Lumina",
        ignore_patterns=["*.pyc", "__pycache__"]
    )
    print(f"      [OK] Dataset pubblicato con successo!")

    print("\n" + "=" * 65)
    print(" TUTTO SINCRONIZZATO CON SUCCESSO SU HUGGING FACE!")
    print(f" - Modello 2.75M  : https://huggingface.co/{model_2m_repo}")
    print(f" - Modello 14M    : https://huggingface.co/{model_14m_repo}")
    print(f" - Modello Next   : https://huggingface.co/{model_next_repo}")
    print(f" - Demo Space     : https://huggingface.co/spaces/{space_repo}")
    print(f" - Dataset Parquet: https://huggingface.co/datasets/{dataset_repo}")
    print("=" * 65)
    print(f" - Dataset       : https://huggingface.co/datasets/{dataset_repo}")
    print("=" * 65)

if __name__ == "__main__":
    main()
