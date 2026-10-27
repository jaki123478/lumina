"""
upload_all_to_hf.py - Script di sincronizzazione globale con Hugging Face Hub.
Carica:
1. Modello jacopof/lumina-next-135m (safetensors, pytorch_model.bin, config, tokenizer, ONNX, GGUF, README)
2. Dataset jacopof/lumina-italian-distilled-qa (v2 con 16.363 esempi unici e README card)
3. Verifica e aggiornamento Space jacopof/lumina-chat
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def upload_lumina_next_model(api: HfApi):
    repo_id = "jacopof/lumina-next-135m"
    print(f"\n" + "=" * 65)
    print(f"[*] 1/3 SINCRONIZZAZIONE MODELLO: {repo_id}")
    print("=" * 65)

    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    folder = Path("hf_export_lumina_next")
    # Copia anche ONNX e GGUF nella cartella per il rilascio completo su HF
    if Path("lumina_next_135m.onnx").exists():
        shutil.copy("lumina_next_135m.onnx", folder / "lumina_next_135m.onnx")
    if Path("lumina_next_135m.gguf").exists():
        shutil.copy("lumina_next_135m.gguf", folder / "lumina_next_135m.gguf")

    print(f"[*] Caricamento cartella {folder} -> {repo_id}...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release Lumina-Next 135M Step 8000 (Loss: 0.1353, 16/16 100% Benchmark Score, Safetensors, ONNX, GGUF)",
    )
    print(f"[+] Modello Lumina-Next 135M aggiornato con successo: https://huggingface.co/{repo_id}")


def upload_distilled_dataset(api: HfApi):
    repo_id = "jacopof/lumina-italian-distilled-qa"
    print(f"\n" + "=" * 65)
    print(f"[*] 2/3 SINCRONIZZAZIONE DATASET: {repo_id}")
    print("=" * 65)

    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    folder = Path("hf_export_dataset")
    print(f"[*] Caricamento 28.807 campioni curati da {folder} -> {repo_id}...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update Lumina Dataset v4 CoT (28,807 curated samples with multi-step reasoning)",
    )
    print(f"[+] Dataset aggiornato con successo: https://huggingface.co/datasets/{repo_id}")


def upload_chat_space(api: HfApi):
    repo_id = "jacopof/lumina-chat"
    print(f"\n" + "=" * 65)
    print(f"[*] 3/3 SINCRONIZZAZIONE SPACE DEMO: {repo_id}")
    print("=" * 65)

    create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True)

    folder = Path("hf_space")
    print(f"[*] Caricamento Web App da {folder} -> {repo_id}...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Sync Lumina Chat Space with latest assets and configurations",
    )
    print(f"[+] Space aggiornato con successo: https://huggingface.co/spaces/{repo_id}")


def main():
    print("=" * 75)
    print("      🤗 LUMINA AI: HUGGING FACE GLOBAL SYNCHRONIZATION")
    print("=" * 75)

    api = HfApi()
    who = api.whoami()
    print(f"[*] Autenticato come: {who.get('name')} ({who.get('fullname')})")

    upload_lumina_next_model(api)
    upload_distilled_dataset(api)
    upload_chat_space(api)

    print("\n" + "=" * 75)
    print("      🎉 TUTTI I REPOSITORI HUGGING FACE AGGIORNATI CON SUCCESSO!")
    print("=" * 75)


if __name__ == "__main__":
    main()
