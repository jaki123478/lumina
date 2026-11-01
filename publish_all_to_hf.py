"""Sincronizzazione globale di tutti i modelli e asset Lumina su Hugging Face (jacopof)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent


def upload_repo_folder(api: HfApi, folder: Path, repo_id: str, commit_msg: str, repo_type: str = "model"):
    print(f"\n[*] Sincronizzazione {repo_id} ({folder.name})...", flush=True)
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_msg,
        ignore_patterns=["*.pyc", "__pycache__", "*.tmp", "*.log"],
    )
    print(f"[+] {repo_id} completato con successo!", flush=True)


def main():
    api = HfApi()
    user = api.whoami()["name"]
    print("=" * 65)
    print(f" 🚀 PUBBLICAZIONE COMPLETA LUMINA SU HUGGING FACE ({user})")
    print("=" * 65)

    # 1. LUMINA NEXT CONSOLIDATO (226M / 135M RoPE YaRN + Safetensors)
    export_226m = ROOT / "exports_lumina_226m_candidate"
    if (export_226m / "model.safetensors").exists():
        # Prepara cartella di esportazione completa
        hf_next_dir = ROOT / "hf_export_lumina_next"
        hf_next_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_226m / "model.safetensors", hf_next_dir / "model.safetensors")
        shutil.copy2(export_226m / "config.json", hf_next_dir / "config.json")
        shutil.copy2(export_226m / "vocab.json", hf_next_dir / "vocab.json")
        shutil.copy2(export_226m / "merges.txt", hf_next_dir / "merges.txt")

        # Includi modello Lumina Image se presente
        img_ckpt = ROOT / "checkpoints_lumina_image" / "lumina_image_v1.pt"
        if img_ckpt.exists():
            img_dst = hf_next_dir / "lumina_image_v1"
            img_dst.mkdir(exist_ok=True)
            shutil.copy2(img_ckpt, img_dst / "lumina_image_v1.pt")
            for f in ("lumina_image_gen.py", "lumina_image_experimental.py"):
                if (ROOT / f).exists():
                    shutil.copy2(ROOT / f, img_dst / f)

        upload_repo_folder(
            api,
            hf_next_dir,
            f"{user}/lumina-next-135m",
            "Update Lumina-Next: Cycle 3 loss-masked SFT consolidation, F1 realtime awareness & Lumina Image v1",
        )

        upload_repo_folder(
            api,
            hf_next_dir,
            f"{user}/lumina-complete",
            "Release Lumina-Complete: Sovereign weights, Italian SFT, Whisper Turbo offline STT & Lumina Image",
        )

    # 2. LUMINA 2.75M E 14M (Architettura leggera sovereign)
    if (ROOT / "hf_export_lumina_large").is_dir():
        upload_repo_folder(
            api,
            ROOT / "hf_export_lumina_large",
            f"{user}/lumina-2.75m",
            "Update Lumina 2.75M: sovereign edge weights & benchmarks",
        )

    if (ROOT / "hf_export_lumina_xlarge").is_dir():
        upload_repo_folder(
            api,
            ROOT / "hf_export_lumina_xlarge",
            f"{user}/lumina-14m",
            "Update Lumina 14M: 100% offline edge model",
        )

    # 3. LUMINA MAIN REPO
    if (ROOT / "hf_export_lumina_next").is_dir():
        upload_repo_folder(
            api,
            ROOT / "hf_export_lumina_next",
            f"{user}/lumina",
            "Update main Lumina repository: sovereign Italian LLM + Image + Voice models",
        )

    print("\n" + "=" * 65)
    print(" ✅ TUTTI GLI ASSET LUMINA SONO STATI PUBBLICATI SU HUGGING FACE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
