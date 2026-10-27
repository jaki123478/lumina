"""Upload Lumina 2.75M to Hugging Face Hub with a single command.

Usage:
  python upload_to_hf.py --repo-id your-username/lumina-2.75m --token hf_...

Or if you already ran `huggingface-cli login`:
  python upload_to_hf.py --repo-id your-username/lumina-2.75m
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Your Hugging Face repo ID, e.g., 'username/lumina-2.75m'")
    parser.add_argument("--token", default=None, help="Your Hugging Face write token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    folder = Path(__file__).parent if (Path(__file__).parent / "model.safetensors").exists() else Path("hf_export_lumina_large")

    print(f"Creating / verifying repo '{args.repo_id}' on Hugging Face...")
    create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)

    print(f"Uploading files from {folder} to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["*.pyc", "__pycache__", "upload_to_hf.py"]
    )
    print(f"\nUpload complete! Your model is live at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
