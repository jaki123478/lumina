"""Generazione immagini locale opzionale per Lumina.

Non scarica modelli e non usa servizi online: viene attivata solo quando
DIFFUSION_MODEL_PATH punta a una pipeline diffusers già presente sul PC.
"""
from __future__ import annotations

import os
from pathlib import Path


def generate_image(prompt: str, output_path: str | None = None, steps: int = 20) -> str:
    model_path = os.environ.get("DIFFUSION_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("Generatore immagini locale non configurato: imposta DIFFUSION_MODEL_PATH")
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise RuntimeError(f"Modello immagini locale non trovato: {model_dir}")
    try:
        from diffusers import DiffusionPipeline
        import torch
    except ImportError as exc:
        raise RuntimeError("Per la generazione locale servono diffusers e torch installati") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=dtype, local_files_only=True)
    pipe = pipe.to(device)
    image = pipe(prompt, num_inference_steps=max(1, min(int(steps), 50))).images[0]
    target = Path(output_path or (Path.home() / ".lumina" / "generated_image.png"))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return str(target)

