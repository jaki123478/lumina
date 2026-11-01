"""Generazione immagini 100% locale per Lumina AI.

Ordine dei motori locali:
1. Se `DIFFUSION_MODEL_PATH` punta a una pipeline Diffusers locale sul PC, usa quella.
2. Altrimenti usa direttamente il modello proprietario nativo `LuminaImageDenoiser`
   (`checkpoints_lumina_image/lumina_image_v1.pt`) con campionamento Rectified Flow
   e Classifier-Free Guidance — zero download esterni e zero errori di configurazione.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_NATIVE_IMAGE_MODEL = None
_NATIVE_IMAGE_SIZE = 96


def _clean_visual_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    text = re.sub(
        r"^(?:per favore\s+)?(?:lumina\s+)?(?:genera|crea|disegna|fammi|mostrami|produci)\s+(?:un['’]?\s*immagine\s+(?:di|con)\s+|una?\s+foto\s+di\s+|un\s+disegno\s+di\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,:;!?")
    return text or "tramonto sul mare"


def _load_native_model():
    global _NATIVE_IMAGE_MODEL, _NATIVE_IMAGE_SIZE
    if _NATIVE_IMAGE_MODEL is not None:
        return _NATIVE_IMAGE_MODEL, _NATIVE_IMAGE_SIZE

    import torch
    from lumina_image_experimental import create_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_v1 = ROOT / "checkpoints_lumina_image" / "lumina_image_v1.pt"
    ckpt_legacy = ROOT / "checkpoints_lumina_image" / "synthetic.pt"

    if ckpt_v1.exists():
        data = torch.load(ckpt_v1, map_location=device)
        vocab_size = int(data.get("vocab_size", 32000))
        channels = int(data.get("channels", 64))
        _NATIVE_IMAGE_SIZE = int(data.get("size", 96))
        model = create_model(device=device, vocab_size=vocab_size, channels=channels)
        state = data.get("ema_state_dict") or data.get("state_dict") or data
        model.load_state_dict(state, strict=True)
    elif ckpt_legacy.exists():
        model = create_model(device=device, vocab_size=32000, channels=64)
        state = torch.load(ckpt_legacy, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        raise RuntimeError(
            "Checkpoint Lumina Image non ancora presente: esegui train_lumina_image_synthetic.py"
        )

    model.eval()
    _NATIVE_IMAGE_MODEL = model
    return _NATIVE_IMAGE_MODEL, _NATIVE_IMAGE_SIZE


def generate_image(
    prompt: str,
    output_path: str | None = None,
    steps: int = 36,
    out_size: int = 384,
) -> str:
    clean_prompt = _clean_visual_prompt(prompt)

    if output_path:
        target = Path(output_path)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", clean_prompt.lower()).strip("_")[:32] or "immagine"
        out_dir = ROOT / "audio_neurale" / "immagini_generate"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"lumina_{slug}_{int(time.time())}.png"
    target.parent.mkdir(parents=True, exist_ok=True)

    # Percorso opzionale 1: pipeline Diffusers locale esterna se configurata
    model_path = os.environ.get("DIFFUSION_MODEL_PATH", "").strip()
    if model_path:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise RuntimeError(f"Modello immagini locale non trovato: {model_dir}")
        from diffusers import DiffusionPipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=dtype, local_files_only=True).to(device)
        image = pipe(clean_prompt, num_inference_steps=max(1, min(int(steps), 50))).images[0]
        image.save(target)
        return str(target)

    # Percorso predefinito 2: modello proprietario nativo LuminaImageDenoiser
    from PIL import Image
    from lumina_image_experimental import generate_rectified_flow

    model, native_size = _load_native_model()
    rgb = generate_rectified_flow(model, clean_prompt, steps=steps, size=native_size, cfg_scale=1.45)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr).resize((out_size, out_size), Image.Resampling.LANCZOS)
    img.save(target)
    return str(target)
