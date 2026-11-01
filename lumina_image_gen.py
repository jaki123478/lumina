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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_NATIVE_IMAGE_MODEL = None
_NATIVE_IMAGE_SIZE = 96


def _resolve_existing(rel: str) -> Path:
    candidates = [Path.cwd() / rel, ROOT / rel]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / rel)
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([exe_dir / rel, exe_dir.parent.parent / rel])
    for c in candidates:
        if c.exists():
            return c
    return ROOT / rel


def _clean_visual_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    text = re.sub(
        r"^(?:per favore\s+)?(?:lumina\s+)?(?:genera|crea|disegna|fammi|mostrami|produci)\s+(?:un['’]?\s*immagine\s+(?:di|con)\s+|una?\s+foto\s+di\s+|un\s+disegno\s+di\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,:;!?")
    return text or "tramonto sul mare"


def _canonicalize_trained_prompt(clean_prompt: str) -> str:
    low = clean_prompt.lower()
    colors = ("rosso", "blu", "azzurro", "verde", "giallo", "arancione", "viola", "rosa", "bianco")
    col = next((c for c in colors if re.search(rf"\b{c}\b", low)), None)
    rules = [
        (r"\bfaro\b", "faro sul mare"),
        (r"\bcastello\b", "castello medievale sulla collina"),
        (r"\baurora\b", "aurora boreale nel cielo notturno"),
        (r"\bvulcano\b", "vulcano in eruzione con lava"),
        (r"\b(?:spiaggia|palma|tropicale)\b", "spiaggia tropicale con palma e mare turchese"),
        (r"\b(?:citt[àa]|grattaciel)\b", "città di notte illuminata"),
        (r"\bpianeta\b", f"pianeta {col or 'blu'} nello spazio"),
        (r"\b(?:gatto|gattino|micio)\b", f"un gatto {col or 'arancione'} nel prato"),
        (r"\bfarfalla\b", f"una farfalla {col or 'blu'}"),
        (r"\brobot\b", "un robot futuristico"),
        (r"\b(?:barca|vela)\b", "barca a vela sul mare"),
        (r"\b(?:casa|casetta|villa)\b", f"una casa con il tetto {col or 'rosso'}"),
        (r"\b(?:foresta|pini|bosco)\b", "foresta di pini"),
        (r"\bmontagn", "montagne innevate"),
        (r"\balbero\b", "un albero verde nel prato"),
        (r"\bfiore\b", f"un fiore {col or 'rosso'}"),
        (r"\bcuore\b", f"un cuore {col or 'rosso'}"),
        (r"\b(?:notte\s+stellata|stelle|luna)\b", "notte stellata"),
        (r"\btramonto\b", "tramonto sul mare"),
    ]
    for pat, target_p in rules:
        if re.search(pat, low):
            return target_p
    return clean_prompt
def _load_native_model():
    global _NATIVE_IMAGE_MODEL, _NATIVE_IMAGE_SIZE
    if _NATIVE_IMAGE_MODEL is not None:
        return _NATIVE_IMAGE_MODEL, _NATIVE_IMAGE_SIZE

    import torch
    from lumina_image_experimental import create_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_v1 = _resolve_existing("checkpoints_lumina_image/lumina_image_v1.pt")
    ckpt_legacy = _resolve_existing("checkpoints_lumina_image/synthetic.pt")

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
        base_out = Path.cwd() if getattr(sys, "frozen", False) else ROOT
        out_dir = base_out / "audio_neurale" / "immagini_generate"
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
    from PIL import Image, ImageFilter
    from lumina_image_experimental import generate_rectified_flow

    focus_prompt = _canonicalize_trained_prompt(clean_prompt)

    model, native_size = _load_native_model()
    rgb = generate_rectified_flow(model, focus_prompt, steps=steps, size=native_size, cfg_scale=1.45)
    arr = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr).resize((out_size, out_size), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8)).filter(ImageFilter.UnsharpMask(radius=1.5, percent=115, threshold=2))
    img.save(target)
    return str(target)
