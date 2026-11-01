"""Smoke-trainer locale per Lumina Image usando forme sintetiche."""
from __future__ import annotations
import random
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from lumina_image_experimental import create_model


def sample(size=64):
    img = Image.new("RGB", (size, size), (random.randrange(20), random.randrange(20), random.randrange(20)))
    draw = ImageDraw.Draw(img)
    color = tuple(random.randrange(40, 240) for _ in range(3))
    x, y = random.randrange(8, 28), random.randrange(8, 28)
    w, h = random.randrange(16, 36), random.randrange(16, 36)
    if random.random() < .5: draw.rectangle((x, y, x+w, y+h), fill=color)
    else: draw.ellipse((x, y, x+w, y+h), fill=color)
    return torch.tensor(list(img.getdata()), dtype=torch.float32).view(size, size, 3).permute(2, 0, 1) / 127.5 - 1


def sample_orange_cat(size=64):
    """Crea un gatto stilizzato originale, senza immagini esterne."""
    img = Image.new("RGB", (size, size), (random.randrange(30, 90), random.randrange(90, 170), random.randrange(150, 230)))
    draw = ImageDraw.Draw(img)
    fur = (random.randrange(190, 245), random.randrange(80, 150), random.randrange(20, 70))
    left, top = random.randrange(8, 28), random.randrange(8, 24)
    width = random.randrange(28, 42); height = random.randrange(30, 46)
    right, bottom = left + width, top + height
    draw.ellipse((left, top + 8, right, bottom), fill=fur)
    draw.polygon((left + 2, top + 14, left + 3, top, left + 14, top + 11), fill=fur)
    draw.polygon((right - 14, top + 11, right - 2, top, right - 3, top + 18), fill=fur)
    eye_y = top + height // 2
    draw.ellipse((left + width // 4, eye_y, left + width // 4 + 4, eye_y + 5), fill=(20, 20, 20))
    draw.ellipse((left + width * 3 // 5, eye_y, left + width * 3 // 5 + 4, eye_y + 5), fill=(20, 20, 20))
    draw.ellipse((left + width // 2 - 2, eye_y + 9, left + width // 2 + 3, eye_y + 13), fill=(80, 25, 25))
    stripe = (120, 45, 15)
    for offset in (8, 15, 22):
        draw.line((left + offset, top + 12, left + offset - 4, top + 25), fill=stripe, width=2)
    for dy in (-3, 2, 7):
        draw.line((left + width // 2, eye_y + 12, left - 8, eye_y + 12 + dy), fill=(245, 245, 245), width=1)
        draw.line((left + width // 2, eye_y + 12, right + 8, eye_y + 12 + dy), fill=(245, 245, 245), width=1)
    draw.arc((left - 3, bottom - 8, right + 3, bottom + 12), 0, 180, fill=(20, 80, 170), width=3)
    return torch.tensor(list(img.getdata()), dtype=torch.float32).view(size, size, 3).permute(2, 0, 1) / 127.5 - 1


def train(steps=20, out="checkpoints_lumina_image/synthetic.pt", cats=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    for i in range(steps):
        source = sample_orange_cat if cats else sample
        clean = torch.stack([source() for _ in range(2)]).to(device)
        noise = torch.randn_like(clean) * .15
        token_ids = torch.zeros((clean.size(0), 4), dtype=torch.long, device=device)
        step = torch.full((clean.size(0),), min(i, 999), dtype=torch.long, device=device)
        pred = model(clean + noise, token_ids, step)
        loss = torch.nn.functional.mse_loss(pred, clean)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 5 == 0: print(f"step {i+1}/{steps} loss={loss.item():.4f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), out)
    print(f"checkpoint salvato: {out}")


if __name__ == "__main__": train()
