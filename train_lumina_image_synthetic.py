"""Trainer Rectified-Flow + Classifier-Free Guidance per Lumina Image.

Addestra il modello nativo LuminaImageDenoiser su un curriculum procedurale
ricco in italiano (12 categorie illustrate + 8 varianti cromatiche, senza
scaricare immagini esterne) e salva sia i pesi EMA sia un mosaico di verifica.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from lumina_image_experimental import (
    create_model,
    encode_prompt,
    generate_rectified_flow,
)

PALETTE = {
    "rosso": ((230, 55, 65), ["rosso", "scarlatto", "vermiglio"]),
    "blu": ((55, 120, 235), ["blu", "azzurro", "celeste"]),
    "verde": ((55, 195, 95), ["verde", "smeraldo"]),
    "giallo": ((245, 210, 55), ["giallo", "dorato", "oro"]),
    "arancione": ((245, 130, 45), ["arancione", "ambra"]),
    "viola": ((165, 75, 225), ["viola", "porpora", "lilla"]),
    "rosa": ((240, 105, 175), ["rosa", "fucsia"]),
    "bianco": ((240, 244, 250), ["bianco", "luminoso", "chiaro"]),
}


def _jitter(rgb: tuple[int, int, int], spread: int = 16) -> tuple[int, int, int]:
    return tuple(max(0, min(255, c + random.randint(-spread, spread))) for c in rgb)


def _vertical_gradient(size: int, top_rgb: tuple[int, int, int], bot_rgb: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    denom = max(1, size - 1)
    for y in range(size):
        alpha = y / denom
        r = int(top_rgb[0] * (1 - alpha) + bot_rgb[0] * alpha)
        g = int(top_rgb[1] * (1 - alpha) + bot_rgb[1] * alpha)
        b = int(top_rgb[2] * (1 - alpha) + bot_rgb[2] * alpha)
        d.line((0, y, size - 1, y), fill=(r, g, b))
    return img


def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    size = img.size[0]
    return torch.tensor(bytearray(img.tobytes()), dtype=torch.float32).view(size, size, 3).permute(2, 0, 1) / 127.5 - 1.0


def scene_cat(size: int = 96) -> tuple[Image.Image, str]:
    cname = random.choice(["arancione", "giallo", "bianco", "viola", "rosa"])
    fur_base, cwords = PALETTE[cname]
    img = _vertical_gradient(size, _jitter((55, 125, 210), 8), _jitter((62, 162, 88), 8))
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(size * 0.68), size, size), fill=_jitter((54, 152, 74), 8))
    fur = _jitter(fur_base, 10)
    fur_dark = tuple(max(0, c - 32) for c in fur)
    cx = size // 2 + random.randint(-2, 2)
    cy = int(size * 0.50)
    hr = int(size * 0.23)
    # Corpo seduto
    d.ellipse((cx - int(hr * 1.05), cy + int(hr * 0.45), cx + int(hr * 1.05), size - 4), fill=fur_dark)
    # Orecchie a punta + interno rosa
    d.polygon([(cx - hr + 2, cy - int(hr * 0.45)), (cx - int(hr * 0.85), cy - int(hr * 1.48)), (cx - int(hr * 0.18), cy - int(hr * 0.82))], fill=fur)
    d.polygon([(cx + int(hr * 0.18), cy - int(hr * 0.82)), (cx + int(hr * 0.85), cy - int(hr * 1.48)), (cx + hr - 2, cy - int(hr * 0.45))], fill=fur)
    d.polygon([(cx - int(hr * 0.78), cy - int(hr * 0.58)), (cx - int(hr * 0.76), cy - int(hr * 1.25)), (cx - int(hr * 0.30), cy - int(hr * 0.80))], fill=(235, 135, 155))
    d.polygon([(cx + int(hr * 0.30), cy - int(hr * 0.80)), (cx + int(hr * 0.76), cy - int(hr * 1.25)), (cx + int(hr * 0.78), cy - int(hr * 0.58))], fill=(235, 135, 155))
    # Testa rotonda
    d.ellipse((cx - hr, cy - int(hr * 0.90), cx + hr, cy + int(hr * 0.92)), fill=fur)
    # Occhi espressivi, naso e baffi
    ey = cy - int(hr * 0.12)
    ex = int(hr * 0.44)
    d.ellipse((cx - ex - 5, ey - 5, cx - ex + 5, ey + 5), fill=(75, 215, 115))
    d.ellipse((cx + ex - 5, ey - 5, cx + ex + 5, ey + 5), fill=(75, 215, 115))
    d.ellipse((cx - ex - 2, ey - 4, cx - ex + 2, ey + 4), fill=(15, 18, 22))
    d.ellipse((cx + ex - 2, ey - 4, cx + ex + 2, ey + 4), fill=(15, 18, 22))
    ny = cy + int(hr * 0.22)
    d.polygon([(cx - 5, ny - 3), (cx + 5, ny - 3), (cx, ny + 4)], fill=(225, 95, 115))
    for dy in (-4, 1, 6):
        d.line((cx - 8, ny + 2, cx - hr - 8, ny + dy), fill=(250, 250, 250), width=1)
        d.line((cx + 8, ny + 2, cx + hr + 8, ny + dy), fill=(250, 250, 250), width=1)
    prompt = random.choice([
        f"un gatto {random.choice(cwords)}",
        f"gattino {random.choice(cwords)} nel prato",
        f"ritratto di gatto {random.choice(cwords)}",
    ])
    return img, prompt


def scene_sunset_sea(size: int = 96) -> tuple[Image.Image, str]:
    horizon = int(size * 0.56)
    img = _vertical_gradient(size, _jitter((95, 35, 115)), _jitter((250, 125, 45)))
    d = ImageDraw.Draw(img)
    sx = size // 2 + random.randint(-14, 14)
    sr = int(size * 0.16)
    d.ellipse((sx - sr - 5, horizon - sr - 5, sx + sr + 5, horizon + sr + 5), fill=(255, 175, 70))
    d.ellipse((sx - sr, horizon - sr, sx + sr, horizon + sr), fill=(255, 232, 120))
    # Mare con riflessi
    for y in range(horizon, size):
        t = (y - horizon) / max(1, size - horizon)
        col = (int(35 * (1 - t) + 12 * t), int(75 * (1 - t) + 35 * t), int(145 * (1 - t) + 95 * t))
        d.line((0, y, size, y), fill=col)
        if y % 3 == 0:
            w = max(3, int(sr * (1.1 - 0.7 * t)))
            d.line((sx - w, y, sx + w, y), fill=(255, 195, 95), width=1)
    prompt = random.choice([
        "tramonto sul mare",
        "sole arancione sul mare al tramonto",
        "paesaggio marino al tramonto",
        "oceano al tramonto con riflessi dorati",
    ])
    return img, prompt


def scene_starry_night(size: int = 96) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((10, 14, 42), 6), _jitter((28, 42, 92), 10))
    d = ImageDraw.Draw(img)
    for _ in range(38):
        x, y = random.randrange(2, size - 2), random.randrange(2, int(size * 0.74))
        b = random.randint(190, 255)
        d.point((x, y), fill=(b, b, 255))
        if random.random() < 0.25:
            d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 250, 210))
    mx, my, mr = random.randint(int(size * 0.25), int(size * 0.75)), int(size * 0.26), int(size * 0.13)
    d.ellipse((mx - mr - 4, my - mr - 4, mx + mr + 4, my + mr + 4), fill=(70, 90, 155))
    d.ellipse((mx - mr, my - mr, mx + mr, my + mr), fill=(250, 246, 210))
    # Colline notturne
    d.polygon([(0, size), (0, int(size * 0.78)), (size // 3, int(size * 0.68)), (2 * size // 3, int(size * 0.76)), (size, int(size * 0.70)), (size, size)], fill=(12, 18, 34))
    prompt = random.choice([
        "notte stellata con la luna",
        "cielo stellato notturno e luna piena",
        "luna luminosa nella notte stellata",
    ])
    return img, prompt


def scene_mountains(size: int = 96) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((70, 145, 235)), _jitter((185, 222, 250)))
    d = ImageDraw.Draw(img)
    # Catena montuosa + cime innevate
    p1 = [(0, int(size * 0.80)), (int(size * 0.32), int(size * 0.26)), (int(size * 0.65), int(size * 0.80))]
    p2 = [(int(size * 0.28), int(size * 0.82)), (int(size * 0.68), int(size * 0.20)), (size, int(size * 0.82))]
    d.polygon(p1, fill=_jitter((78, 92, 122), 10))
    d.polygon([(int(size * 0.22), int(size * 0.43)), (int(size * 0.32), int(size * 0.26)), (int(size * 0.42), int(size * 0.43))], fill=(246, 250, 255))
    d.polygon(p2, fill=_jitter((62, 76, 106), 10))
    d.polygon([(int(size * 0.56), int(size * 0.40)), (int(size * 0.68), int(size * 0.20)), (int(size * 0.79), int(size * 0.40))], fill=(248, 252, 255))
    d.rectangle((0, int(size * 0.78), size, size), fill=_jitter((52, 138, 72), 12))
    prompt = random.choice([
        "montagne innevate con cielo azzurro",
        "paesaggio di montagna sulle alpi",
        "vette innevate e prato verde",
    ])
    return img, prompt


def scene_tree(size: int = 96) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((95, 180, 248)), _jitter((195, 235, 255)))
    d = ImageDraw.Draw(img)
    ground_y = int(size * 0.74)
    d.rectangle((0, ground_y, size, size), fill=_jitter((68, 172, 76), 14))
    tx = size // 2 + random.randint(-8, 8)
    d.rectangle((tx - 6, int(size * 0.48), tx + 6, ground_y + 4), fill=_jitter((115, 72, 38), 10))
    for ox, oy, r in [(0, -16, 22), (-14, -6, 17), (14, -6, 17), (0, -2, 19)]:
        d.ellipse((tx + ox - r, int(size * 0.44) + oy - r, tx + ox + r, int(size * 0.44) + oy + r), fill=_jitter((44, 158, 64), 16))
    prompt = random.choice([
        "albero verde in un prato soleggiato",
        "grande albero nella natura",
        "bosco verde e cielo sereno",
    ])
    return img, prompt


def scene_city_night(size: int = 96) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((18, 14, 48), 8), _jitter((68, 32, 96), 12))
    d = ImageDraw.Draw(img)
    x = 4
    base_y = int(size * 0.84)
    while x < size - 8:
        bw = random.randint(12, 20)
        bh = random.randint(int(size * 0.28), int(size * 0.62))
        top = base_y - bh
        d.rectangle((x, top, x + bw, base_y), fill=_jitter((24, 28, 48), 6))
        for wy in range(top + 4, base_y - 4, 6):
            for wx in range(x + 3, x + bw - 3, 5):
                if random.random() < 0.65:
                    wcol = random.choice([(255, 225, 115), (110, 225, 255), (255, 120, 210)])
                    d.rectangle((wx, wy, wx + 2, wy + 3), fill=wcol)
        x += bw + random.randint(2, 5)
    d.rectangle((0, base_y, size, size), fill=(16, 20, 36))
    prompt = random.choice([
        "città di notte illuminata",
        "grattacieli illuminati di notte",
        "skyline urbano cyberpunk notturno",
    ])
    return img, prompt


def scene_flower(size: int = 96) -> tuple[Image.Image, str]:
    cname = random.choice(["rosso", "rosa", "giallo", "viola", "blu"])
    petal_rgb, cwords = PALETTE[cname]
    img = _vertical_gradient(size, _jitter((95, 185, 245), 8), _jitter((45, 135, 65), 8))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-2, 2), int(size * 0.42)
    d.line((cx, cy + 4, cx, size - 2), fill=(36, 155, 58), width=5)
    d.ellipse((cx - 18, int(size * 0.68), cx - 2, int(size * 0.78)), fill=(44, 172, 66))
    d.ellipse((cx + 2, int(size * 0.62), cx + 18, int(size * 0.72)), fill=(44, 172, 66))
    pr = int(size * 0.105)
    dist = int(size * 0.19)
    pcol = _jitter(petal_rgb, 10)
    for k in range(6):
        ang = k * (2 * math.pi / 6)
        px = int(cx + math.cos(ang) * dist)
        py = int(cy + math.sin(ang) * dist)
        d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=pcol)
    cr = int(size * 0.10)
    d.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill=(252, 215, 52))
    prompt = random.choice([
        f"un fiore {random.choice(cwords)}",
        f"fiore {random.choice(cwords)} nel giardino",
        f"petali di fiore {random.choice(cwords)}",
    ])
    return img, prompt


def scene_house(size: int = 96) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((105, 185, 250)), _jitter((195, 232, 255)))
    d = ImageDraw.Draw(img)
    gy = int(size * 0.72)
    d.rectangle((0, gy, size, size), fill=_jitter((74, 168, 78), 12))
    hx1, hx2 = int(size * 0.28), int(size * 0.72)
    hy1 = int(size * 0.45)
    d.rectangle((hx1, hy1, hx2, gy), fill=_jitter((242, 232, 210), 10))
    d.polygon([(hx1 - 6, hy1), (size // 2, int(size * 0.24)), (hx2 + 6, hy1)], fill=_jitter((205, 58, 52), 12))
    d.rectangle((size // 2 - 6, gy - 16, size // 2 + 6, gy), fill=(120, 74, 42))
    d.rectangle((hx1 + 6, hy1 + 6, hx1 + 14, hy1 + 14), fill=(255, 225, 115))
    d.rectangle((hx2 - 14, hy1 + 6, hx2 - 6, hy1 + 14), fill=(255, 225, 115))
    prompt = random.choice([
        "una casa con il tetto rosso",
        "casetta di campagna nel prato",
        "casa accogliente sotto il cielo azzurro",
    ])
    return img, prompt


def scene_heart(size: int = 96) -> tuple[Image.Image, str]:
    cname = random.choice(["rosso", "rosa", "viola", "giallo"])
    hrgb, cwords = PALETTE[cname]
    img = _vertical_gradient(size, _jitter((28, 16, 42)), _jitter((68, 24, 62)))
    d = ImageDraw.Draw(img)
    pts = []
    cx, cy = size / 2.0, size * 0.50
    scale = size * 0.019
    for deg in range(0, 360, 4):
        t = math.radians(deg)
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * scale, cy + y * scale))
    d.polygon(pts, fill=_jitter(hrgb, 10))
    prompt = random.choice([
        f"un cuore {random.choice(cwords)} luminoso",
        f"cuore {random.choice(cwords)}",
        f"simbolo di cuore {random.choice(cwords)} brillante",
    ])
    return img, prompt


def scene_planet_space(size: int = 96) -> tuple[Image.Image, str]:
    cname = random.choice(["blu", "arancione", "viola", "verde", "rosso"])
    prgb, cwords = PALETTE[cname]
    img = _vertical_gradient(size, (6, 8, 22), (18, 12, 38))
    d = ImageDraw.Draw(img)
    for _ in range(42):
        d.point((random.randrange(size), random.randrange(size)), fill=(235, 240, 255))
    cx, cy, r = size // 2, size // 2, int(size * 0.22)
    ring_box = (cx - int(r * 1.95), cy - int(r * 0.48), cx + int(r * 1.95), cy + int(r * 0.48))
    d.arc(ring_box, 180, 360, fill=(195, 175, 138), width=4)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_jitter(prgb, 12))
    # Fascia atmosferica luminosa
    d.chord((cx - r, cy - int(r * 0.25), cx + r, cy + int(r * 0.25)), 0, 180, fill=tuple(min(255, c + 28) for c in prgb))
    d.arc(ring_box, 0, 180, fill=(242, 222, 182), width=4)
    prompt = random.choice([
        f"pianeta {random.choice(cwords)} nello spazio",
        f"saturno con anelli nello spazio profondo",
        f"galassia e pianeta {random.choice(cwords)}",
    ])
    return img, prompt


def scene_sailboat(size: int = 96) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((110, 190, 252)), _jitter((210, 238, 255)))
    d = ImageDraw.Draw(img)
    sea_y = int(size * 0.62)
    d.rectangle((0, sea_y, size, size), fill=_jitter((28, 105, 195), 14))
    bx = size // 2 + random.randint(-8, 8)
    d.polygon([(bx - 20, sea_y - 2), (bx + 20, sea_y - 2), (bx + 13, sea_y + 8), (bx - 13, sea_y + 8)], fill=(125, 65, 35))
    d.line((bx, sea_y - 2, bx, int(size * 0.22)), fill=(230, 230, 230), width=2)
    d.polygon([(bx + 1, int(size * 0.24)), (bx + 18, sea_y - 5), (bx + 1, sea_y - 5)], fill=(250, 252, 255))
    d.polygon([(bx - 1, int(size * 0.30)), (bx - 14, sea_y - 5), (bx - 1, sea_y - 5)], fill=(240, 245, 252))
    prompt = random.choice([
        "barca a vela sul mare blu",
        "veliero bianco sull'acqua",
        "barca a vela sotto il cielo azzurro",
    ])
    return img, prompt


def scene_geometric(size: int = 96) -> tuple[Image.Image, str]:
    cname = random.choice(list(PALETTE.keys()))
    srgb, cwords = PALETTE[cname]
    img = _vertical_gradient(size, _jitter((18, 22, 36), 8), _jitter((34, 40, 64), 10))
    d = ImageDraw.Draw(img)
    shape_kind = random.choice(["cerchio", "quadrato"])
    pad = int(size * 0.22) + random.randint(-4, 4)
    if shape_kind == "cerchio":
        d.ellipse((pad, pad, size - pad, size - pad), fill=_jitter(srgb, 10))
    else:
        d.rectangle((pad, pad, size - pad, size - pad), fill=_jitter(srgb, 10))
    prompt = f"un {shape_kind} {random.choice(cwords)}"
    return img, prompt


def scene_pine_forest(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((125, 195, 245), 8), _jitter((195, 228, 235), 8))
    d = ImageDraw.Draw(img)
    gy = int(size * 0.76)
    d.rectangle((0, gy, size, size), fill=_jitter((42, 118, 56), 10))
    for tx in (int(size * 0.22), int(size * 0.50), int(size * 0.78)):
        tw = max(4, int(size * 0.05))
        d.rectangle((tx - tw // 2, gy - int(size * 0.14), tx + tw // 2, gy + 2), fill=(96, 62, 36))
        for tier, (w_mul, y_top, y_bot) in enumerate([(0.22, 0.44, 0.66), (0.18, 0.30, 0.52), (0.13, 0.18, 0.38)]):
            hw = int(size * w_mul)
            gcol = _jitter((28 + tier * 10, 115 + tier * 16, 52 + tier * 8), 8)
            d.polygon([(tx - hw, int(size * y_bot)), (tx, int(size * y_top)), (tx + hw, int(size * y_bot))], fill=gcol)
    prompt = random.choice(["foresta di pini verdi", "bosco di abeti sotto il cielo azzurro", "alberi di pino nella natura"])
    return img, prompt


def scene_lighthouse(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((70, 145, 230), 8), _jitter((175, 218, 250), 8))
    d = ImageDraw.Draw(img)
    sea_y = int(size * 0.68)
    d.rectangle((0, sea_y, size, size), fill=_jitter((22, 88, 175), 10))
    # Scogliera e faro a strisce bianche e rosse
    d.polygon([(int(size * 0.18), size), (int(size * 0.30), sea_y - 4), (int(size * 0.56), sea_y - 4), (int(size * 0.66), size)], fill=(86, 82, 78))
    lx = int(size * 0.43)
    bw = int(size * 0.08)
    top_y = int(size * 0.24)
    d.polygon([(lx - bw, sea_y - 4), (lx - int(bw * 0.7), top_y), (lx + int(bw * 0.7), top_y), (lx + bw, sea_y - 4)], fill=(245, 245, 248))
    for sy in (0.34, 0.50):
        y1, y2 = int(size * sy), int(size * (sy + 0.08))
        d.rectangle((lx - int(bw * 0.82), y1, lx + int(bw * 0.82), y2), fill=(215, 48, 52))
    d.rectangle((lx - int(bw * 0.75), top_y - int(size * 0.07), lx + int(bw * 0.75), top_y), fill=(255, 232, 115))
    d.polygon([(lx - bw, top_y - int(size * 0.07)), (lx, top_y - int(size * 0.14)), (lx + bw, top_y - int(size * 0.07))], fill=(195, 42, 46))
    prompt = random.choice(["faro sul mare", "faro bianco e rosso sulla scogliera", "faro marittimo con oceano blu"])
    return img, prompt


def scene_aurora(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, (8, 14, 34), (16, 38, 58))
    d = ImageDraw.Draw(img)
    for _ in range(35):
        d.point((random.randrange(size), random.randrange(int(size * 0.7))), fill=(235, 245, 255))
    # Onde luminose dell'aurora boreale
    for band, col in [(0.26, (45, 235, 145)), (0.36, (65, 210, 195)), (0.20, (125, 95, 235))]:
        pts = []
        for x in range(0, size, 4):
            y = int(size * band + math.sin(x * 0.08) * (size * 0.07))
            pts.append((x, y))
        for x, y in pts:
            d.line((x, y - int(size * 0.09), x, y + int(size * 0.07)), fill=col, width=3)
    hy = int(size * 0.75)
    d.polygon([(0, size), (0, hy), (int(size * 0.32), int(size * 0.58)), (int(size * 0.65), hy), (int(size * 0.86), int(size * 0.62)), (size, hy), (size, size)], fill=(14, 20, 32))
    prompt = random.choice(["aurora boreale nel cielo notturno", "luci del nord verdi sulle montagne", "aurora boreale artica"])
    return img, prompt


def scene_volcano(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((36, 18, 32), 6), _jitter((98, 36, 28), 8))
    d = ImageDraw.Draw(img)
    gy = int(size * 0.82)
    cx = size // 2
    cw = int(size * 0.11)
    top_y = int(size * 0.38)
    # Pennacchio e bagliore lavico
    d.ellipse((cx - int(size * 0.18), int(size * 0.14), cx + int(size * 0.18), top_y + 10), fill=(255, 135, 42))
    d.ellipse((cx - int(size * 0.10), int(size * 0.22), cx + int(size * 0.10), top_y + 8), fill=(255, 220, 85))
    # Cono vulcanico
    d.polygon([(int(size * 0.08), gy), (cx - cw, top_y), (cx + cw, top_y), (int(size * 0.92), gy)], fill=(48, 42, 46))
    # Colate di lava
    d.line((cx - 3, top_y + 2, cx - int(size * 0.14), gy), fill=(255, 115, 32), width=4)
    d.line((cx + 3, top_y + 2, cx + int(size * 0.10), gy), fill=(255, 165, 48), width=3)
    d.rectangle((0, gy, size, size), fill=(26, 22, 26))
    prompt = random.choice(["vulcano in eruzione con lava", "eruzione vulcanica di notte", "vulcano attivo con lava arancione"])
    return img, prompt


def scene_castle(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((95, 168, 242), 8), _jitter((195, 225, 250), 8))
    d = ImageDraw.Draw(img)
    gy = int(size * 0.72)
    d.ellipse((-int(size * 0.15), gy - int(size * 0.08), int(size * 1.15), size + int(size * 0.25)), fill=(62, 148, 68))
    stone = _jitter((188, 184, 176), 8)
    stone_dark = _jitter((156, 152, 145), 8)
    d.rectangle((int(size * 0.28), int(size * 0.44), int(size * 0.72), gy), fill=stone)
    for tx1, tx2 in [(int(size * 0.20), int(size * 0.33)), (int(size * 0.67), int(size * 0.80))]:
        ty = int(size * 0.30)
        d.rectangle((tx1, ty, tx2, gy), fill=stone_dark)
        d.polygon([(tx1 - 3, ty), ((tx1 + tx2) // 2, int(size * 0.18)), (tx2 + 3, ty)], fill=(65, 92, 168))
    d.chord((int(size * 0.43), int(size * 0.56), int(size * 0.57), gy + 12), 180, 360, fill=(92, 58, 36))
    prompt = random.choice(["castello medievale sulla collina", "castello con torri sotto il cielo azzurro", "fortezza medievale nel prato"])
    return img, prompt


def scene_butterfly(size: int = 128) -> tuple[Image.Image, str]:
    cname = random.choice(["blu", "arancione", "rosa", "viola", "giallo"])
    wrgb, cwords = PALETTE[cname]
    img = _vertical_gradient(size, _jitter((62, 148, 86), 8), _jitter((34, 96, 52), 8))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    wcol = _jitter(wrgb, 10)
    # Ali superiori e inferiori simmetriche
    d.ellipse((cx - int(size * 0.34), cy - int(size * 0.28), cx - 3, cy + 4), fill=wcol)
    d.ellipse((cx + 3, cy - int(size * 0.28), cx + int(size * 0.34), cy + 4), fill=wcol)
    d.ellipse((cx - int(size * 0.26), cy - 2, cx - 3, cy + int(size * 0.26)), fill=tuple(max(0, c - 25) for c in wcol))
    d.ellipse((cx + 3, cy - 2, cx + int(size * 0.26), cy + int(size * 0.26)), fill=tuple(max(0, c - 25) for c in wcol))
    # Corpo e antenne
    d.ellipse((cx - 4, cy - int(size * 0.20), cx + 4, cy + int(size * 0.22)), fill=(28, 26, 32))
    d.line((cx - 2, cy - int(size * 0.18), cx - 12, cy - int(size * 0.30)), fill=(28, 26, 32), width=2)
    d.line((cx + 2, cy - int(size * 0.18), cx + 12, cy - int(size * 0.30)), fill=(28, 26, 32), width=2)
    prompt = random.choice([f"una farfalla {random.choice(cwords)}", f"farfalla {random.choice(cwords)} nel giardino"])
    return img, prompt


def scene_robot(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, (18, 26, 48), (32, 48, 86))
    d = ImageDraw.Draw(img)
    cx = size // 2
    metal = _jitter((182, 196, 214), 8)
    metal_dark = _jitter((128, 144, 168), 8)
    # Antenna, testa, visore luminoso e busto
    d.line((cx, int(size * 0.14), cx, int(size * 0.24)), fill=metal, width=3)
    d.ellipse((cx - 5, int(size * 0.10), cx + 5, int(size * 0.17)), fill=(75, 225, 255))
    d.rounded_rectangle((cx - int(size * 0.22), int(size * 0.24), cx + int(size * 0.22), int(size * 0.54)), radius=10, fill=metal)
    d.rounded_rectangle((cx - int(size * 0.16), int(size * 0.31), cx + int(size * 0.16), int(size * 0.43)), radius=6, fill=(22, 34, 52))
    d.ellipse((cx - int(size * 0.10), int(size * 0.34), cx - int(size * 0.03), int(size * 0.41)), fill=(75, 232, 255))
    d.ellipse((cx + int(size * 0.03), int(size * 0.34), cx + int(size * 0.10), int(size * 0.41)), fill=(75, 232, 255))
    d.rounded_rectangle((cx - int(size * 0.26), int(size * 0.57), cx + int(size * 0.26), int(size * 0.92)), radius=10, fill=metal_dark)
    d.rectangle((cx - int(size * 0.10), int(size * 0.66), cx + int(size * 0.10), int(size * 0.76)), fill=(75, 225, 255))
    prompt = random.choice(["un robot futuristico", "androide metallico con occhi azzurri luminosi", "ritratto di robot amichevole"])
    return img, prompt


def scene_beach(size: int = 128) -> tuple[Image.Image, str]:
    img = _vertical_gradient(size, _jitter((88, 188, 252), 8), _jitter((188, 238, 255), 8))
    d = ImageDraw.Draw(img)
    sea_y = int(size * 0.56)
    sand_y = int(size * 0.74)
    d.ellipse((int(size * 0.70), int(size * 0.12), int(size * 0.88), int(size * 0.30)), fill=(255, 232, 95))
    d.rectangle((0, sea_y, size, sand_y), fill=_jitter((24, 178, 198), 10))
    d.polygon([(0, sand_y), (size, sand_y - 6), (size, size), (0, size)], fill=_jitter((242, 214, 148), 10))
    # Palma
    px, py = int(size * 0.26), int(size * 0.82)
    tx, ty = int(size * 0.34), int(size * 0.36)
    d.line((px, py, tx, ty), fill=(132, 84, 44), width=5)
    for dx, dy in [(-24, -4), (-16, 12), (22, -6), (20, 10), (0, -18)]:
        d.line((tx, ty, tx + dx, ty + dy), fill=(38, 158, 64), width=4)
    prompt = random.choice(["spiaggia tropicale con palma e mare turchese", "isola estiva con sabbia dorata e palma", "mare cristallino e spiaggia"])
    return img, prompt


SCENE_GENERATORS = [
    scene_cat,
    scene_sunset_sea,
    scene_starry_night,
    scene_mountains,
    scene_tree,
    scene_city_night,
    scene_flower,
    scene_house,
    scene_heart,
    scene_planet_space,
    scene_sailboat,
    scene_geometric,
    scene_pine_forest,
    scene_lighthouse,
    scene_aurora,
    scene_volcano,
    scene_castle,
    scene_butterfly,
    scene_robot,
    scene_beach,
]


def load_real_image_pairs(real_dir: str = "data/immagini_reali") -> list[tuple[Path, str]]:
    folder = Path(real_dir)
    if not folder.exists():
        return []
    pairs = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            txt = p.with_suffix(".txt")
            if txt.exists():
                cap = txt.read_text(encoding="utf-8", errors="ignore").strip()
            else:
                cap = p.stem.replace("_", " ").replace("-", " ").strip()
            if cap:
                pairs.append((p, cap))
    return pairs


def _load_and_crop_real(path: Path, size: int) -> Image.Image:
    with Image.open(path) as raw:
        im = raw.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return im.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def sample(size: int = 64) -> torch.Tensor:
    img, _ = random.choice(SCENE_GENERATORS)(size)
    return _img_to_tensor(img)


def sample_orange_cat(size: int = 64) -> torch.Tensor:
    img, _ = scene_cat(size)
    return _img_to_tensor(img)


def sample_batch(
    batch_size: int,
    size: int = 128,
    vocab_size: int = 32000,
    cfg_drop: float = 0.10,
    real_pairs: list[tuple[Path, str]] | None = None,
):
    imgs = []
    toks = []
    for _ in range(batch_size):
        if real_pairs and random.random() < 0.50:
            rpath, prompt = random.choice(real_pairs)
            try:
                img = _load_and_crop_real(rpath, size)
            except Exception:
                img, prompt = random.choice(SCENE_GENERATORS)(size)
        else:
            fn = random.choice(SCENE_GENERATORS)
            img, prompt = fn(size)
        imgs.append(_img_to_tensor(img))
        if random.random() < cfg_drop:
            toks.append(torch.zeros((1, 24), dtype=torch.long))
        else:
            toks.append(encode_prompt(prompt, vocab_size=vocab_size))
    return torch.stack(imgs), torch.cat(toks, dim=0)


def train(
    steps: int = 1600,
    batch_size: int = 8,
    size: int = 96,
    lr: float = 3e-4,
    out: str = "checkpoints_lumina_image/lumina_image_v1.pt",
    cats: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Avvio addestramento Lumina Image Rectified-Flow su {device} ({steps} step, {size}x{size})...", flush=True)
    model = create_model(device=device, vocab_size=32000, channels=64)
    out_path = Path(out)
    if out_path.exists():
        try:
            ckpt = torch.load(out_path, map_location=device)
            state = ckpt.get("ema_state_dict") or ckpt.get("state_dict") or ckpt
            model.load_state_dict(state, strict=False)
            print(f"[+] Ripreso checkpoint esistente da {out_path}", flush=True)
        except Exception:
            pass

    ema_model = copy.deepcopy(model).eval()
    real_pairs = load_real_image_pairs("data/immagini_reali")
    if real_pairs:
        print(f"[+] Trovate {len(real_pairs)} immagini reali locali in data/immagini_reali/ (mix 50/50 attivo)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.1)

    t0 = time.time()
    running_loss = 0.0
    model.train()

    for i in range(1, steps + 1):
        if cats:
            clean_list, tok_list = [], []
            for _ in range(batch_size):
                im, pr = scene_cat(size)
                clean_list.append(_img_to_tensor(im))
                tok_list.append(encode_prompt(pr, vocab_size=model.vocab_size))
            x_0 = torch.stack(clean_list).to(device)
            token_ids = torch.cat(tok_list, dim=0).to(device)
        else:
            x_0, token_ids = sample_batch(batch_size, size=size, vocab_size=model.vocab_size, real_pairs=real_pairs)
            x_0, token_ids = x_0.to(device), token_ids.to(device)

        # Rectified Flow: x_t = (1 - t) * x_0 + t * z, target velocity v = z - x_0
        t = torch.rand((batch_size,), device=device)
        t_view = t.view(batch_size, 1, 1, 1)
        z = torch.randn_like(x_0)
        x_t = (1.0 - t_view) * x_0 + t_view * z
        target_v = z - x_0
        step_idx = (t * 999.0).long().clamp(0, 999)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred_v = model(x_t, token_ids, step_idx)
            loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            decay = 0.992
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

        running_loss += float(loss.item())
        if i % 100 == 0 or i == steps:
            avg_l = running_loss / (100 if i % 100 == 0 else (i % 100))
            running_loss = 0.0
            print(f"[Lumina Image Step {i:4d}/{steps}] RF-Loss: {avg_l:.4f} | Tempo: {time.time() - t0:.1f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "vocab_size": model.vocab_size,
        "channels": model.channels,
        "size": size,
        "steps": steps,
    }
    torch.save(payload, out_path)
    # Compatibilità con eventuali riferimenti legacy
    torch.save(ema_model.state_dict(), out_path.parent / "synthetic.pt")
    print(f"[+] Checkpoint Lumina Image salvato in: {out_path}", flush=True)

    # Genera campioni di validazione reali da puro rumore gaussiano
    samples_dir = out_path.parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    demo_prompts = [
        ("gatto_arancione.png", "un gatto arancione nel prato"),
        ("tramonto_mare.png", "tramonto sul mare"),
        ("faro_mare.png", "faro sul mare"),
        ("aurora_boreale.png", "aurora boreale nel cielo notturno"),
        ("citta_notte.png", "città di notte illuminata"),
        ("castello_collina.png", "castello medievale sulla collina"),
        ("pianeta_spazio.png", "pianeta blu nello spazio"),
        ("vulcano_lava.png", "vulcano in eruzione con lava"),
        ("casa_tetto_rosso.png", "una casa con il tetto rosso"),
        ("spiaggia_palma.png", "spiaggia tropicale con palma e mare turchese"),
        ("robot_futuro.png", "un robot futuristico"),
        ("farfalla_blu.png", "una farfalla blu"),
    ]
    tiles = []
    for idx_p, (fname, pr) in enumerate(demo_prompts):
        rgb = generate_rectified_flow(ema_model, pr, steps=36, size=size, cfg_scale=1.45, seed=42 + idx_p)
        arr = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        im = Image.fromarray(arr).resize((384, 384), Image.Resampling.LANCZOS)
        im.save(samples_dir / fname)
        tiles.append(im.resize((192, 192), Image.Resampling.LANCZOS))

    rows_g = (len(tiles) + 3) // 4
    grid = Image.new("RGB", (192 * 4, 192 * rows_g))
    for idx, tile in enumerate(tiles):
        grid.paste(tile, ((idx % 4) * 192, (idx // 4) * 192))
    grid.save(samples_dir / "griglia_validazione_lumina_image.png")
    print(f"[+] Campioni e griglia salvati in: {samples_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints_lumina_image/lumina_image_v1.pt")
    args = ap.parse_args()
    train(steps=args.steps, batch_size=args.batch_size, size=args.size, lr=args.lr, out=args.out)
