"""Prototipo proprietario Lumina Image.

È una base di ricerca da addestrare, non un generatore pronto: non scarica
pesi esterni e non viene usata dalla chat finché non supera una validazione.
"""
from __future__ import annotations

import torch
from torch import nn


class LuminaImageDenoiser(nn.Module):
    """Denoiser convoluzionale condizionato da un embedding testuale semplice."""

    def __init__(self, vocab_size: int = 4096, text_dim: int = 256, channels: int = 64):
        super().__init__()
        self.text = nn.Embedding(vocab_size, text_dim)
        self.time = nn.Embedding(1000, channels)
        self.in_proj = nn.Conv2d(3, channels, 3, padding=1)
        self.mid = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.out_proj = nn.Conv2d(channels, 3, 3, padding=1)
        self.text_proj = nn.Linear(text_dim, channels)

    def forward(self, noisy: torch.Tensor, token_ids: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(noisy)
        cond = self.text_proj(self.text(token_ids).mean(dim=1)) + self.time(step)
        h = h + cond[:, :, None, None]
        return self.out_proj(self.mid(h))


def create_model(device: str | None = None) -> LuminaImageDenoiser:
    model = LuminaImageDenoiser()
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

