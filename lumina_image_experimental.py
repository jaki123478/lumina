"""Modello nativo proprietario Lumina Image (Conditional Rectified-Flow U-Net).

Architettura U-Net multiscala condizionata dal testo in italiano tramite
tokenizer BPE di Lumina (32.000 token), con blocchi residui FiLM, attenzione
spaziale al bottleneck e campionamento Rectified Flow + Classifier-Free Guidance.
100% locale, non scarica pesi esterni.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
_BPE_TOK = None


def encode_prompt(prompt: str, max_len: int = 24, vocab_size: int = 32000) -> torch.Tensor:
    """Codifica un prompt testuale italiano in ID token (1, max_len)."""
    text = " ".join(str(prompt or "").strip().lower().split())
    if not text:
        return torch.zeros((1, max_len), dtype=torch.long)

    global _BPE_TOK
    if _BPE_TOK is None:
        vocab = ROOT / "tokenizer_lumina_next" / "vocab.json"
        merges = ROOT / "tokenizer_lumina_next" / "merges.txt"
        if vocab.exists() and merges.exists():
            try:
                from tokenizers import ByteLevelBPETokenizer
                _BPE_TOK = ByteLevelBPETokenizer(str(vocab), str(merges))
            except Exception:
                _BPE_TOK = False
        else:
            _BPE_TOK = False

    if _BPE_TOK:
        ids = [i % vocab_size for i in _BPE_TOK.encode(text).ids[:max_len]]
    else:
        ids = [
            (int(hashlib.blake2b(w.encode("utf-8"), digest_size=4).hexdigest(), 16) % (vocab_size - 1)) + 1
            for w in text.split()[:max_len]
        ]

    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor([ids[:max_len]], dtype=torch.long)


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=step.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = step.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        groups_in = min(8, in_ch) if in_ch % min(8, in_ch) == 0 else 1
        groups_out = min(8, out_ch) if out_ch % min(8, out_ch) == 0 else 1
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(cond)).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(normed, normed, normed, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class LuminaImageDenoiser(nn.Module):
    """U-Net condizionata da testo italiano e step temporale (Rectified Flow)."""

    def __init__(self, vocab_size: int = 32000, text_dim: int = 256, channels: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.channels = channels
        cond_dim = text_dim

        self.text = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.text_attn = nn.MultiheadAttention(text_dim, num_heads=4, batch_first=True)
        self.text_mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.step_mlp = nn.Sequential(
            SinusoidalStepEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        c1, c2, c3 = channels, channels * 2, channels * 4
        self.in_proj = nn.Conv2d(3, c1, 3, padding=1)

        self.down1 = FiLMResBlock(c1, c1, cond_dim)
        self.pool1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)

        self.down2 = FiLMResBlock(c2, c2, cond_dim)
        self.pool2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        self.mid1 = FiLMResBlock(c3, c3, cond_dim)
        self.mid_attn = SpatialSelfAttention(c3, heads=4)
        self.mid2 = FiLMResBlock(c3, c3, cond_dim)

        self.up2 = FiLMResBlock(c3 + c2, c2, cond_dim)
        self.up1 = FiLMResBlock(c2 + c1, c1, cond_dim)

        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_proj = nn.Conv2d(c1, 3, 3, padding=1)

    def _encode_condition(self, token_ids: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        tok_ids = token_ids.clamp(0, self.vocab_size - 1)
        emb = self.text(tok_ids)
        attn_out, _ = self.text_attn(emb, emb, emb, need_weights=False)
        mask = (tok_ids != 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = ((emb + attn_out) * mask).sum(dim=1) / denom
        has_tokens = (tok_ids.abs().sum(dim=1, keepdim=True) > 0).float()
        pooled = pooled * has_tokens
        return self.text_mlp(pooled) + self.step_mlp(step)

    def forward(self, noisy: torch.Tensor, token_ids: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        cond = self._encode_condition(token_ids, step)
        h0 = self.in_proj(noisy)

        s1 = self.down1(h0, cond)
        d1 = self.pool1(s1)

        s2 = self.down2(d1, cond)
        d2 = self.pool2(s2)

        m = self.mid1(d2, cond)
        m = self.mid_attn(m)
        m = self.mid2(m, cond)

        u2 = F.interpolate(m, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, s2], dim=1), cond)

        u1 = F.interpolate(u2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, s1], dim=1), cond)

        return self.out_proj(F.silu(self.out_norm(u1)))


def create_model(device: str | None = None, vocab_size: int = 32000, channels: int = 64) -> LuminaImageDenoiser:
    model = LuminaImageDenoiser(vocab_size=vocab_size, channels=channels)
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def generate_rectified_flow(
    model: LuminaImageDenoiser,
    prompt: str,
    steps: int = 36,
    size: int = 96,
    cfg_scale: float = 1.45,
    seed: int | None = None,
) -> torch.Tensor:
    """Genera un tensore immagine RGB in [0, 1] tramite integrazione ODE Rectified Flow."""
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))
    else:
        gen.seed()

    x = torch.randn((1, 3, size, size), generator=gen, device=device)
    cond_ids = encode_prompt(prompt, vocab_size=model.vocab_size).to(device)
    uncond_ids = torch.zeros_like(cond_ids)

    n_steps = max(4, min(int(steps), 80))
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for i in range(n_steps):
        t_cur = ts[i]
        dt = ts[i] - ts[i + 1]
        step_idx = torch.full((1,), int(round(float(t_cur) * 999)), dtype=torch.long, device=device)
        if cfg_scale > 1.0:
            x_in = torch.cat([x, x], dim=0)
            tok_in = torch.cat([uncond_ids, cond_ids], dim=0)
            step_in = torch.cat([step_idx, step_idx], dim=0)
            v_uncond, v_cond = model(x_in, tok_in, step_in).chunk(2, dim=0)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, cond_ids, step_idx)
        x = x - dt * v

    return ((x.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()
