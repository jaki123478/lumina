"""
rope_yarn.py - Rotary Position Embeddings (RoPE) con YaRN e NTK-aware Scaling per Lumina-Next.
Supporta contesti nativi da 8.192 token ed estensioni dinamiche fino a 32.768 token.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LuminaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 500000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "yarn",  # "yarn", "ntk", "none"
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        self.scaling_factor = float(scaling_factor)
        self.scaling_type = scaling_type
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        pos = torch.arange(0, self.dim, 2).float()
        inv_freq = 1.0 / (self.base ** (pos / self.dim))

        if self.scaling_factor <= 1.0 or self.scaling_type == "none":
            return inv_freq

        if self.scaling_type == "ntk":
            # Dynamic NTK-aware scaling
            base_scaled = self.base * (
                (self.scaling_factor * self.max_position_embeddings / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            return 1.0 / (base_scaled ** (pos / self.dim))

        elif self.scaling_type == "yarn":
            # YaRN: Yet another RoPE extensioN method (Peng et al., 2023)
            # Interpola frequenze alte e basse per mantenere invariata la risoluzione locale
            low_freq_wavelen = self.max_position_embeddings / self.beta_slow
            high_freq_wavelen = self.max_position_embeddings / self.beta_fast
            wavelens = 2.0 * math.pi / inv_freq

            smooth = torch.clamp(
                (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen),
                0.0,
                1.0,
            )
            # Blend tra frequenze non scalate e frequenze scalate
            inv_freq_yarn = (1.0 - smooth) * inv_freq + smooth * (inv_freq / self.scaling_factor)
            return inv_freq_yarn

        return inv_freq

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Ruota a metà l'ultima dimensione del tensore per applicare la matrice RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applica RoPE a Query e Key:
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    seq_len = q.shape[2]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
