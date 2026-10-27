"""Lumina 50K: a byte-level causal Transformer, initialized from scratch.

All 50,000 parameters participate in the forward pass. Output projection shares
the token embedding; no pretrained model, tokenizer, or remote API is used.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from prompt_format import build_prompt


CONFIG = {
    "vocab_size": 259,
    "context_length": 165,
    "d_model": 40,
    "n_layers": 2,
    "n_heads": 4,
    "ff_dim": 120,
    "bos_id": 256,
    "eos_id": 257,
    "pad_id": 258,
    "layer_norm_eps": 1e-5,
    "parameter_count": 50000,
}


def make_config(d_model: int, n_layers: int, n_heads: int, ff_dim: int, context_length: int = 165) -> dict:
    """Architecture with the same tokenizer and tied output embedding as Lumina 50K."""
    vocab = 259
    block = 4 * d_model + (3 * d_model * d_model + 3 * d_model) + (d_model * d_model + d_model) \
        + (d_model * ff_dim + ff_dim) + (ff_dim * d_model + d_model)
    count = vocab * d_model + context_length * d_model + n_layers * block + 2 * d_model
    return {**CONFIG, "context_length": context_length, "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "ff_dim": ff_dim, "parameter_count": count}


# Presets:
# - "50k": original shipped 50,000 parameter educational model.
# - "large": ~2.75M parameter model for fast experiments.
# - "xlarge": ~14.4M parameter model for pretraining on Italian corpus.
PRESETS = {
    "50k": CONFIG,
    "large": make_config(192, 6, 6, 768, context_length=165),
    "xlarge": make_config(384, 8, 6, 1536, context_length=256),
}


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")


def prompt_tokens(message: str, previous: tuple[str, str] | None = None) -> list[int]:
    # `previous` is the last completed (question, answer) exchange; see prompt_format.py.
    return build_prompt(message, previous)


class CausalAttention(nn.Module):
    def __init__(self, config: dict = CONFIG) -> None:
        super().__init__()
        self.heads = config["n_heads"]
        d = config["d_model"]
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        heads = self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (item.view(b, t, heads, d // heads).transpose(1, 2) for item in (q, k, v))
        # SDPA is mathematically identical to softmax(QK^T/sqrt(head_dim))V.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(attended.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, dropout: float = 0.0, config: dict = CONFIG) -> None:
        super().__init__()
        # Regularization only: dropout adds no parameters and is inactive in eval mode.
        self.dropout = dropout
        d, f = config["d_model"], config["ff_dim"]
        self.ln1 = nn.LayerNorm(d, eps=config["layer_norm_eps"])
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(d, eps=config["layer_norm_eps"])
        self.fc1 = nn.Linear(d, f)
        self.fc2 = nn.Linear(f, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.attn(self.ln1(x)), self.dropout, self.training)
        hidden = self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x + F.dropout(hidden, self.dropout, self.training)


class Lumina50K(nn.Module):
    def __init__(self, dropout: float = 0.0, config: dict = CONFIG) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_embedding = nn.Embedding(config["context_length"], config["d_model"])
        self.blocks = nn.ModuleList(Block(dropout, config) for _ in range(config["n_layers"]))
        self.final_norm = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.apply(self._initialize)
        actual = sum(p.numel() for p in self.parameters())
        if actual != config["parameter_count"]:
            raise RuntimeError(f"Expected exactly {config['parameter_count']:,} parameters; found {actual}")

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config["context_length"]:
            raise ValueError(f"Context exceeds {self.config['context_length']} bytes")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.final_norm(x), self.token_embedding.weight)

    @torch.inference_mode()
    def generate(self, message: str, max_new_tokens: int = 160, temperature: float = 0.0,
                 previous: tuple[str, str] | None = None) -> str:
        self.eval()
        tokens = prompt_tokens(message, previous)
        answer: list[int] = []
        device = self.token_embedding.weight.device
        for _ in range(max_new_tokens):
            chunk = torch.tensor([tokens[-self.config["context_length"]:]], dtype=torch.long, device=device)
            logits = self(chunk)[0, -1].clone()
            logits[self.config["bos_id"]] = float("-inf")
            logits[self.config["pad_id"]] = float("-inf")
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                token = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
            if token == self.config["eos_id"]:
                break
            tokens.append(token)
            answer.append(token)
        return decode(answer).strip()


def load_model(path: str | Path = "model.pt", dropout: float = 0.0) -> tuple[Lumina50K, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config != CONFIG and config not in PRESETS.values():
        raise ValueError("Checkpoint configuration is not a known Lumina preset")
    model = Lumina50K(dropout, config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint.get("training", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the locally trained Lumina 50K model.")
    parser.add_argument("message", nargs="?", default="Ciao")
    parser.add_argument("--checkpoint", default="model.pt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = load_model(args.checkpoint)
    print(model.generate(args.message))
