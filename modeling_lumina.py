"""Lumina 2.75M: A compact byte-level causal Transformer in Italian.

Designed for fast, lightweight local and in-browser inference without external dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")


def prompt_tokens(message: str, previous: tuple[str, str] | None = None, context_length: int = 165) -> list[int]:
    user = encode(message.strip())[-(context_length - 8):]
    base = [256] + encode("U: ") + user + encode("\nL:")
    if previous:
        q, a = " ".join(previous[0].split()), " ".join(previous[1].split())
        if q and a:
            history = encode(f"U: {q}\nL: {a}\n")
            if 1 + len(history) + len(base) - 1 <= 84:
                return [256] + history + base[1:]
    return base


class CausalAttention(nn.Module):
    def __init__(self, config: dict) -> None:
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
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(attended.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, config: dict, dropout: float = 0.0) -> None:
        super().__init__()
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


class LuminaForCausalLM(nn.Module):
    def __init__(self, config: dict, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_embedding = nn.Embedding(config["context_length"], config["d_model"])
        self.blocks = nn.ModuleList(Block(config, dropout) for _ in range(config["n_layers"]))
        self.final_norm = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config["context_length"]:
            raise ValueError(f"Context exceeds {self.config['context_length']} bytes")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.final_norm(x), self.token_embedding.weight)

    @classmethod
    def from_pretrained(cls, pretrained_path: str | Path = ".") -> LuminaForCausalLM:
        folder = Path(pretrained_path)
        cfg_path = folder / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in {folder}")
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        model = cls(config)

        safetensors_path = folder / "model.safetensors"
        pt_path = folder / "model.pt"

        if safetensors_path.exists() and HAS_SAFETENSORS:
            state_dict = load_safetensors(str(safetensors_path))
        elif pt_path.exists():
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(f"Neither model.safetensors nor model.pt found in {folder}")

        model.load_state_dict(state_dict)
        return model.eval()

    @torch.inference_mode()
    def generate(self, message: str, max_new_tokens: int = 120, temperature: float = 0.0,
                 previous: tuple[str, str] | None = None) -> str:
        self.eval()
        tokens = prompt_tokens(message, previous, self.config["context_length"])
        answer: list[int] = []
        device = self.token_embedding.weight.device
        for _ in range(max_new_tokens):
            chunk = torch.tensor([tokens[-self.config["context_length"]:]], dtype=torch.long, device=device)
            logits = self(chunk)[0, -1].clone()
            logits[self.config["bos_id"]] = float("-inf")
            logits[self.config["pad_id"]] = float("-inf")
            if temperature <= 0.01:
                token = int(logits.argmax())
            else:
                token = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
            if token == self.config["eos_id"]:
                break
            tokens.append(token)
            answer.append(token)
        return decode(answer).strip()
