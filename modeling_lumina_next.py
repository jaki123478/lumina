"""
modeling_lumina_next.py - Architettura Lumina-Next 135M.
Caratteristiche:
- Grouped Query Attention (GQA 3:1)
- Rotary Position Embeddings (RoPE) con supporto a YaRN (8k -> 32k)
- Normalizzazione RMSNorm ad alta velocità
- Attivazioni SwiGLU (Standard Llama 3 / Mistral)
- Scaled Dot-Product Attention nativa (FlashAttention / SDPA)
- Supporto a Gradient Checkpointing per training su 8GB VRAM
- Piena compatibilità con Hugging Face Transformers
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope_yarn import LuminaRotaryEmbedding, apply_rotary_pos_emb

DEFAULT_CONFIG_135M = {
    "architectures": ["LuminaNextForCausalLM"],
    "model_type": "lumina_next",
    "vocab_size": 32000,
    "d_model": 576,
    "n_layers": 24,
    "n_heads": 9,
    "n_kv_heads": 3,
    "head_dim": 64,
    "d_ff": 1536,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
    "rope_scaling_type": "yarn",
    "rope_scaling_factor": 1.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LuminaGQAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Gestione KV Cache per inferenza veloce
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        current_kv = (k, v) if use_cache else None

        # Espansione GQA per le teste Query
        if self.num_kv_groups > 1:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_expanded, v_expanded = k, v

        # FlashAttention / SDPA nativo
        # Se siamo in decoding step (S==1 con cache), non serve maschera causale triangolare
        is_causal = (S > 1 and past_kv is None)
        out = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), current_kv


class LuminaNextDecoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(config["d_model"], eps=config.get("rms_norm_eps", 1e-6))
        self.self_attn = LuminaGQAAttention(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            head_dim=config["head_dim"],
        )
        self.post_attention_layernorm = RMSNorm(config["d_model"], eps=config.get("rms_norm_eps", 1e-6))
        self.mlp = SwiGLUMLP(config["d_model"], config["d_ff"])

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x), cos, sin, past_kv=past_kv, use_cache=use_cache
        )
        x = residual + attn_out

        residual = x
        mlp_out = self.mlp(self.post_attention_layernorm(x))
        x = residual + mlp_out
        return x, new_kv


class LuminaNextForCausalLM(nn.Module):
    def __init__(self, config: dict = DEFAULT_CONFIG_135M):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(self.vocab_size, config["d_model"])
        self.rotary_emb = LuminaRotaryEmbedding(
            dim=config["head_dim"],
            max_position_embeddings=config["max_position_embeddings"],
            base=config.get("rope_theta", 500000.0),
            scaling_factor=config.get("rope_scaling_factor", 1.0),
            scaling_type=config.get("rope_scaling_type", "yarn"),
        )
        self.layers = nn.ModuleList([LuminaNextDecoderLayer(config) for _ in range(config["n_layers"])])
        self.norm = RMSNorm(config["d_model"], eps=config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(config["d_model"], self.vocab_size, bias=False)

        # Weight tying
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def count_parameters(self) -> int:
        """Calcola il numero esatto di parametri del modello."""
        return sum(p.numel() for p in self.parameters())

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)

        offset = 0
        if past_key_values is not None and past_key_values[0] is not None:
            offset = past_key_values[0][0].shape[2]

        total_seq_len = offset + S
        cos, sin = self.rotary_emb(total_seq_len, x.device, dtype=x.dtype)
        cos = cos[offset:]
        sin = sin[offset:]

        new_kvs = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            p_kv = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                # Custom forward wrapper for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                x, layer_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, cos, sin, p_kv, use_cache,
                    use_reentrant=False
                )
            else:
                x, layer_kv = layer(x, cos, sin, past_kv=p_kv, use_cache=use_cache)

            if use_cache:
                new_kvs.append(layer_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kvs
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Generazione autoregressiva accelerata con KV Cache."""
        self.eval()
        device = input_ids.device
        curr_ids = input_ids
        past_kv = None

        for _ in range(max_new_tokens):
            if past_kv is None:
                step_input = curr_ids
            else:
                step_input = curr_ids[:, -1:]

            logits, past_kv = self.forward(step_input, past_key_values=past_kv, use_cache=True)
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / max(temperature, 1e-5)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break

        return curr_ids

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Salva config.json
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        # Salva pesi PyTorch
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path]):
        save_dir = Path(save_directory)
        with open(save_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(config)
        safetensors_path = save_dir / "model.safetensors"
        weights_path = save_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict)
                return model
            except Exception:
                pass
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        return model
