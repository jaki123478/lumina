"""
export_gguf.py - Esporta i modelli Lumina in formato standard GGUF.

Compatibile con LM Studio, llama.cpp, Ollama e gli strumenti di inferenza open-source.

Uso:
    python export_gguf.py --checkpoint model-large.pt --output lumina-2.75m.gguf
    python export_gguf.py --checkpoint model-xlarge.pt --output lumina-14m.gguf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

def export_to_gguf(checkpoint_path: Path, output_path: Path, model_name: str = "Lumina"):
    print(f"Caricamento checkpoint: {checkpoint_path}...")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = cp["config"]
    state_dict = cp["state_dict"]

    print(f"Configurazione: {config}")
    print(f"Esportazione GGUF -> {output_path}...")

    writer = GGUFWriter(str(output_path), "lumina")
    writer.add_name(model_name)
    writer.add_description(f"{model_name}: Modello linguistico italiano compatto ed efficiente.")
    writer.add_context_length(config["context_length"])
    writer.add_embedding_length(config["d_model"])
    writer.add_block_count(config["n_layers"])
    writer.add_feed_forward_length(config["ff_dim"])
    writer.add_head_count(config["n_heads"])
    writer.add_layer_norm_eps(config.get("layer_norm_eps", 1e-5))

    # Tokenizer a livello byte
    tokens = [bytes([i]).decode("latin1") for i in range(256)] + ["<|bos|>", "<|eos|>", "<|pad|>"]
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * len(tokens))
    writer.add_bos_token_id(config.get("bos_id", 256))
    writer.add_eos_token_id(config.get("eos_id", 257))
    writer.add_pad_token_id(config.get("pad_id", 258))

    # Aggiunta tensori
    print(f"Scrittura di {len(state_dict)} tensori...")
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().to(torch.float32).numpy()
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"GGUF creato con successo: {output_path} ({size_mb:.2f} MB)\n")


def main():
    parser = argparse.ArgumentParser(description="Esporta Lumina in formato GGUF")
    parser.add_argument("--checkpoint", type=Path, default=Path("model-large.pt"), help="Percorso del checkpoint .pt")
    parser.add_argument("--output", type=Path, default=Path("lumina-2.75m.gguf"), help="File .gguf di destinazione")
    parser.add_argument("--name", type=str, default="Lumina-2.75M", help="Nome del modello")
    args = parser.parse_args()

    export_to_gguf(args.checkpoint, args.output, args.name)


if __name__ == "__main__":
    main()
