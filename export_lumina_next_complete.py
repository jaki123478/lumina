"""
export_lumina_next_complete.py - Esportatore Completo per Lumina-Next 135M.
Genera formati:
1. ONNX (FP32/FP16) con dynamic axes per inferenza ultra-rapida CPU/DirectML.
2. Safetensors per Hugging Face Hub.
3. GGUF per LM Studio / llama.cpp.
4. Pacchetto di rilascio distribuito su Syncthing in test/lenovo_releases/ per nodi AMD e Intel.
"""

import os
import sys
import json
import shutil
from pathlib import Path

import torch
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modeling_lumina_next import LuminaNextForCausalLM, DEFAULT_CONFIG_135M


class LuminaNextOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # forward standard senza cache
        logits = self.model(input_ids, use_cache=False)
        return logits


def export_onnx(model, output_path: str):
    print(f"\n[*] Esportazione ONNX verso '{output_path}'...")
    wrapper = LuminaNextOnnxWrapper(model).eval().cpu().float()
    dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        dynamo=False,
    )
    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] ONNX esportato con successo: {sz:.2f} MB ({output_path})")


def export_safetensors(model, hf_dir: Path):
    from safetensors.torch import save_model
    print(f"\n[*] Conversione pesi in safetensors per Hugging Face Hub...")
    st_path = hf_dir / "model.safetensors"
    save_model(model, str(st_path))
    sz = os.path.getsize(st_path) / (1024 * 1024)
    print(f"[+] Safetensors creato con successo: {sz:.2f} MB ({st_path})")


def export_gguf(checkpoint_path: Path, output_path: Path, model_name: str = "Lumina-Next-135M"):
    try:
        from gguf import GGUFWriter
        print(f"\n[*] Esportazione GGUF verso '{output_path}'...")
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = cp.get("config", DEFAULT_CONFIG_135M)
        state_dict = cp.get("state_dict", cp)

        writer = GGUFWriter(str(output_path), "lumina_next")
        writer.add_name(model_name)
        writer.add_description(f"{model_name}: Modello compatto con RoPE, YaRN e SwiGLU")
        writer.add_context_length(config.get("max_position_embeddings", 8192))
        writer.add_embedding_length(config.get("d_model", 576))
        writer.add_block_count(config.get("n_layers", 24))
        writer.add_feed_forward_length(config.get("d_ff", 1536))
        writer.add_head_count(config.get("n_heads", 9))
        writer.add_head_count_kv(config.get("n_kv_heads", 3))

        tok_dir = Path("tokenizer_lumina_next")
        if (tok_dir / "vocab.json").exists():
            with open(tok_dir / "vocab.json", "r", encoding="utf-8") as f:
                vocab_dict = json.load(f)
            tokens = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]
            writer.add_tokenizer_model("gpt2")
            writer.add_token_list(tokens)
            writer.add_token_types([1] * len(tokens))

        for name, tensor in state_dict.items():
            arr = tensor.detach().cpu().to(torch.float32).numpy()
            writer.add_tensor(name, arr)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        sz = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[+] GGUF creato con successo: {sz:.2f} MB ({output_path})")
    except Exception as e:
        print(f"[!] Avviso GGUF: {e}")


def main():
    print("=" * 75)
    print("      🚀 LUMINA-NEXT 135M: PIPELINE DI ESPORTAZIONE COMPLETA")
    print("=" * 75)

    ckpt_dir = Path("checkpoints_lumina_next")
    hf_dir = Path("hf_export_lumina_next")
    hf_dir.mkdir(parents=True, exist_ok=True)
    syncthing_releases = Path("C:/Users/Jaki1/Desktop/test/lenovo_releases")
    syncthing_releases.mkdir(parents=True, exist_ok=True)

    # 1. Carica Modello
    print("[*] Caricamento modello da checkpoints_lumina_next...")
    model = LuminaNextForCausalLM.from_pretrained(ckpt_dir).eval().cpu()

    # 2. Esporta ONNX
    onnx_path = "lumina_next_135m.onnx"
    export_onnx(model, onnx_path)

    # 3. Esporta Safetensors in hf_export_lumina_next
    export_safetensors(model, hf_dir)

    # Copia altri file essenziali in hf_export_lumina_next
    shutil.copy("checkpoints_lumina_next/pytorch_model.bin", hf_dir / "pytorch_model.bin")
    shutil.copy("checkpoints_lumina_next/config.json", hf_dir / "config.json")
    if Path("tokenizer_lumina_next/vocab.json").exists():
        shutil.copy("tokenizer_lumina_next/vocab.json", hf_dir / "vocab.json")
        shutil.copy("tokenizer_lumina_next/merges.txt", hf_dir / "merges.txt")
    shutil.copy("modeling_lumina_next.py", hf_dir / "modeling_lumina_next.py")
    shutil.copy("rope_yarn.py", hf_dir / "rope_yarn.py")

    # 4. Esporta GGUF
    latest_pt = list(ckpt_dir.glob("checkpoint_step*.pt"))
    if latest_pt:
        latest_ckpt = max(latest_pt, key=lambda p: int(p.stem.replace("checkpoint_step", "").replace("_safe_backup", "")))
        print(f"[*] Checkpoint per GGUF: {latest_ckpt}")
        export_gguf(latest_ckpt, Path("lumina_next_135m.gguf"))
    else:
        latest_ckpt = ckpt_dir / "pytorch_model.bin"

    # 5. Distribuisci su Syncthing (lenovo_releases/) per AMD e Intel
    print(f"\n[*] Distribuzione file di rilascio su Syncthing -> {syncthing_releases}...")
    if os.path.exists(onnx_path):
        shutil.copy(onnx_path, syncthing_releases / onnx_path)
    if os.path.exists("lumina_next_135m.gguf"):
        shutil.copy("lumina_next_135m.gguf", syncthing_releases / "lumina_next_135m.gguf")
    shutil.copy(hf_dir / "model.safetensors", syncthing_releases / "model.safetensors")
    shutil.copy(hf_dir / "config.json", syncthing_releases / "config.json")
    if (hf_dir / "vocab.json").exists():
        shutil.copy(hf_dir / "vocab.json", syncthing_releases / "vocab.json")
        shutil.copy(hf_dir / "merges.txt", syncthing_releases / "merges.txt")

    if latest_pt:
        shutil.copy(latest_ckpt, syncthing_releases / latest_ckpt.name)

    # Scrivi release notes
    step_num = "8000"
    if latest_pt:
        step_num = latest_ckpt.stem.replace("checkpoint_step", "").replace("_safe_backup", "")

    release_notes = f"""# Lumina-Next 135M — Release Checkpoint Step {step_num}
- Data rilascio: 27 Ottobre 2026
- Architettura: Lumina-Next 135M (103.4M parametri con weight tying)
- Checkpoint: {latest_ckpt.name}
- Formati disponibili:
  - `model.safetensors` (Hugging Face / PyTorch nativo)
  - `lumina_next_135m.onnx` (Inferenza CPU / DirectML per AMD & Intel)
  - `lumina_next_135m.gguf` (LM Studio / llama.cpp)
  - `vocab.json` + `merges.txt` (Tokenizer BPE 32.000 vocab)
- Addestramento: {int(step_num):,} step totali su NVIDIA RTX 5060 Laptop GPU
- Dataset: 17,057 dialoghi distillati, curati e arricchiti con CoT Scratchpad (<think>...</think>)
- Capacità di ragionamento: decomposizione aritmetica, logica temporale sequenziale, regole grammaticali ausiliari, onestà epistemica
"""
    (syncthing_releases / f"RELEASE_NOTES_STEP{step_num}.md").write_text(release_notes, encoding="utf-8")
    (syncthing_releases / "RELEASE_NOTES_LATEST.md").write_text(release_notes, encoding="utf-8")
    print(f"[+] Release Syncthing completata con successo in: {syncthing_releases}")


if __name__ == "__main__":
    main()
