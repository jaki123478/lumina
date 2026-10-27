"""
Esportatore ONNX per Lumina AI (2.75M e 14M).
Produce file .onnx ottimizzati per l'inferenza ultra-rapida su CPU, WebAssembly e ONNX Runtime Web.
"""

import os
import sys
import torch
from modeling_lumina import LuminaForCausalLM

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class LuminaOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids)
        return out if isinstance(out, torch.Tensor) else out.logits


def export_model(model_dir: str, output_path: str, model_name: str):
    print(f"\n[*] Caricamento pesi per {model_name} da '{model_dir}'...")
    if not os.path.exists(model_dir):
        print(f"[!] Directory {model_dir} non trovata. Salto.")
        return False

    model = LuminaForCausalLM.from_pretrained(model_dir).eval()
    wrapper = LuminaOnnxWrapper(model)

    dummy_input = torch.tensor([[1, 2, 3]], dtype=torch.long)

    print(f"[*] Esportazione in corso verso '{output_path}'...")
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

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] {model_name} esportato con successo in ONNX!")
    print(f"          Dimensione: {size_mb:.2f} MB ({os.path.getsize(output_path):,} byte)")
    return True


def main():
    print("=" * 65)
    print("      ⚡ LUMINA ONNX EXPORTER (CPU & BROWSER READY)")
    print("=" * 65)

    # 1. Export 2.75M
    export_model("hf_export_lumina_large", "lumina_2.75m.onnx", "Lumina 2.75M")

    # 2. Export 14M se disponibile
    if os.path.exists("hf_export_lumina_xlarge"):
        export_model("hf_export_lumina_xlarge", "lumina_14m.onnx", "Lumina 14M")

    print("\n[+] Esportazione completata. I modelli .onnx sono pronti per l'uso.")


if __name__ == "__main__":
    main()
