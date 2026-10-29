"""Inferenza Lumina via ONNX Runtime DirectML, con fallback CPU."""
from pathlib import Path
import numpy as np


class LuminaOnnxRuntime:
    def __init__(self, model_path=None, vocab_path=None, merges_path=None):
        import onnxruntime as ort
        from tokenizers import ByteLevelBPETokenizer
        root = Path(__file__).resolve().parent
        self.model_path = Path(model_path or root / "lumina_next_135m.onnx")
        vocab = vocab_path or root / "tokenizer_lumina_next" / "vocab.json"
        merges = merges_path or root / "tokenizer_lumina_next" / "merges.txt"
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        selected = [p for p in providers if p in available]
        if not selected:
            selected = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=selected)
        self.providers = self.session.get_providers()
        self.tokenizer = ByteLevelBPETokenizer(str(vocab), str(merges))

    def generate(self, prompt, max_new_tokens=96, eos_id=1):
        ids = self.tokenizer.encode(prompt).ids
        for _ in range(max_new_tokens):
            values = np.asarray([ids], dtype=np.int64)
            logits = self.session.run(["logits"], {"input_ids": values})[0]
            next_id = int(np.argmax(logits[0, -1]))
            ids.append(next_id)
            if next_id == eos_id:
                break
        text = self.tokenizer.decode(ids)
        # Restituisce solo il turno assistant, senza prompt o marcatori interni.
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant", 1)[1]
        for marker in ("<|im_end|>", "<|im_start|>", "<|pad|>"):
            text = text.split(marker, 1)[0]
        return text.strip()

    @property
    def provider_name(self):
        return "DmlExecutionProvider" if "DmlExecutionProvider" in self.providers else "CPUExecutionProvider"


def gpu_provider_available():
    import onnxruntime as ort
    return "DmlExecutionProvider" in ort.get_available_providers()
