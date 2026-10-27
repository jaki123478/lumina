"""Meaningful numerical checks for the from-scratch model (stdlib unittest)."""
import unittest
import sys
import json
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import CONFIG, Lumina50K, decode, encode, prompt_tokens
from train import export


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_exact_parameter_count_and_gradient_flow(self):
        model = Lumina50K()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 50000)
        tokens = torch.tensor([prompt_tokens("Ciao")])
        model(tokens).square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_future_tokens_cannot_change_prefix_logits(self):
        model = Lumina50K().eval()
        prefix = torch.tensor([prompt_tokens("Ciao")])
        extended = torch.cat([prefix, torch.tensor([[32, 97, 98, 99]])], dim=1)
        with torch.inference_mode():
            before = model(prefix)
            after = model(extended)[:, :prefix.shape[1]]
        torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-5)

    def test_byte_tokenizer_preserves_unicode(self):
        original = "Perché è così? Città, più, già. 😀"
        self.assertEqual(decode(encode(original)), original)

    def test_long_prompt_keeps_delimiters_and_fits_context(self):
        tokens = prompt_tokens("x" * 1000)
        self.assertEqual(tokens[0], CONFIG["bos_id"])
        self.assertEqual(tokens[-3:], encode("\nL:"))
        self.assertLessEqual(len(tokens), CONFIG["context_length"])

    def test_invalid_context_rejected(self):
        with self.assertRaises(ValueError):
            Lumina50K()(torch.zeros((1, CONFIG["context_length"] + 1), dtype=torch.long))

    def test_browser_export_roundtrip_preserves_model(self):
        original = Lumina50K().eval()
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(test_root))
            destination = Path(directory) / "weights.json"
            export(original, {"steps": 0}, destination)
            artifact = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(artifact["config"], CONFIG)
        restored = Lumina50K().eval()
        restored.load_state_dict({name: torch.tensor(values, dtype=torch.float32) for name, values in artifact["state_dict"].items()})
        self.assertEqual(sum(torch.tensor(values).numel() for values in artifact["state_dict"].values()), 50000)
        tokens = torch.tensor([prompt_tokens("Perché?")])
        with torch.inference_mode():
            torch.testing.assert_close(original(tokens), restored(tokens), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
