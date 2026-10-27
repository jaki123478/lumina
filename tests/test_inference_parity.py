"""Verify browser inference numerically against the trained PyTorch model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from model import CONFIG, Lumina50K, encode, prompt_tokens


class BrowserInferenceTest(unittest.TestCase):
    def test_browser_matches_pytorch_and_rollover(self):
        torch.set_num_threads(2)
        artifact = json.loads((ROOT / "dist/model/weights.json").read_text(encoding="utf-8"))
        model = Lumina50K().eval()
        model.load_state_dict({key: torch.tensor(value) for key, value in artifact["state_dict"].items()})
        examples = [prompt_tokens("Ciao"), prompt_tokens("Come stai? È una prova."), [CONFIG["bos_id"]] + encode("abcde" * 32)]
        fixture = {"artifact": artifact, "examples": []}
        # Prompt construction (including memory of the last exchange) must match byte for byte.
        long_answer = "Una piccola stella perse la strada. La Luna le mostrò come tornare a casa."
        fixture["prompts"] = []
        for messages in [
            [{"role": "user", "content": "Ciao"}],
            [{"role": "user", "content": "Come ti chiami?"}, {"role": "assistant", "content": "Sono Lumina."}, {"role": "user", "content": "Ripeti"}],
            [{"role": "user", "content": "Raccontami una storia"}, {"role": "assistant", "content": long_answer}, {"role": "user", "content": "Ripeti"}],
            [{"role": "user", "content": "Ciao"}, {"role": "assistant", "content": "Ciao!\n  Come   va?"}, {"role": "user", "content": "Grazie è"}],
            [{"role": "assistant", "content": "Salve"}, {"role": "user", "content": "Ciao"}],
        ]:
            turns = [m for m in messages]
            previous = (turns[-3]["content"], turns[-2]["content"]) if len(turns) >= 3 and turns[-2]["role"] == "assistant" and turns[-3]["role"] == "user" else None
            fixture["prompts"].append({"messages": messages, "tokens": prompt_tokens(messages[-1]["content"], previous)})
        with torch.inference_mode():
            for tokens in examples:
                expected = model(torch.tensor([tokens]))[0, -1].tolist()
                fixture["examples"].append({"tokens": tokens, "expected": expected})
            additions = encode("Nuovi byte")
            rolled = (examples[-1] + additions)[-CONFIG["context_length"]:]
            fixture["rollover"] = {"tokens": examples[-1], "append": additions,
                                   "expected": model(torch.tensor([rolled]))[0, -1].tolist()}
        with tempfile.TemporaryDirectory(prefix=".parity-", dir=ROOT / "tests") as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            result = subprocess.run(["node", str(ROOT / "tests/inference_parity.cjs"), str(path)], check=True, capture_output=True, text=True, cwd=ROOT)
        report = json.loads(result.stdout)
        self.assertEqual(report["parameters"], 50000)
        self.assertLess(report["maximumDifference"], 0.00015)
        print("Browser/PyTorch parity:", report)


if __name__ == "__main__":
    unittest.main()
