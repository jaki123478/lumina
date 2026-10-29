import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumina_onnx_runtime import LuminaOnnxRuntime, gpu_provider_available


class OnnxRuntimeTests(unittest.TestCase):
    def test_provider_and_generation(self):
        runtime = LuminaOnnxRuntime()
        self.assertTrue(runtime.providers)
        text = runtime.generate('<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n', max_new_tokens=4)
        self.assertIsInstance(text, str)
        self.assertTrue(text)

    def test_directml_is_detected_when_installed(self):
        self.assertTrue(gpu_provider_available())


if __name__ == '__main__':
    unittest.main()
