import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lumina_desktop.server as server


class ServerImportTests(unittest.TestCase):
    def test_project_root_is_importable(self):
        self.assertEqual(server.PROJECT_ROOT, ROOT)
        import lumina_general  # noqa: F401
        import lumina_voice_neural  # noqa: F401


if __name__ == '__main__':
    unittest.main()
