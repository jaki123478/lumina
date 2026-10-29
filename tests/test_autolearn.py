import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lumina_autolearn as learner


class AutoLearnTests(unittest.TestCase):
    def test_only_approved_examples_enter_curriculum(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(learner, 'STATE_DIR', root), patch.object(learner, 'QUEUE', root/'queue.jsonl'), patch.object(learner, 'CURRICULUM', root/'curriculum.jsonl'):
                learner.record_example('domanda', 'risposta', False)
                learner.record_example('domanda buona', 'risposta buona', True, 'coding')
                self.assertEqual(learner.approved_count(), 1)
                rows=[json.loads(x) for x in (root/'curriculum.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['user'], 'domanda buona')

    def test_empty_example_rejected(self):
        with self.assertRaises(ValueError): learner.record_example('', 'risposta')


if __name__ == '__main__': unittest.main()
