import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lumina_autonomy as autonomy


class AutonomyTests(unittest.TestCase):
    def test_offline_check_is_structured(self):
        with patch('urllib.request.urlopen', side_effect=OSError('offline')):
            result = autonomy.check_server()
        self.assertFalse(result['online'])
        self.assertIn('error', result)

    def test_record_writes_only_supervisor_state(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            with patch.object(autonomy, 'STATE_DIR', target), \
                 patch.object(autonomy, 'HEARTBEAT', target / 'heartbeat.json'), \
                 patch.object(autonomy, 'EVENT_LOG', target / 'events.jsonl'):
                event = autonomy.record({'online': True, 'http': 200})
            self.assertTrue((target / 'heartbeat.json').exists())
            self.assertEqual(json.loads((target / 'heartbeat.json').read_text())['http'], 200)
            self.assertEqual(len((target / 'events.jsonl').read_text().splitlines()), 1)
            self.assertTrue(event['online'])


if __name__ == '__main__':
    unittest.main()
