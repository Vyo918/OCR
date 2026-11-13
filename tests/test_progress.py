import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docsearch.progress import percent_for, read_progress, report


class ProgressTests(unittest.TestCase):
    def test_report_writes_atomic_valid_record_and_rate_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'progress.json'
            with patch.dict(os.environ, DOCSEARCH_PROGRESS_FILE=str(path)), patch('docsearch.progress.time.monotonic', side_effect=[1, 1.1, 1.3]):
                report('ocr', 1, 4, 'pages')
                first = path.read_text()
                report('ocr', 2, 4, 'pages')
                self.assertEqual(path.read_text(), first)
                report('ocr', 3, 4, 'pages')
                self.assertEqual(read_progress(path), {'stage': 'ocr', 'done': 3, 'total': 4, 'unit': 'pages'})

    def test_reader_rejects_corrupt_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'progress.json'
            path.write_text('{')
            self.assertIsNone(read_progress(path))
            path.write_text(json.dumps({'stage': 'wat', 'done': 1, 'total': 2, 'unit': 'pages'}))
            self.assertIsNone(read_progress(path))

    def test_report_without_env_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            report('ocr', 1, 2, 'pages')

    def test_percent_models_clamp_and_unknown(self):
        self.assertEqual(percent_for('collection', 'prepare', None, None), 0)
        self.assertEqual(percent_for('collection', 'ocr', 3, 2), 65)
        self.assertEqual(percent_for('collection', 'unknown', 1, 2), None)
        self.assertEqual(percent_for('question', 'reason', 4000, None), 95)
        self.assertLessEqual(percent_for('question', 'reason', 9000, None), 95)


if __name__ == '__main__':
    unittest.main()
