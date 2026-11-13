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
    def test_ocr_percent_uses_tiles(self):
        self.assertEqual(percent_for('collection', 'ocr', 5, 10), 35)

    def test_index_reports_in_slices(self):
        import numpy as np
        from docsearch import index
        class Model:
            def half(self): return self
            def encode(self, texts, **kwargs):
                calls.append((len(texts), kwargs))
                return np.ones((len(texts), 2), dtype=np.float32)
        calls, reports = [], []
        chunks = [{'text': str(i)} for i in range(130)]
        fake = type('SentenceTransformerModule', (), {'SentenceTransformer': lambda *args, **kwargs: Model()})
        with tempfile.TemporaryDirectory() as temp, patch.dict('sys.modules', {'sentence_transformers': fake}), \
                patch.object(index, 'device', lambda: 'cpu'), \
                patch.object(index, 'build_doc', return_value=(chunks, [])), \
                patch.object(index.progress, 'report', side_effect=lambda *args: reports.append(args)), \
                patch.object(index.config, 'EMBED_ID', 'stub'):
            source = Path(temp) / 'in.json'
            source.write_text(json.dumps({'doc': 'd', 'image': 'i', 'size': [1, 1]}))
            index.build_index([source], Path(temp) / 'out')
        self.assertEqual([count for count, _ in calls], [64, 64, 2])
        self.assertTrue(all(options['batch_size'] == 16 and options['normalize_embeddings'] for _, options in calls))
        self.assertEqual([r for r in reports if r[0] == 'index'], [
            ('index', 0, 130, 'chunks'), ('index', 64, 130, 'chunks'),
            ('index', 128, 130, 'chunks'), ('index', 130, 130, 'chunks')])


if __name__ == '__main__':
    unittest.main()
