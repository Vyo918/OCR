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
    def test_ocr_percent_uses_page_fraction(self):
        values = [0, 200, 1000, 1500, 2000]
        percents = [percent_for('collection', 'ocr', n, 2000) for n in values]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual((percents[0], percents[-1]), (5, 65))

    def test_ingest_reports_page_fractions_and_cascade_stages(self):
        from PIL import Image
        from docsearch import ocr_ingest, progress

        class Plain:
            tile, overlap = 10, 0
            def spot(self, tile): return []
            def close(self): pass

        class FakeCascade:
            tile, overlap = 10, 0
            def __init__(self, zero_vlm=False): self.zero_vlm = zero_vlm
            def read_page(self, page, name, limit, on_tile=None, on_fraction=None):
                on_fraction(.1)
                on_fraction(.2)
                if self.zero_vlm:
                    on_fraction(1)
                else:
                    on_fraction(.2)
                    on_fraction(.2 + .8 / 3)
                    on_fraction(.2 + 1.6 / 3)
                    on_fraction(1)
                return []
            def close(self): pass

        def run(engine, count):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                paths = []
                for i in range(count):
                    path = root / f'{i}.png'
                    Image.new('RGB', (20, 10), 'black').save(path)
                    paths.append(path)
                reports, clock = [], [0.0]
                replace = os.replace
                def capture(src, dst):
                    reports.append(json.loads(Path(src).read_text()))
                    replace(src, dst)
                with patch.dict(os.environ, DOCSEARCH_PROGRESS_FILE=str(root / 'progress.json')), \
                        patch.object(progress, '_last', None), patch.object(progress, '_last_at', 0), \
                        patch.object(progress.time, 'monotonic', side_effect=lambda: clock.__setitem__(0, clock[0] + 1) or clock[0]), \
                        patch.object(progress.os, 'replace', side_effect=capture), \
                        patch.object(ocr_ingest.config, 'OCR_DIR', root / 'out'), \
                        patch.object(ocr_ingest, 'page_images', side_effect=lambda src, dest: [(src.stem, src)]), \
                        patch.object(ocr_ingest, 'collect_inputs', return_value=paths), \
                        patch.object(ocr_ingest, 'merge_overlap', side_effect=lambda lines: lines), \
                        patch.object(ocr_ingest.engines, 'make', return_value=engine):
                    ocr_ingest.ingest(paths, force=True)
                return [(r['stage'], r['done'], r['total'], r['unit']) for r in reports]

        plain = run(Plain(), 2)
        self.assertEqual(plain[-1], ('ocr', 2000, 2000, None))
        self.assertEqual([r[1] for r in plain], sorted(r[1] for r in plain))
        self.assertTrue(all(r[2:] == (2000, None) for r in plain))
        cascade = run(FakeCascade(), 1)
        fractions = [r[1] / r[2] for r in cascade]
        self.assertTrue(any(0 < f <= .2 for f in fractions))
        self.assertTrue(any(.2 < f < 1 for f in fractions))
        self.assertEqual(fractions[-1], 1)
        percents = [percent_for('collection', r[0], r[1], r[2]) for r in cascade]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(run(FakeCascade(True), 1)[-1][1:], (1000, 1000, None))

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
