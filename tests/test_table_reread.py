import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from docsearch.ocr_ingest import ingest_image
from docsearch.facts_tables import rows
from tests.test_facts_tables import sheet


class Engine:
    def __init__(self, lines, crop_lines):
        self.lines = lines
        self.crop_lines = crop_lines
        self.calls = []

    def read_page(self, page, name, limit, on_tile=None, on_fraction=None):
        return self.lines

    def spot(self, crop):
        self.calls.append(crop.size)
        return self.crop_lines


class TableReread(unittest.TestCase):
    def ingest(self, engine, switch='1'):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'DOCSEARCH_TABLE_REREAD': switch}):
            path = Path(tmp) / 'page.png'
            Image.new('RGB', (500, 500), 'white').save(path)
            out = ingest_image('page', path, engine, Path(tmp))
            return json.loads(out.read_text())['lines']

    def test_crop_replaces_lines_and_feeds_pump_parser(self):
        old = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
               dict(doc='page', text='口得', bbox=[100, 140, 140, 155]),
               dict(doc='page', text='5', bbox=[160, 140, 180, 155]),
               dict(doc='page', text='outside', bbox=[400, 400, 450, 420])]
        engine = Engine(old, [('污水泵A規格', (90, 30, 450, 90), .9),
                              ('口徑', (90, 150, 210, 195), .8),
                              ('3"', (270, 150, 330, 195), .7)])
        lines = self.ingest(engine)
        self.assertEqual(engine.calls, [(666, 690)])
        self.assertEqual(lines, [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120], score=.9),
                                 dict(doc='page', text='口徑', bbox=[100, 140, 140, 155], score=.8),
                                 dict(doc='page', text='3"', bbox=[160, 140, 180, 155], score=.7), old[-1]])
        bore = next(r for r in rows(sheet([(line['text'], line['bbox']) for line in lines])) if r['attr'] == 'bore')
        self.assertEqual(bore['value'], 3)
        self.assertAlmostEqual(bore['value_si'], 76.2)

    def test_fallbacks_and_no_title(self):
        lines = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
                 dict(doc='page', text='口徑', bbox=[100, 140, 140, 155]),
                 dict(doc='page', text='5', bbox=[160, 140, 180, 155])]
        engine = Engine(lines, [('nothing', (0, 0, 3, 3), None)])
        self.assertEqual(self.ingest(engine), lines)
        self.assertEqual(len(engine.calls), 1)
        engine = Engine(lines, [])
        self.assertEqual(self.ingest(engine, '0'), lines)
        self.assertEqual(engine.calls, [])
        engine = Engine([lines[1]], [])
        self.assertEqual(self.ingest(engine), [lines[1]])
        self.assertEqual(engine.calls, [])
        class NoSpot:
            def read_page(self, page, name, limit, on_tile=None, on_fraction=None):
                return lines
        self.assertEqual(self.ingest(NoSpot()), lines)
