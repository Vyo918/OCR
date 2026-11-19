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
        return self.crop_lines if len(self.calls) == 1 else []


class VoteEngine(Engine):
    def __init__(self, lines, votes):
        super().__init__(lines, [])
        self.votes = iter(votes)

    def spot(self, crop):
        self.calls.append(crop.size)
        if len(self.calls) == 1:
            return []
        text = next(self.votes, '')
        return [(text, (0, 0, 1, 1), .9)] if text else []


class Cascade:
    def __init__(self, fast):
        self.fast = fast

    def read_page(self, *args, **kwargs):
        return self.fast.read_page(*args, **kwargs)


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
        self.assertEqual(len(engine.calls), 41)
        self.assertEqual(lines, [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
                                 dict(doc='page', text='口徑', bbox=[100, 140, 140, 155], score=.8),
                                 dict(doc='page', text='3"', bbox=[160, 140, 180, 155], score=.7), old[-1]])
        bore = next(r for r in rows(sheet([(line['text'], line['bbox']) for line in lines])) if r['attr'] == 'bore')
        self.assertEqual(bore['value'], 3)
        self.assertAlmostEqual(bore['value_si'], 76.2)

    def test_preserves_missing_leading_phase_in_reread_value(self):
        cases = [
            ('1 220V 60HZ', '220V 60HZ', '1 220V 60HZ'),
            ('3 380V 60HZ', '380V 60HZ', '3 380V 60HZ'),
            ('1 220V 60HZ', '1 220V 60HZ', '1 220V 60HZ'),
            ('1 220V 60HZ', '1 220V 60HZ AC', '1 220V 60HZ AC'),
            ('1 220V 60HZ', '1相 220V 60HZ', '1相 220V 60HZ'),
            ('', '220V 60HZ', '220V 60HZ'),
            ('1 220V 60HZ', '380V 60HZ', '380V 60HZ'),
        ]
        for original, reread_value, expected in cases:
            with self.subTest(original=original, reread=reread_value):
                old = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
                       dict(doc='page', text='電源', bbox=[100, 140, 140, 155])]
                if original:
                    old.append(dict(doc='page', text=original, bbox=[160, 140, 250, 155]))
                engine = Engine(old, [('電源', (90, 150, 210, 195), .9),
                                      (reread_value, (270, 150, 540, 195), .9)])
                lines = self.ingest(engine)
                self.assertIn(expected, [line['text'] for line in lines])
                if expected.startswith(('1 ', '3 ')):
                    power = next(r for r in rows(sheet([(line['text'], line['bbox']) for line in lines]))
                                 if r['attr'] == 'power')
                    self.assertEqual(power['extra']['phase'], int(expected[0]))

    def test_fallbacks_and_no_title(self):
        lines = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
                 dict(doc='page', text='口徑', bbox=[100, 140, 140, 155]),
                 dict(doc='page', text='5', bbox=[160, 140, 180, 155])]
        engine = Engine(lines, [('nothing', (0, 0, 3, 3), None)])
        self.assertEqual(self.ingest(engine), lines)
        self.assertEqual(len(engine.calls), 41)
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

    def test_cell_vote_and_pump_parser(self):
        old = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
               dict(doc='page', text='口得', bbox=[100, 140, 140, 155]),
               dict(doc='page', text='5', bbox=[160, 140, 180, 155]),
               dict(doc='page', text='outside', bbox=[400, 400, 450, 420])]
        for votes, expected in [(['3'] * 24 + ['5'] * 9 + ['1'], 3),
                                (['3'] * 6 + ['5'] * 2, 3),
                                (['3'] * 6 + ['5'] * 4, None),
                                (['3'] * 5 + ['5'] * 5, None),
                                (['3'] * 7, None), (['abc'] * 40, None)]:
            with self.subTest(votes=votes):
                engine = VoteEngine(old, votes)
                lines = self.ingest(engine)
                self.assertEqual(len(engine.calls), 41)
                bore = next(r for r in rows(sheet([(line['text'], line['bbox']) for line in lines]))
                            if r['attr'] == 'bore')
                self.assertEqual(bore['value'], expected)
                self.assertIn(old[-1], lines)
                if expected is None:
                    self.assertIn(old[2], lines)
                    self.assertIsNone(bore['value_si'])
                else:
                    self.assertAlmostEqual(bore['value_si'], 76.2)
                    self.assertNotIn(old[2], lines)
                    if len(votes) == 34:
                        self.assertIn(dict(doc='page', text='3"', bbox=[142, 138, 232, 157],
                                           score=24 / 34, src='cellvote'), lines)

    def test_cell_vote_label_variants_and_cascade(self):
        for label in ('口徑', '口径', '口得', '口律', '口'):
            with self.subTest(label=label):
                lines = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
                         dict(doc='page', text=label, bbox=[100, 140, 140, 155]),
                         dict(doc='page', text='5', bbox=[160, 140, 180, 155])]
                engine = VoteEngine(lines, ['3'] * 40)
                result = self.ingest(Cascade(engine))
                self.assertTrue(any(line.get('src') == 'cellvote' for line in result))
                self.assertEqual(len(engine.calls), 41)

    def test_cell_vote_requires_label_and_enabled_switch(self):
        lines = [dict(doc='page', text='污水泵A規格', bbox=[100, 100, 220, 120]),
                 dict(doc='page', text='5', bbox=[160, 140, 180, 155])]
        engine = VoteEngine(lines, ['3'] * 40)
        self.assertEqual(self.ingest(engine), lines)
        self.assertEqual(len(engine.calls), 1)
        lines.insert(1, dict(doc='page', text='口徑', bbox=[100, 140, 140, 155]))
        engine = VoteEngine(lines, ['3'] * 40)
        self.assertEqual(self.ingest(engine, '0'), lines)
        self.assertEqual(engine.calls, [])
