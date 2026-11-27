import json
from pathlib import Path
import unittest
from docsearch.facts import sheets, dedupe, stitch
from docsearch.facts_tables import rows
from tests.facts_fixtures import chunks, DOCS


def sheet(lines):
    s = sheets(chunks(lines), DOCS)[0]
    return type(s)(s.doc, s.size, stitch(dedupe(s.lines), s.h), s.h)


class Tables(unittest.TestCase):
    def test_legend_and_symbol(self):
        s = sheet([('說明', [3778,1449,3843,1495]), ('图例',[3419,1450,3480,1491]),
                   ('60cm防臭式鑄鐵蓋',[3499,1498,3747,1540]), ('15只',[4224,1499,4312,1546]),
                   ('8',[3502,1503,3519,1537]), ('調整池泵浦',[3496,1910,3633,1951]),
                   ('(0.40m3/min×8mH× 2HP)',[3635,1907,4053,1961]), ('2台',[4229,1911,4303,1959])])
        r = rows(s)
        self.assertTrue(any(x['entity'] == '60cm防臭式鑄鐵蓋' and x['value'] == 15 for x in r))
        self.assertFalse(any(x['entity'] == '8' for x in r))

    def test_no_anchor(self):
        self.assertEqual(rows(sheet([('一、污水量計算：',[370,312,590,346])])), [])

    def test_pump(self):
        s = sheet([('洪放流泵规格',[588,1552,724,1586]), ('数是',[572,1578,626,1608]),
                   ('3台',[633,1579,704,1608]), ('流量',[571,1667,625,1697]),
                   ('>1860',[621,1672,671,1695]), ('L/min',[668,1672,723,1698])])
        r = rows(s)
        self.assertTrue(any(x['value'] == 3 for x in r))
        self.assertTrue(any(x['value_si'] == 1.86 for x in r))

    def test_bore_requires_an_inch_mark(self):
        for reading, expected in [('5', None), ('2', None), ('2°', 2), ('3"', 3), ('4吋', 4), ('6″', 6)]:
            with self.subTest(reading=reading):
                s = sheet([('污水泵A規格', [100,100,220,120]),
                           ('口得', [100,140,140,155]), (reading, [160,140,180,155])])
                bore = next(r for r in rows(s) if r['attr'] == 'bore')
                self.assertEqual(bore['value'], expected)
                if expected is None:
                    self.assertEqual(bore['extra']['suspect'], 'no inch mark')
                    self.assertIsNone(bore['value_si'])
                    self.assertIsNone(bore['unit'])
                    self.assertIsNone(bore['unit_si'])
                else:
                    self.assertAlmostEqual(bore['value_si'], expected * 25.4)

    def test_overlapping_unmarked_and_marked_bore_reads(self):
        s = sheet([('污水泵B規格', [100,100,220,120]),
                   ('口径', [100,140,140,155]), ('2', [155,135,185,160]),
                   ('2°', [162,142,178,154])])
        bore = next(r for r in rows(s) if r['attr'] == 'bore')
        self.assertEqual(bore['value'], 2)
        self.assertEqual(bore['value_si'], 50.8)





REAL = Path('work/final-v2/index/chunks.json')


@unittest.skipUnless(REAL.exists(), 'real OCR index unavailable')
class RealTables(unittest.TestCase):
    def test_all_legend_and_pump_values(self):
        data = json.loads(REAL.read_text())
        docs = {c['doc']: {'size': [4964, 3510]} for c in data['chunks']}
        found = []
        for s in sheets(data['chunks'], docs):
            s = type(s)(s.doc, s.size, stitch(dedupe(s.lines), s.h), s.h)
            found.extend(rows(s))
        legend = [r for r in found if r['source'] == 'legend' and r['label'] in ('數量', '長度')
                  and any(s['bbox'][0] >= 4210 for s in r['src'])]
        self.assertEqual([r['value'] for r in legend], [15, 3, 7, 2, 7.9, 76, 58, 4, 14490, 2, 2, 2, 2, 2, 2, 1, 1])
        expected = {
            '前置雨水泵': (2, 380, None, 267, 2, 1),
            '洪放流泵': (3, 380, 6, 1860, 17, 15),
            '污水泵B': (2, 220, 2, 170, 10, 2),
            '污水泵C': (2, 380, None, 500, 15, 5),
            '污水泵A': (2, 380, None, 500, 15, 5),
        }
        for entity, values in expected.items():
            actual = {r['attr']: r['value'] for r in found if r['source'] == 'pump_table' and r['entity'] == entity}
            self.assertEqual(tuple(actual.get(a) for a in ('qty', 'power', 'bore', 'flow', 'head', 'hp')), values)