import unittest

from docsearch import facts, index
from tests.facts_fixtures import DOCS, DUPLICATES, PAIRS, chunks


class Core(unittest.TestCase):
    def test_sheets_and_dimensions(self):
        self.assertEqual(facts.sheets([], DOCS), [])
        s = facts.sheets(chunks(PAIRS[0]), DOCS)[0]
        self.assertEqual(s.doc, 'sheet')
        self.assertEqual(len(s.lines), 2)
        self.assertEqual(index.parse_dims('(1) L790×W352.5×H350/D320'), [('1', 7.9, 3.525, 3.5, 3.2)])

    def test_dedupe(self):
        for pair in DUPLICATES:
            lines = facts.sheets(chunks(pair), DOCS)[0].lines
            self.assertEqual([x.text for x in facts.dedupe(lines)], [pair[0][0]])

    def test_stitch(self):
        for pair in PAIRS:
            with self.subTest(pair=pair):
                s = facts.sheets(chunks(pair), DOCS)[0]
                result = facts.stitch(s.lines, s.h)
                self.assertEqual(len(result), 1)
                self.assertEqual(len(result[0].src), 2)
        for n, expected in [(0, 'a.設計尺寸：L790×W352.5×H350/ D305'), (2, '實際表面積負荷=425÷27.85=15.26m3/m2.d ...0K')]:
            s = facts.sheets(chunks(PAIRS[n]), DOCS)[0]
            self.assertEqual(facts.stitch(s.lines, s.h)[0].text, expected)

    def test_quantities(self):
        for text, op, value, unit in [('>1860 L/min', '>', 1.86, 'm3/min'), ('二台', None, 2, 'count'), ('5 mg/!', None, 5, 'mg/L'), ('∅3.0"', None, 76.2, 'mm'), ('425 CMD', None, 425, 'm3/d')]:
            with self.subTest(text=text):
                q = facts.quantity(text)
                self.assertEqual(q['op'], op)
                self.assertAlmostEqual(q['value_si'], value)
                self.assertEqual(q['unit_si'], unit)
        self.assertIsNone(facts.quantity('沒有數字'))

    def test_equations(self):
        for expr, printed, computed, consistent in [('DT=237.37÷425×24÷13.4(hr)', '13.4', 237.37/425*24, True), ('82+62+5+96+17+1338=1552人', '1552', 1600, False), ('7.9×3.52×1.85×2+4.675×7.9×1.85×2=239.5m3', '239.5', 239.5, True), ('7.7x3.52×3.5x6+4.675×7.9×3.5×4=379m3', '379', 1086.24, False), ('307個×1/2×0.4÷62人', '62', 61.4, True)]:
            with self.subTest(expr=expr):
                result = facts.equation(expr, printed)
                self.assertAlmostEqual(result['computed'], computed, delta=0.1)
                self.assertEqual(result['consistent'], consistent)

    def test_families(self):
        for text, expected in [('桐沉池1-1)', '初沉池'), ('抽水泡1-7', '抽水池'), ('行充贮留池)', None), ('接順暖氣池-2)', None)]:
            self.assertEqual(facts.family_of(text), expected)
