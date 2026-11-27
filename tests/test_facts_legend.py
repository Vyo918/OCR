import unittest

from docsearch import factquery, facts
from docsearch.facts import sheets, dedupe, stitch
from docsearch.facts_tables import rows
from tests.facts_fixtures import chunks, DOCS

# System sheet legend table: 圖例 header at x~3420, 說明 at x~3780, quantity column x~4220-4320.
LEGEND = [('說明', [3778, 1449, 3843, 1495]), ('圖例', [3419, 1450, 3480, 1491]),
          ('60cm防臭式鑄鐵蓋', [3499, 1498, 3747, 1540]), ('15只', [4224, 1499, 4312, 1546]),
          ('8', [3502, 1503, 3519, 1537]),
          ('φ75cm防臭式鑄鐵蓋', [3500, 1546, 3748, 1586]), ('3只', [4230, 1545, 4307, 1592]),
          ('8', [3502, 1549, 3519, 1584]),
          ('浮渣擋板', [3496, 1585, 3612, 1632]), ('(FRP/SUS 304)', [3615, 1585, 3838, 1638]),
          ('7组', [4231, 1590, 4306, 1638]), ('7', [3415, 1589, 3478, 1635]),
          ('整流設備', [3496, 1633, 3612, 1677]), ('(FRP/SUS 304)', [3613, 1629, 3838, 1684]),
          ('2组', [4229, 1637, 4306, 1685]),
          ('溢流堰+擋板', [3493, 1675, 3700, 1730]), ('(FRP/SUS 304）', [3702, 1675, 3884, 1730]),
          ('7.9 m', [4215, 1677, 4318, 1728]),
          ('接觸曝氣池細氣泡散氣盤', [3497, 1725, 3782, 1767]), ('76 只', [4224, 1725, 4313, 1772]),
          ('調整池粗氣泡散氣盤', [3497, 1770, 3732, 1813]), ('58 只', [4223, 1774, 4313, 1820]),
          ('消毒池粗氣泡散氣盤', [3497, 1818, 3732, 1858]), ('4只', [4234, 1819, 4307, 1863]),
          ('接觸曝氣池繩状濾材(濾材ø50mm,比表面積≥5m2/m)', [3493, 1859, 4139, 1915]), ('14490m', [4210, 1868, 4328, 1912]),
          ('調整池泵浦', [3496, 1910, 3633, 1951]), ('(0.40m3/min×8mH× 2HP)', [3635, 1907, 4053, 1961]),
          ('2台', [4229, 1911, 4303, 1959]),
          ('螺旋式流量計(4")', [3495, 2229, 3714, 2281]), ('1组', [4226, 2232, 4299, 2281])]


def sheet(lines):
    s = sheets(chunks(lines), DOCS)[0]
    return type(s)(s.doc, s.size, stitch(dedupe(s.lines), s.h), s.h)


def legend_rows(lines=LEGEND):
    return [r for r in rows(sheet(lines)) if r['source'] == 'legend']


class LegendRows(unittest.TestCase):
    def pick(self, name, attr='qty'):
        return [r for r in legend_rows() if r['entity'] == name and r['attr'] == attr]

    def test_counts_pair_with_their_names(self):
        self.assertEqual([r['value'] for r in self.pick('60cm防臭式鑄鐵蓋')], [15])
        self.assertEqual([r['value'] for r in self.pick('φ75cm防臭式鑄鐵蓋')], [3])
        self.assertEqual([r['value'] for r in self.pick('調整池粗氣泡散氣盤')], [58])
        self.assertEqual([r['value'] for r in self.pick('浮渣擋板')], [7])
        self.assertEqual([r['value'] for r in self.pick('整流設備')], [2])
        self.assertEqual([r['value'] for r in self.pick('螺旋式流量計')], [1])

    def test_units_are_count_for_组_and_只(self):
        for name in ['浮渣擋板', '整流設備', '螺旋式流量計']:
            self.assertEqual(self.pick(name)[0]['unit'], 'count')

    def test_material_from_parenthesised_line(self):
        self.assertEqual(self.pick('整流設備')[0]['extra'], {'material': 'FRP/SUS 304'})
        self.assertEqual(self.pick('浮渣擋板')[0]['extra'], {'material': 'FRP/SUS 304'})

    def test_length_row(self):
        length = self.pick('接觸曝氣池繩状濾材', 'length')
        self.assertEqual([(r['value'], r['unit']) for r in length], [(14490, 'm')])

    def test_stray_glyphs_make_no_rows(self):
        self.assertFalse([r for r in legend_rows() if r['entity'] in {'8', '7', '⑧', 'O', 'M'}])
        self.assertFalse([r for r in legend_rows() if not r['entity'] or len(r['entity']) <= 2])


class LegendQuery(unittest.TestCase):
    def test_diameter_selects_matching_legend_row(self):
        facts_rows = facts.extract(chunks(LEGEND), DOCS)
        result = factquery.context('直徑75cm防臭蓋有幾個?', facts_rows)
        ids = [r['id'] for r in result['rows']]
        big = [r['id'] for r in facts_rows if r['source'] == 'legend' and r['entity'] == 'φ75cm防臭式鑄鐵蓋']
        small = [r['id'] for r in facts_rows if r['source'] == 'legend' and r['entity'] == '60cm防臭式鑄鐵蓋']
        self.assertEqual(len(big), 1)
        self.assertIn(big[0], ids)
        self.assertNotIn(small[0], ids)

    def selected(self, question):
        facts_rows = facts.extract(chunks(LEGEND), DOCS)
        return [r['entity'] for r in factquery.select(facts_rows, factquery.parse(question))]

    def test_only_best_matching_legend_names_are_selected(self):
        self.assertEqual(self.selected('浮渣擋板的數量是多少?'), ['浮渣擋板'])
        self.assertEqual(self.selected('溢流堰及擋板的材質與總長度?'), ['溢流堰+擋板'])
        self.assertEqual(sorted(self.selected('粗氣泡數量是多少?')), ['消毒池粗氣泡散氣盤', '調整池粗氣泡散氣盤'])

    def test_family_question_gets_only_its_own_legend_row(self):
        self.assertEqual(self.selected('消毒池的粗氣泡散氣盤數量?'), ['消毒池粗氣泡散氣盤'])

    def test_pump_question_gets_no_legend_rows_added(self):
        self.assertEqual(self.selected('調整池泵浦有幾台?'), ['調整池泵浦'])


if __name__ == '__main__':
    unittest.main()
