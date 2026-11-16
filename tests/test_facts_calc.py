import unittest

from docsearch import facts
from tests.facts_fixtures import chunks, DOCS


class CalcRows(unittest.TestCase):
    def test_cross_column_heading(self):
        from docsearch.facts_calc import rows
        lines = [
            ('污水處理設施計算書', [365, 252, 750, 286]),
            ('一、污水量計算：', [370, 312, 700, 350]),
            ('三、處理單元功能計算', [369, 2162, 750, 2200]),
            ('1. 初沉池', [388, 2197, 550, 2230]),
            ('2. 調整池', [385, 2642, 550, 2675]),
            ('A.設計參數：', [410, 2681, 600, 2710]),
            ('設計水量 Q=425 m3/d', [436, 2714, 750, 2745]),
            ('V≥ 212.5 m3', [441, 2882, 700, 2915]),
            ('B. 規劃計算：', [1222, 314, 1342, 355]),
            ('a. 設計尺寸：', [1229, 350, 1400, 380]),
            ('(1)L790×W467.5×H350/ D315', [1235, 383, 1750, 415]),
            ('(2)L790×W352.5×H350/ D315', [1235, 417, 1750, 450]),
            ('(3)L352.5×W300×H350/ D315', [1237, 456, 1750, 485]),
            ('有效容積：237.37m3 > 212.5m3----0K！', [1252, 489, 1708, 530]),
        ]
        sheet = facts.sheets(chunks(lines), DOCS)[0]
        found = [r for r in rows(sheet) if r['attr'] == 'effective_volume' and r['value'] == 237.37]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['entity'], '調整池')
        self.assertIn('2. 調整池 > B. 規劃計算', found[0]['section'])

    def test_unrelated_sheet(self):
        from docsearch.facts_calc import rows
        self.assertEqual(rows(facts.sheets(chunks([('unrelated 123', [0, 0, 100, 30])]), DOCS)[0]), [])
