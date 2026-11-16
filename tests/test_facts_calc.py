import unittest
import json
from pathlib import Path

CHUNKS = Path('work/final-v2/index/chunks.json')

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
            ('3.接觸曝氣池', [1212, 1027, 1600, 1060]),
        ]
        sheet = facts.sheets(chunks(lines), DOCS)[0]
        found = [r for r in rows(sheet) if r['source'] == 'calc' and r['attr'] == 'effective_volume' and r['value'] == 237.37]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['entity'], '調整池')
        self.assertIn('2. 調整池 > B. 規劃計算', found[0]['section'])

    def test_unrelated_sheet(self):
        from docsearch.facts_calc import rows
        self.assertEqual(rows(facts.sheets(chunks([('unrelated 123', [0, 0, 100, 30])]), DOCS)[0]), [])

    def test_primary_tanks(self):
        from docsearch.facts_calc import rows
        lines = [
            ('三、處理單元功能計算', [369,2162,750,2200]),
            ('1. 初沉池', [388,2197,550,2230]),
            ('A.設計參數:', [405,2233,600,2260]),
            ('設計水量:Q=425 CMD', [434,2265,800,2298]),
            ('設計水力停留時間=12hr,則所需容積:', [436,2301,900,2333]),
            ('V=425m3/日x12÷24=212.5m3', [436,2335,900,2366]),
            ('B.規劃計算:', [407,2368,600,2398]),
            ('設計尺寸:', [433,2401,600,2431]),
            ('(1) L790xW352.5xH350/ D320', [447,2434,1000,2465]),
            ('(2) L790xW352.5xH350/ D320', [448,2468,1000,2500]),
            ('(3) L467.5xW300xH350/ D320', [447,2504,1000,2535]),
            ('有效容積:223.12m3> 212.5 m3----0K!', [448,2546,1000,2574]),
            ('停留時間:223.12÷410x24= 13.1(hr)', [435,2576,1000,2608]),
            ('2. 調整池', [385,2642,550,2675]),
            ('3.接觸曝氣池', [1212,1027,1600,1060]),
        ]
        result = rows(facts.sheets(chunks(lines), DOCS)[0])
        tanks = [r for r in result if r['attr'] == 'tank_dims']
        self.assertEqual([r['index'] for r in tanks], ['(1)', '(2)', '(3)'])
        self.assertAlmostEqual(sum(r['extra']['volume'] for r in tanks), 223.104)
        self.assertTrue(any(r['source'] == 'check' and r['extra'].get('printed') == 223.12
                            and r['extra']['consistent'] for r in result))

    def test_sewage_total_check(self):
        from docsearch.facts_calc import rows
        lines = [
            ('一、污水量計算:', [370,312,700,350]),
            ('(1)店鋪部分總樓地板面積共1013.55m2', [391,473,1000,505]),
            ('82人 x0.25 m3/人.d= 20.5 m3/day', [430,611,1000,645]),
            ('(2)圖書館部分設固定席次者', [392,651,1000,680]),
            ('62人 x0.10 m3/人.d= 6.2 m3/day', [431,788,1000,820]),
            ('估計建築物總使用人數:82+62+5+96+17+1338=1552人', [407,1537,1100,1565]),
            ('估計建築物平均日污水量:20.5+6.2+0.75+19.2+2.55+301.05', [406,1568,1100,1600]),
            ('=350.25m3/day', [633,1605,900,1637]),
            ('三、處理單元功能計算', [369,2162,750,2200]),
            ('1. 初沉池', [388,2197,550,2230]),
            ('2. 調整池', [385,2642,550,2675]),
            ('3.接觸曝氣池', [1212,1027,1600,1060]),
        ]
        result = rows(facts.sheets(chunks(lines), DOCS)[0])
        self.assertTrue(any(r['entity'] == '店鋪' and r['value'] == 82 for r in result))
        checks = [r['extra'] for r in result if r['source'] == 'check']
        self.assertTrue(any(e.get('printed') == 1552 and e['computed'] == 1600
                            and not e['consistent'] for e in checks))
        self.assertTrue(any(e.get('printed') == 350.25 and e['consistent'] for e in checks))

    @unittest.skipUnless(CHUNKS.exists(), 'real OCR index is not available')
    def test_real_sheet_acceptance(self):
        from docsearch.facts_calc import rows
        data = json.loads(CHUNKS.read_text())
        sheet = next(s for s in facts.sheets(data['chunks'], data['docs'])
                     if any('污水處理設施計算書' in line.norm for line in s.lines))
        result = rows(sheet)
        def has(entity, value, attr=None, unit=None):
            return any(r['entity'] == entity and r['value'] == value
                       and (attr is None or r['attr'] == attr)
                       and (unit is None or r['unit'] == unit)
                       for r in result if r['source'] == 'calc')
        required = {
            '初沉池': [425, 12, 212.5, 223.12, 13.1],
            '調整池': [237.37, 13.4, .30],
            '調整池泵浦': [.40, 8, 2],
            '調整池鼓風機': [5.42, 4000, 80, 10],
            '調整池粗氣泡散氣盤': [58],
            '接觸曝氣池': [68, 226.67, 280.33, 3.98, 6.78],
            '接觸曝氣池鼓風機': [7.31, 4000, 15],
            '接觸曝氣池細氣泡散氣盤': [76],
            '消毒池散氣盤': [4],
            '終沉池': [59.45, 3.4, 27.85, 15.26, 7.9],
            '終沉池泵浦': [.25, 7, 1, 2],
            '消毒池': [26.44, 1.5],
            '消毒池定量加藥機': [2],
            '放流池': [41.42, .9, 10.7],
            '放流池泵浦': [1, 18, 5, 2],
            '污泥貯留池': [4.13, 41.3, 71.41],
            '': [1552, 350.25, 420.3, 425],
        }
        for entity, values in required.items():
            for value in values:
                with self.subTest(entity=entity, value=value):
                    self.assertTrue(has(entity, value))
        for entity, value, attr, unit in [
            ('初沉池', 12, 'retention_time', 'hr'),
            ('初沉池', 13.1, 'retention_time', 'hr'),
            ('調整池', 13.4, 'retention_time', 'hr'),
            ('接觸曝氣池', 68, 'water_quality', 'kg/d'),
            ('終沉池', 3.4, 'retention_time', 'hr'),
            ('消毒池', 1.5, 'retention_time', 'hr'),
            ('放流池', .9, 'flow', 'm3/min'),
            ('放流池', 10.7, 'head', 'm'),
            ('調整池泵浦', 2, 'qty', 'count'),
            ('調整池鼓風機', 5.42, 'air', 'Nm3/min'),
            ('接觸曝氣池鼓風機', 7.31, 'air', 'Nm3/min'),
            ('消毒池PE貯藥桶', 1, 'qty', 'count'),
        ]:
            with self.subTest(unit=(entity, value, attr, unit)):
                self.assertTrue(has(entity, value, attr, unit))
        for entity, person, flow in [('店鋪',82,20.5), ('圖書館',62,6.2),
                                     ('關懷據點',5,.75), ('日間照護',96,19.2),
                                     ('幼兒園',17,2.55), ('集合住宅',1338,301.05)]:
            self.assertTrue(has(entity, person, 'persons'))
            self.assertTrue(has(entity, flow, 'sewage_flow'))
        for entity, values, op in [('原污水水質', [200,400,200,10000000], None),
                                   ('放流水質', [30,100,30,200000], '<')]:
            actual = [(r['value'], r['op']) for r in result
                      if r['entity'] == entity and r['attr'] == 'water_quality']
            self.assertEqual(actual, [(value, op) for value in values])
        dims = {(r['entity'], r['index']): r['extra'] for r in result
                if r['attr'] == 'tank_dims'}
        for key, expected in {
            ('初沉池','(1)'): (7.9,3.525,3.2), ('初沉池','(2)'): (7.9,3.525,3.2),
            ('初沉池','(3)'): (4.675,3,3.2), ('調整池','(1)'): (7.9,4.675,3.15),
            ('調整池','(2)'): (7.9,3.525,3.15), ('調整池','(3)'): (3.525,3,3.15),
            ('終沉池',None): (7.9,3.525,3.05), ('污泥貯留池','(1)'): (4.675,4.7,3.25),
        }.items():
            with self.subTest(dims=key):
                self.assertEqual(tuple(dims[key][k] for k in ('L','W','D')), expected)
        self.assertTrue(any(r['source'] == 'check' and r['extra'].get('printed') == 1552
                            and r['extra'].get('computed') == 1600
                            and not r['extra']['consistent'] for r in result))
        self.assertTrue(any(r['source'] == 'check' and r['extra'].get('printed') == 350.25
                            and r['extra']['consistent'] for r in result))
        self.assertFalse(has('初沉池', 237.37))
        self.assertTrue(all(src['bbox'][0] <= 4400 for r in result if r['source'] == 'calc'
                            for src in r['src']))
        self.assertTrue(all(not (src['bbox'][0] >= 3400 and src['bbox'][1] >= 1450)
                            for r in result if r['source'] == 'calc' for src in r['src']))
