import json
from pathlib import Path
import unittest

from docsearch import facts
from docsearch.facts_labels import rows
from tests.facts_fixtures import chunks, DOCS


class PlanLabels(unittest.TestCase):
    def test_labels_and_volume(self):
        lines = [
            ('抽水池1-7)', [2292, 1863, 2399, 1901]),
            ('初沉池1-6)', [2529, 1864, 2642, 1902]),
            ('H=1.85m', [2313, 1892, 2394, 1921]),
            ('H=1.85m', [2549, 1893, 2630, 1922]),
            ('X14污水處理槽(1)平面圖X15', [2628, 2162, 3230, 2235]),
            ('7.9×3.52×1.85×2+4.675×7.9×1.85×2=239.5m3', [2603, 2225, 3168, 2267]),
            ('調整池(2+1)', [3251, 2691, 3367, 2732]),
            ('H=3.5m', [3275, 2718, 3346, 2747]),
            ('接觸曝氣池-3)H=3.5m', [2288, 2934, 2503, 2973]),
            ('污水處理(2)流程平面圖', [2750, 3038, 3263, 3098]),
            ('7.7x3.52×3.5x6+4.675×7.9×3.5×4=379m3', [2748, 3105, 3259, 3146]),
        ]
        result = rows(facts.sheets(chunks(lines), DOCS)[0])
        depths = {(r['family'], r['index'], r['value']) for r in result if r['attr'] == 'label_depth'}
        self.assertTrue({('抽水池', '1-7', 1.85), ('初沉池', '1-6', 1.85), ('調整池', '2-1', 3.5), ('接觸曝氣池', '3-3', 3.5)} <= depths)
        checks = [r for r in result if r['source'] == 'check']
        self.assertEqual([r['extra']['consistent'] for r in checks], [True, False])
        self.assertAlmostEqual(checks[1]['extra']['computed'], 1086.239)
        volume = next(r for r in result if r['source'] == 'plan_volume')
        self.assertEqual({result[i]['index'] for i in volume['extra']['member_idx']}, {'1-7', '1-6'})

    def test_equipment_and_rejected_labels(self):
        lines = [
            ('污水泵C', [3378, 1806, 3451, 1838]), ('5HPx2', [3379, 1826, 3429, 1852]),
            ('桐沉池1-1)', [2023, 2377, 2085, 2399]), ('H=3.5m', [2035, 2393, 2077, 2413]),
            ('行充贮留池)', [1814, 2405, 1875, 2419]), ('H=15m', [1823, 2414, 1863, 2430]),
            ('初沉池(1-2)', [1937, 2399, 1988, 2413]), ('H=J1.5m', [1943, 2408, 1985, 2427]),
        ]
        result = rows(facts.sheets(chunks(lines), DOCS)[0])
        self.assertIn(('污水泵C', 'hp', 5), {(r['entity'], r['attr'], r['value']) for r in result})
        self.assertIn(('初沉池', '1-1', 3.5), {(r['family'], r['index'], r['value']) for r in result})
        self.assertFalse(any('H=15m' in r['text'] or 'H=J1.5m' in r['text'] for r in result))


REAL = Path('work/final-v2/index/chunks.json')


@unittest.skipUnless(REAL.exists(), 'real OCR index not available')
class RealPlanLabels(unittest.TestCase):
    def test_both_sheets(self):
        data = json.loads(REAL.read_text())
        chunks_ = data['chunks']
        docs = {c['doc']: {'size': [5000, 3500]} for c in chunks_}
        sheets = facts.sheets(chunks_, docs)
        parsed = [rows(s) for s in sheets]
        sys = next(r for r in parsed if any(x['source'] == 'plan_volume' and x['value'] == 239.5 for x in r))
        raft = next(r for r in parsed if any(x['entity'] == '污水泵C' and x['attr'] == 'hp' for x in r))
        depths = {(r['family'], r['index'], r['value']) for r in sys if r['attr'] == 'label_depth'}
        expected = {('抽水池', '1-7', 1.85)}
        expected |= {('初沉池', f'1-{n}', 1.85) for n in (4, 5, 6)}
        expected |= {('初沉池', f'1-{n}', 3.5) for n in (1, 2, 3)}
        expected |= {('調整池', f'2-{n}', 3.5) for n in (1, 2)}
        expected |= {('接觸曝氣池', f'3-{n}', 3.5) for n in (1, 2, 3)}
        expected |= {('消毒池', '5', 3.5), ('放流池', '6', 3.5), ('終沉池', None, 3.5), ('污泥貯留池', None, 3.5)}
        self.assertTrue(expected <= depths, expected - depths)
        self.assertTrue(any(r['index'] == '2-3' and r['attr'] == 'label' for r in sys))
        self.assertFalse(any(r['index'] == '2-3' and r['attr'] == 'label_depth' for r in sys))
        volume = next(r for r in sys if r['source'] == 'plan_volume' and r['value'] == 239.5)
        self.assertTrue({'1-4', '1-5', '1-6', '1-7'} <= {sys[i]['index'] for i in volume['extra']['member_idx']})
        checks = {r['value']: r['extra'] for r in sys if r['source'] == 'check'}
        self.assertTrue(checks[239.5]['consistent'])
        self.assertFalse(checks[379]['consistent'])
        self.assertAlmostEqual(checks[379]['computed'], 1086.239)
        depths = {(r['family'], r['index'], r['value']) for r in raft if r['attr'] == 'label_depth'}
        expected = {('初沉池', f'1-{n}', 1.85) for n in (4, 5, 6)}
        expected |= {('抽水池', '1-7', 1.85), ('初沉池', '1-1', 3.5), ('調整池', '2-1', 3.5), ('調整池', '2-2', 3.5)}
        self.assertTrue(expected <= depths, expected - depths)
        values = {(r['entity'], r['attr'], r['value']) for r in raft}
        self.assertTrue({('污水泵A', 'hp', 5), ('污水泵B', 'hp', 1), ('污水泵C', 'hp', 5)} <= values)
        self.assertTrue({('污水泵' + c, 'qty', 2) for c in 'ABC'} <= values)
        self.assertFalse(any(any(s in r['text'] for s in ('行充贮留池)', 'H=15m', 'H=J1.5m', 'PVC')) for r in raft))
