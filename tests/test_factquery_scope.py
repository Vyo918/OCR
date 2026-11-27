import unittest

from docsearch import factquery
from tests.test_factquery import row


class FactQueryScopeTests(unittest.TestCase):
    def test_treatment_phrase_is_not_a_qualifier(self):
        rows = [row(1, '初沉池(1-2)', 'label_depth', 2, 'plan_label', '初沉池', '1-2', 'm')]
        for phrase in ('污水處理槽(2)', '污水處理系統', '污水處理'):
            with self.subTest(phrase=phrase):
                result = factquery.context(f'{phrase}的初沉池(1-2)水深？', rows)
                self.assertEqual(result['rows'][0]['id'], 'F001')

    def test_equipment_qualifiers_remain_filters(self):
        rows = [row(1, '污水泵B', 'hp', 2, 'pump_table', index='B'),
                row(2, '雨水泵B', 'hp', 3, 'pump_table', index='B'),
                row(3, '消防泵B', 'hp', 4, 'pump_table', index='B')]
        for name, expected in [('污水', 'F001'), ('雨水', 'F002'), ('消防', 'F003')]:
            with self.subTest(name=name):
                result = factquery.context(f'{name}泵B馬力多少？', rows)
                self.assertEqual([r['id'] for r in result['rows']], [expected])

    def test_group_and_size_terms(self):
        rows = [row(i, '調整池', 'tank_dims', None, family='調整池', index=f'({i})', L=7.9, W=4.675, H=3.5, D=3.15, volume=116.34) for i in range(1, 4)]
        for question in ('調整池第2組的規劃尺寸？', '調整池第 二 組的尺寸？', '調整池第2組 planned size?'):
            with self.subTest(question=question):
                self.assertEqual(factquery.context(question, rows)['rows'][0]['id'], 'F002')

    def test_scoped_count_and_fallback(self):
        rows = [row(i, f'初沉池(1-{i})', 'label_depth', 2, 'plan_label', '初沉池', f'1-{i}', 'm') for i in range(1, 5)]
        rows.append(row(5, '抽水池(1-5)', 'label_depth', 2, 'plan_label', '抽水池', '1-5', 'm'))
        for i, members in [(1, ['F001', 'F002']), (2, ['F003', 'F004', 'F005'])]:
            r = row(5+i, '處理槽', 'plan_volume', 20, 'plan_volume', members=members)
            r['section'] = f'污水處理槽({i})平面圖'
            rows.append(r)
        for scope in ('污水處理槽(2)', '槽(2)', '(2)流程平面圖'):
            with self.subTest(scope=scope):
                result = factquery.context(f'{scope}初沉池共有幾槽？', rows)
                self.assertIn('plan 污水處理槽(2)平面圖 labels by name: 2 [1-3, 1-4]', result['computed'])
                self.assertIn('other members: 抽水池(1-5)', result['computed'])
        for question in ('初沉池共有幾槽？', '槽(9)初沉池共有幾槽？'):
            self.assertIn('labels by name: 4 [1-1, 1-2, 1-3, 1-4]', factquery.context(question, rows)['computed'])

        self.assertIn('labels by name: 4 [1-1, 1-2, 1-3, 1-4]',
                      factquery.context('綜合計算書與污水處理槽(2)平面圖，初沉池共幾槽？', rows)['computed'])

    def test_measured_plan_rows_and_conflicts(self):
        rows = [row(1, '初沉池(1-2)', 'tank_dims', None, 'plan_dims', '初沉池', '1-2', length_m=4, width_m=3, depth_m=2, volume_m3=24, basis='plan_geometry', formula_row='F004'),
                row(2, '初沉池(1-2)', 'effective_volume', 24, 'plan_dims', '初沉池', '1-2', 'm3', length_m=4, width_m=3, depth_m=2, volume_m3=24, basis='plan_geometry', formula_row='F004'),
                row(3, '初沉池(1-2)', 'effective_volume', 40, 'calc', '初沉池', '1-2', 'm3')]
        for question in ('平面圖初沉池(1-2)的尺寸？', '平面圖初沉池(1-2)有效容積？', '平面圖初沉池(1-2)約多少？'):
            with self.subTest(question=question):
                result = factquery.context(question, rows)
                self.assertIn('F001', [r['id'] for r in result['rows']])
                self.assertIn('F002', [r['id'] for r in result['rows']])
                self.assertIn('measured plan volume for 初沉池(1-2): 24 m3 (F002)', result['computed'])
                self.assertNotIn('the drawings give two values', '\n'.join(result['computed']))
                self.assertIn('| plan_dims:', result['text'])


if __name__ == '__main__':
    unittest.main()
