import unittest

from docsearch import factquery


def row(i, entity, attr, value, source='calc', family=None, index=None, unit='', **extra):
    return dict(id=f'F{i:03}', doc='drawing', source=source, section='', entity=entity,
                family=family, equipment='泵浦' if '泵' in entity else None,
                index=index, ekey=entity, attr=attr, label=attr, op=None,
                value=value, unit=unit, value_si=value, unit_si=unit,
                text='printed', src=[dict(chunk=f'drawing#{i}', bbox=[])], extra=extra)


class FactQueryTests(unittest.TestCase):
    def test_index_and_fallback(self):
        rows = [row(1, '初沉池', 'effective_depth', 3.2, family='初沉池', index='(1)', unit='m'),
                row(2, '初沉池(1-5)', 'label_depth', 1.85, 'plan_label', '初沉池', '1-5', 'm')]
        result = factquery.context('初沉池(1-5)的有效水深是多少？', rows)
        self.assertEqual([r['id'] for r in result['rows']], ['F002'])
        self.assertIn('drawing#2', result['text'])
        self.assertIsNone(factquery.context('消防泵浦幾HP?', rows))
        self.assertIsNone(factquery.context('筏基混凝土強度?', rows))

    def test_pump_comparison_and_conflict(self):
        rows = [row(1, '污水泵B', 'hp', 2, 'pump_table', index='B', unit='HP'),
                row(2, '污水泵C', 'hp', 5, 'pump_table', index='C', unit='HP'),
                row(3, '污水泵B', 'hp', 1, 'plan_label', index='B', unit='HP'),
                row(4, '污水泵C', 'hp', 5, 'plan_label', index='C', unit='HP')]
        result = factquery.context('污水泵 B 與污水泵 C 哪一種馬力較大？', rows)
        self.assertEqual(len(result['rows']), 4)
        text = '\n'.join(result['computed'])
        self.assertIn('pump_table', text)
        self.assertIn('plan_label', text)
        self.assertIn('F001', text)
        self.assertIn('F003', text)

    def test_person_check(self):
        rows = [row(i, name, 'persons', val, unit='person') for i, (name, val) in enumerate(
            [('店鋪', 82), ('圖書館', 62), ('關懷據點', 5), ('日間照護', 96), ('幼兒園', 17), ('集合住宅', 1338)], 1)]
        rows.append(row(7, '', 'persons', 1552, unit='person'))
        result = factquery.context('計算書列出的各用途使用人數逐項相加是否一致？', rows)
        self.assertIn('1600', '\n'.join(result['computed']))
        self.assertIn('1552', '\n'.join(result['computed']))

    def test_soft_qualifier(self):
        rows = [row(1, '接觸曝氣池細氣泡散氣盤', 'qty', 76, 'legend', '接觸曝氣池', unit='count')]
        self.assertEqual(factquery.context('接觸曝氣池所配置的總粗氣泡散氣盤數量是多少？', rows)['rows'][0]['value'], 76)

    def test_pump_name_does_not_match_unit_synonym(self):
        rows = [row(1, '洪放流泵', 'head', 17, 'pump_table', unit='m'),
                row(2, '放流池泵浦', 'head', 18, 'calc', '放流池', unit='m')]
        result = factquery.context('潛洪放流泵的揚程要求是多少？', rows)
        self.assertEqual([r['id'] for r in result['rows']], ['F001'])

    def test_volume_members_and_no_cross_source_sum(self):
        rows = [row(1, '初沉池(1-1)', 'label_depth', 3.5, 'plan_label', '初沉池', '1-1', 'm'),
                row(2, '初沉池', 'effective_volume', 223.12, family='初沉池', unit='m3'),
                row(3, '處理槽', 'plan_volume', 239.5, 'plan_volume', unit='m3', members=['F001'])]
        result = factquery.context('兩區初沉池依圖面列出的容積合計約多少？', rows)
        self.assertEqual([r['id'] for r in result['rows']], ['F002', 'F003'])
        self.assertEqual(result['computed'], [])

    def test_family_count_distinguishes_numbering_and_names(self):
        rows = [row(i, '初沉池', 'tank_dims', None, family='初沉池', index=f'({i})') for i in range(1, 4)]
        rows += [row(i+3, f'初沉池(1-{i})', 'label_depth', 1.85, 'plan_label', '初沉池', f'1-{i}', 'm') for i in range(1, 7)]
        rows.append(row(10, '抽水池(1-7)', 'label_depth', 1.85, 'plan_label', '抽水池', '1-7', 'm'))
        result = factquery.context('初沉池共幾槽？', rows)
        self.assertEqual(len(result['computed']), 3)
        self.assertIn('3 [', result['computed'][0])
        self.assertIn('6 [', result['computed'][1])
        self.assertIn('7 [', result['computed'][2])
