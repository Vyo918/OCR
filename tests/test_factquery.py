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

    def test_pool_plan_volume_and_scoped_family_count(self):
        rows = [row(i, f'初沉池(2-{i})', 'label_depth', 2, 'plan_label', '初沉池', f'2-{i}', 'm')
                for i in range(1, 4)]
        rows.append(row(4, '抽水池(2-4)', 'label_depth', 2, 'plan_label', '抽水池', '2-4', 'm'))
        rows.append(row(5, '抽水池(2-4)', 'effective_volume', 24, 'plan_dims', '抽水池', '2-4', 'm3'))
        plan = row(6, '處理槽', 'plan_volume', 96, 'plan_volume',
                   members=['F001', 'F002', 'F003', 'F004'])
        plan['section'] = '污水處理槽(2)平面圖'
        rows.append(plan)
        result = factquery.context('抽水池(2-4)平面圖尺寸及有效容積？', rows)
        self.assertEqual([r['id'] for r in result['rows']], ['F005'])
        self.assertIn('measured plan volume for 抽水池(2-4): 24 m3 (F005)', result['computed'])
        count = factquery.context('污水處理槽(2)平面圖初沉池共有幾槽？', rows)
        self.assertIn('plan 污水處理槽(2)平面圖 labels by name: 3 [2-1, 2-2, 2-3]', count['computed'])

    def test_family_count_distinguishes_numbering_and_names(self):
        rows = [row(i, '初沉池', 'tank_dims', None, family='初沉池', index=f'({i})') for i in range(1, 4)]
        rows += [row(i+3, f'初沉池(1-{i})', 'label_depth', 1.85, 'plan_label', '初沉池', f'1-{i}', 'm') for i in range(1, 7)]
        rows.append(row(10, '抽水池(1-7)', 'label_depth', 1.85, 'plan_label', '抽水池', '1-7', 'm'))
        result = factquery.context('初沉池共幾槽？', rows)
        self.assertEqual(len(result['computed']), 3)
        self.assertIn('3 [', result['computed'][0])
        self.assertIn('6 [', result['computed'][1])
        self.assertIn('7 [', result['computed'][2])

    def test_lookup_ranks_entity_named_in_question(self):
        rows = [row(i, '調整池', 'water_quality', float(i), family='調整池', unit='mg/L') for i in range(1, 31)]
        rows.append(row(31, '調整池填料', 'qty', 14490, 'legend', '調整池', unit='m'))
        result = factquery.context('調整池填料長度是多少?', rows)
        self.assertIn('F031', [r['id'] for r in result['rows']])

    def test_pipe_question_gets_no_tank_counts(self):
        rows = [row(1, '調整池', 'tank_dims', None, family='調整池', index='(1)'),
                row(2, '調整池', 'tank_dims', None, family='調整池', index='(2)'),
                row(3, '調整池(2-1)', 'label_depth', 3.5, 'plan_label', '調整池', '2-1', 'm')]
        self.assertEqual(factquery.context('調整池至沉砂池的連通管幾支?', rows)['computed'], [])
        self.assertIn('calc tanks: 2', '\n'.join(factquery.context('調整池共幾槽?', rows)['computed']))

    def test_fact_line_shows_group_index_and_tank_dims_in_cm(self):
        rows = [row(1, '調整池', 'tank_dims', None, family='調整池', index='(1)', L=7.9, W=4.675, H=3.5, D=3.15, volume=116.34),
                row(2, '調整池', 'tank_dims', None, family='調整池', index='(2)'),
                row(3, '調整池', 'effective_depth', 3.15, family='調整池', index='(1)', unit='m')]
        text = factquery.context('調整池(1)的設計尺寸和有效水深是多少？', rows)['text']
        self.assertIn('[F001] 調整池 (1) | tank_dims = 長 790 cm、寬 467.5 cm、池高 350 cm、有效水深 315 cm（有效容積 L×W×D = 116.34 m3） | ', text)
        self.assertIsNone(factquery.context('調整池(2)的設計尺寸是多少？', rows))
        self.assertIn('[F003] 調整池 (1) | effective_depth = 3.15 m | ', text)
        pump = [row(4, '污水泵B', 'hp', 2, 'pump_table', index='B', unit='HP')]
        self.assertIn('[F004] 污水泵B | hp = 2 HP | ', factquery.context('污水泵B的馬力是多少？', pump)['text'])

    def volume_rows(self):
        rows = [row(i, '初沉池', 'tank_dims', None, family='初沉池', index=f'({i})', L=7.9, W=3.52, H=3.2, D=2.9, volume=80.61) for i in range(1, 4)]
        for i, index, depth, volume in [(4, '1-4', 1.85, 51.4448), (6, '1-5', 2.457, 68.325125), (8, '1-6', 1.85, 51.4448)]:
            dims = dict(length_m=7.9, width_m=3.52, depth_m=depth, volume_m3=volume)
            rows.append(row(i, f'初沉池({index})', 'tank_dims', None, 'plan_dims', '初沉池', index, 'm', **dims))
            rows.append(row(i + 1, f'初沉池({index})', 'effective_volume', volume, 'plan_dims', '初沉池', index, 'm3', **dims))
        rows += [row(10 + n, f'初沉池(1-{n})', 'label_depth', 1.85, 'plan_label', '初沉池', f'1-{n}', 'm') for n in range(1, 7)]
        return rows

    def test_mixed_count_and_volume_keeps_per_tank_size_rows(self):
        result = factquery.context('初沉池幾槽?體積各多少?', self.volume_rows())
        kinds = {(r['source'], r['attr']) for r in result['rows']}
        self.assertLessEqual({('calc', 'tank_dims'), ('plan_dims', 'tank_dims'), ('plan_dims', 'effective_volume')}, kinds)
        self.assertIn('51.44', result['text'])
        self.assertIn('68.33', result['text'])
        self.assertNotIn('=  |', result['text'])

    def test_per_tank_volumes_reconcile_to_labels_by_name(self):
        computed = factquery.context('初沉池幾槽?體積各多少?', self.volume_rows())['computed']
        self.assertEqual(computed[0], 'per-tank volumes: calc (1) 80.61 m3 [F001], (2) 80.61 m3 [F002], (3) 80.61 m3 [F003]; '
                                      'plan-measured 1-4 51.44 m3 [F005], 1-5 68.33 m3 [F007], 1-6 51.44 m3 [F009]')
        self.assertEqual(computed[1], 'sized tanks: 3 by calc sheet + 3 by plan measurement = 6, matching the 6 labels by name')
        self.assertEqual(computed[2:], ['calc tanks: 3 [(1), (2), (3)]', 'labels by name: 6 [1-1, 1-2, 1-3, 1-4, 1-5, 1-6]',
                                        'label numbering series: 6 [1-1, 1-2, 1-3, 1-4, 1-5, 1-6]'])

    def test_sized_mismatch_reports_each_source(self):
        rows = [r for r in self.volume_rows() if not (r['source'] == 'plan_label' and r['index'] == '1-6')]
        computed = factquery.context('初沉池幾槽?體積各多少?', rows)['computed']
        self.assertEqual(computed[1], 'calc sheet sizes 3 tanks, plan measurement sizes 3, labels by name 5: report each source')

    def test_pure_count_computed_lines_unchanged(self):
        computed = factquery.context('初沉池共幾槽？', self.volume_rows())['computed']
        self.assertEqual(computed, ['calc tanks: 3 [(1), (2), (3)]', 'labels by name: 6 [1-1, 1-2, 1-3, 1-4, 1-5, 1-6]',
                                    'label numbering series: 6 [1-1, 1-2, 1-3, 1-4, 1-5, 1-6]'])

    def test_pure_count_selects_count_rows_only(self):
        rows = self.volume_rows()
        result = factquery.context('初沉池共幾槽？', rows)
        self.assertEqual(sorted(r['id'] for r in result['rows']), sorted(r['id'] for r in rows if r['attr'] in ('tank_dims', 'label_depth')))

    def test_plan_tank_dims_renders_size_and_volume(self):
        dims = next(r for r in self.volume_rows() if r['id'] == 'F004')
        text = factquery._v(dims)
        self.assertIn('長 7.9 m', text)
        self.assertIn('51.44 m3', text)

    def test_size_rows_survive_the_row_cap(self):
        rows = self.volume_rows() + [row(100 + n, f'初沉池(1-{n})', 'label_depth', 1.85, 'plan_label', '初沉池', f'1-{n}', 'm') for n in range(7, 40)]
        result = factquery.context('初沉池幾槽?體積各多少?', rows)
        self.assertEqual(len(result['rows']), 24)
        self.assertLessEqual({f'F{i:03}' for i in range(1, 10)}, {r['id'] for r in result['rows']})
