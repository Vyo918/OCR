import unittest

from docsearch.entities import anchor, chunk_entities, question_entities


class EntityTest(unittest.TestCase):
    def test_unit_notation_and_exact_ids(self):
        for text in ("初沉池(1-5)", "初沉池 1-5", "初沉池（１－５）", "初沉池\n(1–5)", "初沉池1-5)"):
            with self.subTest(text=text):
                self.assertIn(("初沉池", "1-5"), question_entities(text))
        for text in ("初沉池(1-4)", "初沉池(1-15)", "初沉池(11-5)"):
            self.assertNotIn(("初沉池", "1-5"), chunk_entities(text))
        self.assertIn(("污水泵", "C"), question_entities("污水泵 C 的口徑"))
        self.assertIn(("處理槽", "1"), question_entities("處理槽(1)"))
        self.assertIn(("組", "3"), question_entities("第３組"))

    def test_real_chunk_excerpts(self):
        cases = [
            ("初沉池1-5 H=1.85m", ("初沉池", "1-5")),
            ("初沉池(1-4)", ("初沉池", "1-4")),
            ("初沉池1-6)\nH=1.85m", ("初沉池", "1-6")),
            ("污水泵B规格 B規格", ("污水泵", "B")),
            ("WP(污水泵C", ("污水泵", "C")),
            ("調整池(2-2)\nH=3.5m", ("調整池", "2-2")),
            ("X14污水處理槽(1)平面圖X15 X15", ("處理槽", "1")),
            ("接觸曝氣池細氣泡散氣盤", ("接觸曝氣池", None)),
            ("調整池粗氣泡散氣盤", ("調整池", None)),
            ("消毒池粗氣泡散氣盤", ("消毒池", None)),
            ("1. 初沉池\nB.規劃計算： 設計尺寸：\n(1) L790×W352.5×H350/ D320\n(2) L790×W352.5×H350/ D320", ("初沉池", "2")),
        ]
        for text, entity in cases:
            with self.subTest(text=text):
                self.assertIn(entity, chunk_entities(text))

    def test_anchor_replaces_only_lowest_nonmatches(self):
        chunks = [{"text": s} for s in ("初沉池(1-4)", "初沉池(1-5)", "其他", "初沉池1-5)", "初沉池（1－5）", "初沉池(1-5)")]
        self.assertEqual(anchor("初沉池(1-5)深度", [0, 1, 2], [0, 1, 3, 4, 5, 2], chunks, 3), [1, 3, 4])
        self.assertEqual(anchor("初沉池(1-5)深度", [0, 2], [0, 3, 2], chunks, 2), [0, 3])
        self.assertEqual(anchor("初沉池(1-15)深度", [0, 2], [0, 3, 2], chunks, 2), [0, 2])
        self.assertEqual(anchor("初沉池深度", [0, 2], [0, 3, 2], chunks, 2), [0, 2])

    def test_ordinal_group_matches_numbered_calculation_line(self):
        chunks = [{"text": "其他"}, {"text": "1. 初沉池\n(3) L790×W352.5×H350/ D320"}]
        self.assertEqual(anchor("第3組尺寸", [0], [0, 1], chunks, 1), [1])


if __name__ == "__main__":
    unittest.main()
