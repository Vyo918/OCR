import os
import unittest
from unittest import mock

from docsearch import absent
from docsearch.search import Engine


def chunk(i, *lines):
    text = "\n".join(lines)
    return {"id": f"D#{i}", "doc": "D", "kind": "block", "bbox": [0, 10 * i, 20, 10 * i + 10], "section": "",
            "body": text, "text": text, "lines": [{"text": t, "bbox": [0, 0, 1, 1]} for t in lines]}


CHUNKS = [
    chunk(0, "消防水池", "污水泵B", "洪放流泵"),
    chunk(1, "消防蓄水池", "調整池泵浦", "终沉池泉浦"),
    chunk(2, "放流池泵浦", "调整池立式鲁氏鼓风机", "除臭抽风机", "NaOCL加藥機", "螺旋式流量計"),
    chunk(3, "初沉池", "連通管 4\"PVC x2支"),
]

# A location and its equipment word on one OCR line, with the location's characters misread or dropped on the drawing.
LINE_CHUNKS = [
    chunk(0, "調整池立式魯氏鼓風機", "觸曝氣池立式魯氏鼓風機"),
    chunk(1, "調整池泵浦"),
]

# 消防 and 泵 are both in this chunk, but on different lines.
SPLIT_CHUNKS = [chunk(0, "消防水池", "洪放流泵")]


class CheckTests(unittest.TestCase):
    def test_absent_equipment_is_reported_with_nearest_lines(self):
        hit = absent.check("消防泵浦幾HP?", CHUNKS)
        self.assertEqual(hit["term"], "消防泵浦")
        lines = [line for line, _ in hit["nearest"]]
        self.assertIn("消防水池", lines)
        self.assertIn("消防蓄水池", lines)
        self.assertEqual(hit["nearest"][0][1]["id"], "D#0")

    def test_leading_question_words_are_dropped_from_the_term(self):
        self.assertEqual(absent.check("請問消防泵浦幾HP?", CHUNKS)["term"], "消防泵浦")

    def test_present_equipment_is_not_absent(self):
        for question in ["潛洪放流泵的電源?",           # OCR dropped 潛 from 潛洪放流泵
                         "請問調整池泵浦幾台?",
                         "污水泵B的電源規格?",
                         "調整池立式魯氏鼓風機馬力?",    # simplified spelling in the chunk
                         "调整池立式鲁氏鼓风机马力?",
                         "終沉池泵浦的揚程?",            # one misread character (泉 for 泵)
                         "除臭抽風機幾HP?",              # traditional question, simplified chunk
                         "NaOCL加藥機幾台?",
                         ]:
            with self.subTest(question=question):
                self.assertIsNone(absent.check(question, CHUNKS))

    def test_location_and_equipment_on_one_line_are_present(self):
        for question in ["調整池鼓風機的馬力?",            # 立式魯氏 between the location and the word on the line
                         "接觸曝氣池鼓風機的風量?",         # OCR dropped 接 from the line
                         "調整池選用的泵浦有幾台?",
                         "調整池所選泵浦數量?"]:
            with self.subTest(question=question):
                self.assertIsNone(absent.check(question, LINE_CHUNKS))

    def test_location_and_equipment_on_different_lines_are_absent(self):
        for question in ["消防泵浦的馬力?", "消防泵浦的數量?"]:
            with self.subTest(question=question):
                self.assertEqual(absent.check(question, SPLIT_CHUNKS)["term"], "消防泵浦")
        self.assertEqual(absent.check("消防泵浦的馬力?", LINE_CHUNKS + SPLIT_CHUNKS)["term"], "消防泵浦")

    def test_term_is_the_longest_head_of_at_most_four_characters(self):
        self.assertEqual(absent.check("調整池鼓風機的馬力?", SPLIT_CHUNKS)["term"], "調整池鼓風機")
        self.assertEqual(absent.check("請問污水處理池甲乙泵的流量?", SPLIT_CHUNKS)["term"], "理池甲乙泵")

    def test_questions_without_equipment_are_never_absent(self):
        for question in ["初沉至調整池連通管數量?", "初沉池幾槽?"]:
            with self.subTest(question=question):
                self.assertIsNone(absent.check(question, CHUNKS))

    def test_bare_suffix_is_not_a_name(self):
        self.assertIsNone(absent.check("泵浦幾台?", CHUNKS))

    def test_multiple_mentions_report_the_first_absent_one(self):
        hit = absent.check("調整池泵浦和消防泵浦各幾台?", CHUNKS)
        self.assertEqual(hit["term"], "消防泵浦")


class EngineGuardTests(unittest.TestCase):
    def engine(self):
        e = Engine.__new__(Engine)
        e.chunks = CHUNKS
        return e

    def test_absent_question_returns_not_found_without_retrieval(self):
        e = self.engine()
        with mock.patch.dict(os.environ, {"DOCSEARCH_ABSENT_GUARD": "1"}), \
                mock.patch.object(e, "prepare", side_effect=AssertionError("retrieval ran")), \
                mock.patch.object(e, "chat", side_effect=AssertionError("LLM ran")):
            r = e.answer("消防泵浦幾HP?")
        self.assertTrue(r["answer"].startswith("NOT FOUND"))
        self.assertIn("消防泵浦", r["answer"])
        self.assertIn("消防水池", r["answer"])
        self.assertEqual([c["id"] for c in r["evidence"]][:1], ["D#0"])
        self.assertEqual(r["absent"]["term"], "消防泵浦")

    def test_present_question_goes_on_to_retrieval(self):
        e = self.engine()
        with mock.patch.dict(os.environ, {"DOCSEARCH_ABSENT_GUARD": "1"}), \
                mock.patch.object(e, "prepare", side_effect=RuntimeError("retrieval ran")):
            with self.assertRaisesRegex(RuntimeError, "retrieval ran"):
                e.answer("污水泵B的電源規格?")

    def test_switch_off_disables_the_guard(self):
        e = self.engine()
        with mock.patch.dict(os.environ, {"DOCSEARCH_ABSENT_GUARD": "0"}), \
                mock.patch.object(e, "prepare", side_effect=RuntimeError("retrieval ran")):
            with self.assertRaisesRegex(RuntimeError, "retrieval ran"):
                e.answer("消防泵浦幾HP?")


if __name__ == "__main__":
    unittest.main()
