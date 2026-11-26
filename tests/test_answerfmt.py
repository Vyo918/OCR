import os
import re
import unittest
from unittest.mock import patch

from docsearch import answerfmt
from docsearch.search import Engine


class AnswerFormatTests(unittest.TestCase):
    def test_phase_from_matching_row(self):
        self.assertEqual(answerfmt.normalize("設備為 380v 50hz。", "電源規格？", ["電源 3 380V 50HZ"]),
                         "設備為 三相 380V 50Hz。")

    def test_existing_phase_and_mismatches_are_untouched(self):
        for answer, question, contexts in [
            ("三相 380V 50Hz", "電壓？", ["電源 3 380V 50HZ"]),
            ("380V 50Hz", "電源？", ["電源 1 400V 50HZ"]),
            ("380V 50Hz", "電源？", ["電源 1 380V 60HZ"]),
            ("380V 50Hz", "尺寸？", ["電源 1 380V 50HZ"]),
            ("380V 50Hz", "電源？", ["電源 1 380V 50HZ", "電源 3 380V 50HZ"]),
        ]:
            with self.subTest(answer=answer, contexts=contexts):
                self.assertEqual(answerfmt.normalize(answer, question, contexts), answer)

    def test_unit_format_preserves_numbers(self):
        answer = "380 v 50 hz，容量 12.5 M3 和 8 M³。"
        self.assertEqual(answerfmt.normalize(answer, "規格？", []),
                         "380V 50Hz，容量 12.5 m3 和 8 m³。")
        self.assertEqual(answerfmt.normalize("型號 380v2", "規格？", []), "型號 380v2")

    def test_switch_off_preserves_pipeline_result(self):
        engine = Engine.__new__(Engine)
        engine.prepare = lambda question, top: {"queries": [], "cand": [], "best": [], "ev": [], "fx": None}
        engine.build_prompt = lambda *args: ("system", "user")
        engine.chat = lambda *args: ("380v 50hz", "reason")
        with patch.dict(os.environ, {"DOCSEARCH_ANSWER_FORMAT": "0"}), \
             patch("docsearch.search.config.REREAD", "0"), patch("docsearch.search.config.VERIFY", False):
            result = engine.answer("電源？")
        self.assertEqual(result["answer"], "380v 50hz")
        self.assertNotIn("answer_raw", result)

    def test_switch_on_formats_and_retains_raw_answer(self):
        engine = Engine.__new__(Engine)
        chunk = {"id": "a", "doc": "d", "bbox": [], "body": "電源 3 380V 50HZ", "text": "電源 3 380V 50HZ"}
        engine.prepare = lambda question, top: {"queries": [], "cand": [], "best": [0], "ev": [chunk], "fx": None}
        engine.build_prompt = lambda *args: ("system", "user")
        engine.chat = lambda *args: ("380v 50hz", "reason")
        with patch.dict(os.environ, {"DOCSEARCH_ANSWER_FORMAT": "1"}), \
             patch("docsearch.search.config.REREAD", "0"), patch("docsearch.search.config.VERIFY", False):
            result = engine.answer("電源？")
        self.assertEqual(result["answer"], "三相 380V 50Hz")
        self.assertEqual(result["answer_raw"], "380v 50hz")

    def test_tank_size_is_decoded_for_each_size(self):
        answer = "調整池 (1)L500×W200.5×H310/D280 與 (2)L600 x W240 X H300 / D250"
        result = answerfmt.normalize(answer, "調整池尺寸？", [])
        self.assertEqual(result, "調整池 (1)L500×W200.5×H310/D280（長 500 cm、寬 200.5 cm、池高 310 cm、有效水深 280 cm） 與 (2)L600 x W240 X H300 / D250（長 600 cm、寬 240 cm、池高 300 cm、有效水深 250 cm）")
        self.assertEqual(re.sub(r"（[^）]*）", "", result), answer)

    def test_complete_or_unasked_tank_size_is_untouched(self):
        complete = "(1)L500×W200.5×H310/D280（長 500 cm、寬 200.5 cm、池高 310 cm、有效水深 280 cm）"
        for answer, question in [(complete, "調整池尺寸？"), ("(1)L500×W200.5×H310/D280", "調整池容量？")]:
            with self.subTest(answer=answer, question=question):
                self.assertEqual(answerfmt.normalize(answer, question, []), answer)

    def test_bare_voltage_gets_phase_and_frequency_from_one_row(self):
        self.assertEqual(answerfmt.normalize("設備電壓為 220v。", "電壓規格？", ["電源 1 220V 60HZ"]),
                         "設備電壓為 單相 220V 60Hz。")

    def test_ambiguous_bare_voltage_is_untouched(self):
        for answer, question, contexts in [
            ("220V", "電壓？", ["電源 1 220V 60HZ", "電源 3 220V 60HZ"]),
            ("220V", "電壓？", ["電源 1 220V 50HZ", "電源 1 220V 60HZ"]),
            ("220V", "相數？", ["電源 1 220V 60HZ"]),
            ("220V 和 380V", "電壓？", ["電源 1 220V 60HZ"]),
            ("220V", "電壓？", ["電源 1 380V 60HZ"]),
        ]:
            with self.subTest(answer=answer, question=question, contexts=contexts):
                self.assertEqual(answerfmt.normalize(answer, question, contexts), answer)

    def test_limit_gets_comparator_from_evidence(self):
        for question, contexts, answer, expected in [
            ("排放上限？", ["XY<30mg/L"], "30 mg/L", "小於 30 mg/L"),
            ("排放上限？", ["XY ≤ 30mg/L"], "30 mg/L", "不大於 30 mg/L"),
            ("排放下限？", ["XY>5 mg/L"], "5 mg/L", "大於 5 mg/L"),
            ("排放下限？", ["XY≧5mg/L"], "5 mg/L", "不小於 5 mg/L"),
        ]:
            with self.subTest(question=question, contexts=contexts):
                result = answerfmt.normalize(answer, question, contexts)
                self.assertEqual(result, expected)
                self.assertEqual(re.findall(r"\d+(?:\.\d+)?", result), re.findall(r"\d+(?:\.\d+)?", answer))

    def test_ambiguous_limit_is_untouched(self):
        for question, contexts, answer in [
            ("排放上限？", ["XY<30mg/L", "XY>30mg/L"], "30 mg/L"),
            ("排放上限？", ["XY<30NTU"], "30 mg/L"),
            ("排放上限？", ["XY 30 mg/L"], "30 mg/L"),
            ("排放上限？", ["XY<130mg/L"], "30 mg/L"),
            ("排放上限？", ["XY<30mg/L"], "30 mg/L 與 5 mg/L"),
            ("排放上限？", ["XY<30mg/L"], "小於 30 mg/L"),
            ("排放量？", ["XY<30mg/L"], "30 mg/L"),
        ]:
            with self.subTest(question=question, contexts=contexts, answer=answer):
                self.assertEqual(answerfmt.normalize(answer, question, contexts), answer)


if __name__ == "__main__":
    unittest.main()
