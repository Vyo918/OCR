import os
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


if __name__ == "__main__":
    unittest.main()
