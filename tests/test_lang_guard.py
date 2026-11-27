import os
import unittest
from unittest import mock

from docsearch import config
from docsearch.search import Engine


CHUNK = {"id": "D#0", "doc": "D", "kind": "block", "bbox": [0, 0, 20, 10], "body": "初沉池 3 槽", "text": "初沉池 3 槽", "section": ""}
ZH_QUESTION = "初沉池幾槽?體積各多少?"
EN_QUESTION = "How many primary sedimentation tanks are there?"
EN_ANSWER = "There are 3 tanks, 89.11 m3 each [00#585]."
ZH_ANSWER = "初沉池有3槽，每槽89.11 m3 [00#585]。"


class LangGuardTests(unittest.TestCase):
    def engine(self):
        e = Engine.__new__(Engine)
        e.chunks = [CHUNK]
        e.outlines = {"D": ["outline"]}
        return e

    def run_answer(self, question, answers, env=None):
        e = self.engine()
        prepared = {"queries": [], "cand": [0], "best": [0], "ev": [CHUNK], "fx": None}
        overrides = {"DOCSEARCH_ABSENT_GUARD": "0", "DOCSEARCH_LANG_GUARD": "1", **(env or {})}
        with mock.patch.dict(os.environ, overrides), mock.patch.object(config, "REREAD", "0"), \
                mock.patch.object(config, "VERIFY", False), mock.patch.object(e, "prepare", return_value=prepared), \
                mock.patch.object(e, "chat", side_effect=answers) as chat:
            result = e.answer(question)
        return result, chat

    def test_english_answer_to_chinese_question_is_translated(self):
        result, chat = self.run_answer(ZH_QUESTION, [(EN_ANSWER, "r"), (ZH_ANSWER, "")])
        self.assertEqual(chat.call_count, 2)
        self.assertIn("Traditional Chinese", chat.call_args.args[0])
        self.assertEqual(chat.call_args.args[1], EN_ANSWER)
        self.assertEqual(result["answer"], ZH_ANSWER)
        self.assertEqual(result["answer_raw"], EN_ANSWER)
        self.assertEqual(result["lang_guard"], {"triggered": True, "accepted": True})

    def test_translation_that_changes_a_digit_keeps_the_original(self):
        result, chat = self.run_answer(ZH_QUESTION, [(EN_ANSWER, "r"), ("初沉池有3槽，每槽89.12 m3 [00#585]。", "")])
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["answer"], EN_ANSWER)
        self.assertEqual(result["lang_guard"], {"triggered": True, "accepted": False})

    def test_chinese_answer_makes_no_extra_call(self):
        result, chat = self.run_answer(ZH_QUESTION, [(ZH_ANSWER, "")])
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["answer"], ZH_ANSWER)
        self.assertNotIn("lang_guard", result)

    def test_english_question_makes_no_extra_call(self):
        result, chat = self.run_answer(EN_QUESTION, [(EN_ANSWER, "")])
        self.assertEqual(chat.call_count, 1)
        self.assertNotIn("lang_guard", result)

    def test_switch_off_makes_no_extra_call(self):
        result, chat = self.run_answer(ZH_QUESTION, [(EN_ANSWER, "")], env={"DOCSEARCH_LANG_GUARD": "0"})
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["answer"], EN_ANSWER)
        self.assertNotIn("lang_guard", result)


if __name__ == "__main__":
    unittest.main()
