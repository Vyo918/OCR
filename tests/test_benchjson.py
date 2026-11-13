import json
import tempfile
import unittest
from pathlib import Path

from docsearch.benchjson import grade, select_questions, run


class BenchJsonTests(unittest.TestCase):
    def test_subset_is_deterministic_and_includes_mandatory_questions(self):
        questions = [
            {"編號": f"Q{i:03}", "類別": "A" if i % 2 else "B", "難度": "簡單", "答案狀態": "有依據"}
            for i in range(1, 9)
        ]
        questions[0]["答案狀態"] = "圖面未標示"
        questions[1]["難度"] = "整合"
        self.assertEqual(select_questions(questions, 3, 42), select_questions(questions, 3, 42))
        ids = {q["編號"] for q in select_questions(questions, 3, 42)}
        self.assertTrue({"Q001", "Q002"} <= ids)

    def test_numeric_grading(self):
        q = {"問題": "尺寸計算", "驗證方式": "尺寸計算", "答案": "約 1,000 立方公尺", "答案狀態": "有依據", "類別": "設計基礎"}
        self.assertEqual(grade(q, "約1000 m3", [{"id": "p1"}]), "pass")
        self.assertEqual(grade(q, "約 999 m³", [{"id": "p1"}]), "pass")
        self.assertEqual(grade({**q, "問題": "一般問題", "驗證方式": ""}, "約 999 m3", [{"id": "p1"}]), "fail")
        self.assertEqual(grade({**q, "答案": "1,200 HP"}, "1200hp", [{"id": "p1"}]), "pass")

    def test_unit_synonyms_and_mismatches(self):
        q = {"答案": "425 CMD", "答案狀態": "有依據"}
        self.assertEqual(grade(q, "425 m³／日", [{"id": "p1"}]), "pass")
        q = {"答案": "12 小時", "答案狀態": "有依據"}
        self.assertEqual(grade(q, "12 hours", [{"id": "p1"}]), "pass")
        self.assertEqual(grade(q, "12 HP", [{"id": "p1"}]), "fail")

    def test_tolerance_uses_validation_method(self):
        q = {"問題": "一般問尺寸計算", "答案": "100", "答案狀態": "有依據"}
        self.assertEqual(grade(q, "99", [{"id": "p1"}]), "fail")
        q["驗證方式"] = "尺寸計算"
        self.assertEqual(grade(q, "99", [{"id": "p1"}]), "pass")

    def test_chat_cli_help(self):
        from docsearch.cli import main
        with self.assertRaises(SystemExit) as exc:
            main(["chat", "--help"])
        self.assertEqual(exc.exception.code, 0)

    def test_refusal_and_conflict_grading(self):
        q = {"問題": "數量", "答案": "未標示", "答案狀態": "圖面未標示"}
        self.assertEqual(grade(q, "圖面未顯示此資訊", [{"id": "p1"}]), "pass")
        self.assertEqual(grade(q, "圖面未標示，應為 3 台", [{"id": "p1"}]), "review")
        self.assertEqual(grade({**q, "答案": "10 HP", "答案狀態": "圖面存在差異"}, "10 HP", [{"id": "p1"}]), "review")

    def test_run_resume_and_error_records(self):
        questions = [{"編號": f"Q{i}", "問題": f"q{i}", "答案": "1", "答案狀態": "有依據", "類別": "A", "難度": "簡單"} for i in range(3)]
        class Engine:
            def __init__(self): self.calls = []
            def answer(self, question):
                self.calls.append(question)
                if question == "q1": raise RuntimeError("broken")
                return {"answer": "1", "evidence": [{"id": "p1"}]}
        with tempfile.TemporaryDirectory() as td:
            source, out = Path(td) / "input.json", Path(td) / "out"
            source.write_text(json.dumps({"題目": questions}), encoding="utf-8")
            engine = Engine()
            results = run(source, engine, out=out)
            self.assertEqual(len(results), 3)
            lines = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            self.assertEqual([x["grade"] for x in lines], ["pass", "error", "pass"])
            resumed = Engine()
            run(source, resumed, out=out, resume=True)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
