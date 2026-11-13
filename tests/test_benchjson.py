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
        q = {"問題": "尺寸計算", "答案": "約 1,000 立方公尺", "答案狀態": "有依據", "類別": "設計基礎"}
        self.assertEqual(grade(q, "約1000 m3", [{"id": "p1"}]), "pass")
        self.assertEqual(grade(q, "約 999 m³", [{"id": "p1"}]), "pass")
        self.assertEqual(grade({**q, "問題": "一般問題"}, "約 999 m3", [{"id": "p1"}]), "fail")
        self.assertEqual(grade({**q, "答案": "1,200 HP"}, "1200hp", [{"id": "p1"}]), "pass")

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
