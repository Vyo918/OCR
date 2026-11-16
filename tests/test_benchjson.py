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
        q = {"答案": "301.05 m³／日", "答案狀態": "有依據"}
        for answer in ("301.05 m³/天", "301.05 m3/day", "301.05 立方公尺/日", "301.05 m3/d"):
            with self.subTest(answer=answer):
                self.assertEqual(grade(q, answer, [{"id": "p1"}]), "pass")
        self.assertEqual(grade(q, "301.05 m3/hr", [{"id": "p1"}]), "fail")

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
                return {"answer": "1", "evidence": [{"id": "p1"}], "facts": [{"id": "F001"}], "computed": ["count = 1"]}
        with tempfile.TemporaryDirectory() as td:
            source, out = Path(td) / "input.json", Path(td) / "out"
            source.write_text(json.dumps({"題目": questions}), encoding="utf-8")
            engine = Engine()
            results = run(source, engine, out=out)
            self.assertEqual(len(results), 3)
            lines = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            self.assertEqual([x["grade"] for x in lines], ["pass", "error", "pass"])
            self.assertEqual(lines[0]["facts"], ["F001"])
            self.assertEqual(lines[0]["computed"], ["count = 1"])
            self.assertIn("facts_mode", lines[0])
            self.assertEqual(lines[1]["facts"], [])
            self.assertIn(" facts=", (out / "summary.md").read_text())
            resumed = Engine()
            run(source, resumed, out=out, resume=True)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 3)

    def test_real_case_grading(self):
        ev = [{"id": "p1"}]
        count = {"問題": "終沉池泵浦數量與馬力？", "答案": "2 台，每台 1 HP", "答案狀態": "有依據"}
        self.assertEqual(grade(count, "終沉池泵浦共有2台，每台馬力1 HP。  chunk ids used: C.D棟 _汙水處理系統平面圖、剖面圖(1)#180", ev), "pass")
        self.assertEqual(grade({**count, "答案": "5 HP，共 2 台（交互運轉）"}, "5馬力, 2台   #172", ev), "pass")
        self.assertEqual(grade(count, "2台，每台 5 HP", ev), "fail")
        self.assertEqual(grade(count, "3台，每台 1 HP", ev), "fail")
        self.assertEqual(grade(count, "NOT FOUND", ev), "fail")
        self.assertEqual(grade({"問題": "數量", "答案": "15 台", "答案狀態": "有依據"}, "115 台", ev), "fail")
        self.assertEqual(grade({"問題": "數量", "答案": "1 台", "答案狀態": "有依據"}, "2台 chunk ids used: 剖面圖(1)#180", ev), "fail")

        text = {"問題": "管材", "答案": "PVC(B) 管", "答案狀態": "有依據"}
        self.assertEqual(grade(text, "PVC(B)管  chunk ids: C.D棟 _平面圖#180", ev), "pass")
        self.assertEqual(grade(text, "PVC(B)管  chunk ids: C.D棟 _平面圖#180", []), "review")

        absent = {"問題": "消防水池容量", "答案": "未標示。", "答案狀態": "圖面未標示"}
        self.assertEqual(grade({**absent, "答案": "未標示，無法由這兩張圖面確認。"}, "NOT FOUND  The document does not state the capacity", ev), "pass")
        self.assertEqual(grade(absent, "這兩張圖面未列出消防水池的設計容量。", ev), "pass")
        self.assertEqual(grade(absent, "消防水池為 7.5 HP", ev), "review")
        pump = {"問題": "是否能由污水泵 C 的 5 HP 規格判定消防泵浦馬力？", "答案": "不能。污水泵 C 與消防泵浦是不同設備，圖面未標示消防泵浦馬力。", "答案狀態": "圖面未標示"}
        self.assertEqual(grade(pump, "NO. 消防泵浦資訊未在提供的文檔中找到，無法由污水泵C的 5 HP 規格推斷消防泵浦規格。 chunk ids used: 消防 #344", ev), "pass")

        conflict = {"問題": "計算差異", "答案": "不一致：計算欄為 1.0 m³／min，設備圖例為 0.4 m³／min。", "答案狀態": "圖面存在差異"}
        self.assertEqual(grade(conflict, "No, they are not consistent: 0.9 m³/min vs 0.4 m³/min", ev), "fail")
        self.assertEqual(grade(conflict, "No, 1.0 m³/min in calc and 0.4 m³/min in legend", ev), "review")

    def test_final_claim_and_reference_forms(self):
        ev = [{"id": "p1"}]
        depth = {"問題": "有效水深？", "答案": "1.85 m", "答案狀態": "有依據", "驗證方式": "尺寸計算"}
        self.assertEqual(grade(depth, "3.2m  The calculation sheet in chunk #413 specifies the effective depth (D320) as 3.2m ... (e.g., 1.85m or 3.5m) refers to total tank height", ev), "fail")
        diffusers = {"問題": "粗氣泡散氣盤？", "答案": "76 只", "答案狀態": "有依據"}
        self.assertEqual(grade(diffusers, "接觸曝氣池所配置的總粗氣泡散氣盤數量是 0。 在提供的證據塊中 ... 根據 chunk #182 記載：本單元計採用 76只細氣泡散氣盤", ev), "fail")
        tank = {"問題": "尺寸？", "答案": "約 51.44 m³（7.9 × 3.52 × 1.85）", "答案狀態": "有依據", "驗證方式": "尺寸計算"}
        self.assertEqual(grade(tank, "約 51.4 m³", ev), "pass")
        self.assertEqual(grade({**tank, "驗證方式": ""}, "約 51.4 m³", ev), "fail")
        total = {"問題": "總量？", "答案": "約 462.62 m³（223.12＋239.5）", "答案狀態": "有依據"}
        self.assertEqual(grade(total, "約 462.62 m³", ev), "pass")
        dims = {"問題": "尺寸？", "答案": "長 467.5 cm、寬 470 cm、池高 350 cm、有效水深 325 cm", "答案狀態": "有依據"}
        self.assertEqual(grade(dims, "L467.5×W470×H350/D325", ev), "pass")
        self.assertEqual(grade(dims, "L467.5×W470×H350/D325", []), "review")
        verdict = {"問題": "是否一致？", "答案": "不一致：計算欄為 1.0 m³／min，設備圖例為 0.4 m³／min。", "答案狀態": "圖面存在差異"}
        self.assertEqual(grade(verdict, "不一致", ev), "review")
        self.assertEqual(grade(verdict, "一致：兩者皆為 1.0 m³/min", ev), "fail")
        winner = {"問題": "哪一種污水泵的馬力較大？", "答案": "污水泵 C（5 HP），大於污水泵 B（2 HP）", "答案狀態": "有依據"}
        self.assertEqual(grade(winner, "污水水泵 C 較大，為 5 HP", ev), "review")
        self.assertEqual(grade(winner, "污水泵 C 較大，為 10 HP", ev), "fail")


if __name__ == "__main__":
    unittest.main()
