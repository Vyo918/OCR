import json
import re
import tempfile
import unittest
from pathlib import Path

from docsearch import history

ROOT = Path(__file__).resolve().parents[1]
REAL_JSON = ROOT / "docs" / "history" / "history.json"
SVG_NS = "http://www.w3.org/2000/svg"


def fixture() -> dict:
    row = lambda rid, grade, secs: {"id": rid, "category": "cat-a", "difficulty": "easy",
                                    "grade": grade, "seconds": secs}
    return {
        "schema_version": 1,
        "as_of": "2000-01-01",
        "project": "fixture project",
        "default": {"llm": "model-x", "ocr": "engine-y", "top_k": 12, "seed": 0, "facts": False,
                    "hand_score": "1 of 2 correct", "caveat": "fixture caveat"},
        "best": {"label": "fixture best", "hand": "1 / 0 / 1", "caveat": "fixture caveat"},
        "retained40": {"ids": ["Q001", "Q002"], "note": "fixture retained set"},
        "experiments": [{
            "id": "exp-one", "title": "Experiment one", "date": "2000-01-02", "status": "done",
            "purpose": "p", "method": "m", "dataset": "d", "models": ["model-x"],
            "configuration": ["c"], "environment": "e", "duration": "1 s",
            "results": {"columns": ["a", "b"], "rows": [["1", "2"]]},
            "judgement": "j", "caveats": ["k"], "artifacts": ["run-a"],
        }],
        "runs": [{
            "run": "run-a", "label": "Run A", "experiment": "exp-one", "model": "model-x",
            "machine": "machine-z", "dataset": "dataset-d", "status": "done", "config": "cfg",
            "hand": [{"scope": "all", "correct": 1, "partial": 0, "wrong": 1, "source": "fixture"}],
            "notes": "n", "grid": True, "expected": 3, "appended_questions": ["Q002"],
            "per_question": [row("Q001", "pass", 10.0), row("Q002", "fail", 20.0)],
        }],
        "models": [{"id": "model-x", "name": "Model X", "role": "answer", "size": "1B",
                    "where": "here", "status": "tested", "verdict": "ok"}],
        "machines": [{"id": "machine-z", "name": "Box", "hardware": "cpu", "notes": ""}],
        "datasets": [{"id": "dataset-d", "name": "Fixture set", "description": "neutral"}],
        "milestones": [{"date": "2000-01-01", "commit": "abc1234", "title": "start"}],
        "not_tested": [{"name": "Other", "role": "r", "protocol": "p", "priority": "low"}],
        "findings": ["f"],
        "caveats": ["c"],
        "open_items": ["o"],
    }


class HistoryTests(unittest.TestCase):
    def test_fixture_and_real_json_validate(self):
        self.assertEqual(history.validate(fixture()), [])
        self.assertEqual(history.validate(json.loads(REAL_JSON.read_text(encoding="utf-8"))), [])

    def test_validation_rejects_missing_fields_and_forbidden_keys(self):
        data = fixture()
        del data["experiments"][0]["purpose"]
        self.assertTrue(any("purpose" in e for e in history.validate(data)))
        data = fixture()
        data["runs"][0]["per_question"][0]["answer"] = "text"
        self.assertTrue(any("forbidden" in e for e in history.validate(data)))

    def test_every_section_is_collapsed_details_except_overview(self):
        page = history.render(fixture())
        self.assertIn('<section id="overview"', page)
        for anchor, _ in history.NAV[1:]:
            self.assertRegex(page, rf'<details class="section" id="{anchor}">')
        self.assertNotIn(" open>", page)
        self.assertEqual(page.count("<details"), page.count("</details>"))

    def test_no_external_requests_and_no_placeholder_left(self):
        page = history.render(fixture())
        urls = set(re.findall(r"https?://[^\s\"'<>)]+", page))
        self.assertTrue(urls <= {SVG_NS}, urls)
        self.assertNotRegex(page, r"<(script|link|img)[^>]*\b(src|href)=\"https?:")
        self.assertNotIn(history.PLACEHOLDER, page)

    def test_real_json_renders_with_only_whitelisted_row_fields(self):
        data = json.loads(REAL_JSON.read_text(encoding="utf-8"))
        for run in data["runs"]:
            for row in run["per_question"]:
                self.assertEqual(set(row), set(history.ROW_FIELDS))
        page = history.render(data)
        self.assertIn("Testing and benchmarking history", page)

    def test_read_bench_ignores_reference_and_answer_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp)
            (bench / "run-a").mkdir()
            (bench / "run-a" / "results.jsonl").write_text(json.dumps({
                "id": "Q001", "類別": "cat-a", "難度": "easy", "grade": "pass", "seconds": 1.26,
                "reference": "NEUTRAL-REF", "answer": "NEUTRAL-ANS"}, ensure_ascii=False) + "\n",
                encoding="utf-8")
            (bench / "run-a" / "summary.md").write_text("Config: fixture-config\n", encoding="utf-8")
            found = history.read_bench(bench)
        self.assertEqual(set(found), {"run-a"})
        self.assertEqual(found["run-a"]["config"], "fixture-config")
        row = found["run-a"]["per_question"][0]
        self.assertEqual(set(row), set(history.ROW_FIELDS))
        self.assertEqual(row["seconds"], 1.3)
        self.assertNotIn("NEUTRAL", json.dumps(found, ensure_ascii=False))

    def test_metrics_exclude_appended_questions(self):
        m = history.metrics(fixture()["runs"][0])
        self.assertEqual((m["n"], m["pass"], m["fail"], m["expected"], m["appended"]), (1, 1, 0, 3, 1))
        self.assertEqual(m["pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
