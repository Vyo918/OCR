import contextlib
import io
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


def embedded_detail(page: str) -> dict:
    start = page.index('id="qdetail">') + len('id="qdetail">')
    return json.loads(page[start:page.index("</script>", start)])


def private_inputs(tmp: Path, question: str = "FIXTURE QUESTION TEXT") -> tuple[Path, Path, Path]:
    curated = tmp / "curated.json"
    curated.write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
    benchmark = tmp / "benchmark.json"
    benchmark.write_text(json.dumps({"基準資訊": {}, "題目": [
        {"編號": "Q001", "類別": "cat-a", "難度": "easy", "問題": question, "答案": "FIXTURE REFERENCE TEXT"},
        {"編號": "Q002", "類別": "cat-a", "難度": "easy", "問題": "FIXTURE SECOND", "答案": "FIXTURE SECOND REF"},
    ]}, ensure_ascii=False), encoding="utf-8")
    bench = tmp / "bench"
    (bench / "run-a").mkdir(parents=True)
    rows = [
        {"id": "Q001", "grade": "pass", "seconds": 10.0, "reference": "FIXTURE REFERENCE TEXT",
         "answer": "FIXTURE ENGINE ANSWER\ndrawing#153, drawing#391"},
        {"id": "Q002", "grade": "fail", "seconds": 20.0, "reference": "FIXTURE SECOND REF",
         "answer": "FIXTURE SECOND ANSWER"},
    ]
    (bench / "run-a" / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return curated, benchmark, bench


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

    def test_default_build_embeds_no_question_or_answer_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            curated, _, bench = private_inputs(Path(tmp))
            out = Path(tmp) / "public.html"
            history.build(curated, bench, out)
            page = out.read_text(encoding="utf-8")
        self.assertFalse(embedded_detail(page)["text"])
        for secret in ("FIXTURE QUESTION TEXT", "FIXTURE REFERENCE TEXT", "FIXTURE ENGINE ANSWER", "drawing#153"):
            self.assertNotIn(secret, page)

    def test_benchmark_build_embeds_question_reference_and_engine_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            curated, benchmark, bench = private_inputs(Path(tmp))
            out = Path(tmp) / "private.html"
            history.build(curated, bench, out, benchmark)
            page = out.read_text(encoding="utf-8")
        detail = embedded_detail(page)
        self.assertTrue(detail["text"])
        q = detail["questions"]["Q001"]
        self.assertEqual((q["question"], q["reference"]), ("FIXTURE QUESTION TEXT", "FIXTURE REFERENCE TEXT"))
        self.assertEqual(q["results"][0]["answer"], "FIXTURE ENGINE ANSWER")
        self.assertNotIn("drawing#153", page)
        self.assertIn("Reference:", page)
        self.assertIn("Engine answer", page)

    def test_benchmark_build_warns_and_refuses_docs_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            curated, benchmark, bench = private_inputs(Path(tmp))
            args = ["--curated", str(curated), "--bench", str(bench), "--benchmark", str(benchmark)]
            refused = ROOT / "docs" / "history-refusal-check.html"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(history.main(args + ["--out", str(refused)]), 2)
            self.assertFalse(refused.exists())
            out = Path(tmp) / "ok.html"
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                self.assertEqual(history.main(args + ["--out", str(out)]), 0)
            self.assertIn("do not commit or publish", err.getvalue())
            self.assertIn("FIXTURE QUESTION TEXT", out.read_text(encoding="utf-8"))

    def test_embedded_json_survives_script_close_and_quotes(self):
        hostile = 'FIXTURE </script><img src=x onerror="alert(1)"> & "quoted" <!-- end'
        with tempfile.TemporaryDirectory() as tmp:
            curated, benchmark, bench = private_inputs(Path(tmp), question=hostile)
            out = Path(tmp) / "private.html"
            history.build(curated, bench, out, benchmark)
            page = out.read_text(encoding="utf-8")
        self.assertEqual(page.count("</script>"), page.count("<script"))
        self.assertEqual(embedded_detail(page)["questions"]["Q001"]["question"], hostile)

    def test_grid_marks_not_run_cells_non_interactive(self):
        data = fixture()
        data["runs"].append({**data["runs"][0], "run": "run-b", "label": "Run B",
                             "per_question": [{"id": "Q001", "category": "cat-a", "difficulty": "easy",
                                               "grade": "pass", "seconds": 11.0}]})
        page = history.render(data)
        self.assertIn('<td class="g g-none"></td>', page)
        self.assertEqual(page.count('data-q="Q002" data-r='), 1)
        self.assertIsNone(embedded_detail(page)["questions"]["Q002"]["results"][1])


if __name__ == "__main__":
    unittest.main()
