"""Crop re-read precheck and fill on synthetic chunks, rows, GT and images: no model, no network, no benchmark text."""
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image

from docsearch import config, reread, rereadcheck

SECRET_Q = "SECRETQUESTIONTEXT 42"
C1 = {"id": "c1", "doc": "C.D系統", "kind": "block", "lines": [
    {"text": "口得", "bbox": [100, 100, 130, 120]},
    {"text": "25", "bbox": [140, 100, 170, 120]},
    {"text": "D20", "bbox": [600, 100, 640, 120]}]}
RAFT = {"id": "r1", "doc": "C.D筏基", "kind": "block", "lines": [{"text": "DN25 L=1.85m", "bbox": [10, 10, 200, 30]}]}
ROWS = [{"id": "F001", "attr": "bore", "src": [{"chunk": "c1", "bbox": [140, 100, 170, 120]}]},
        {"id": "F002", "attr": "legend", "src": [{"chunk": "c1", "bbox": [600, 100, 640, 120]}]}]


def rec(kind="B", attrs=(), status="agree", ocr=True, vlm=True, cer=0.0, n=1, crop="1.png"):
    return {"n": n, "kind": kind, "doc": "d", "chunk": "c", "rows": [], "attrs": list(attrs), "crop": crop,
            "ocr": "", "vlm": "", "status": status, "seconds": 1.0, "cached": False, "category": "legend",
            "cer_ocr": cer, "cer_vlm": cer, "digits_ocr_ok": ocr, "digits_vlm_ok": vlm}


class TargetsA(unittest.TestCase):
    def test_bore_row_merges_header_cell_and_skips_other_attrs(self):
        ts = rereadcheck.targets_a(ROWS, {"c1": C1})
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0]["rows"], ["F001"])
        self.assertEqual(ts[0]["text"], "口得 25")
        self.assertEqual(ts[0]["box"], [100, 100, 170, 120])

    def test_items_combine_a_and_b(self):
        gt = [("sys", "bore", "25", 0)]
        with mock.patch.object(rereadcheck.facts, "extract", return_value=ROWS), \
                mock.patch.object(rereadcheck, "GT", gt):
            items, unmatched = rereadcheck._items([C1, RAFT], {})
        self.assertEqual([i["kind"] for i in items], ["A", "B"])
        self.assertEqual(items[0]["attrs"], ["bore"])
        self.assertEqual(unmatched, 0)


class TargetsB(unittest.TestCase):
    def test_match_threshold_sheet_and_digit_filter(self):
        chunks = [C1, {**RAFT, "lines": [{"text": "12345678", "bbox": [10, 10, 200, 30]},
                                          {"text": "1234abcd", "bbox": [10, 40, 200, 60]}]}]
        gt = [("raft", "legend", "12345678", 0),       # exact -> matched
              ("raft", "legend", "no digits here", 0),  # no digit -> not a target at all
              ("sys", "legend", "DN25 L=1.85m", 0),    # sys sheet has no such line -> unmatched
              ("raft", "legend", "99999999", 0)]       # raft lines fuzz 50 -> below GT_MIN -> unmatched
        with mock.patch.object(rereadcheck, "GT", gt):
            items, unmatched = rereadcheck.targets_b(chunks)
        self.assertEqual([i["gt_index"] for i in items], [0])
        self.assertEqual(items[0]["target"]["text"], "12345678")
        self.assertEqual(unmatched, 2)

    def test_threshold_boundary(self):
        self.assertGreaterEqual(rereadcheck.fuzz.ratio("12345678", "12345678x"), rereadcheck.GT_MIN)
        self.assertLess(rereadcheck.fuzz.ratio("12345678", "1234abcd"), rereadcheck.GT_MIN)


class Metrics(unittest.TestCase):
    def test_cer_and_digit_exact(self):
        self.assertEqual(rereadcheck.cer("ABC1", "abc1"), 0.0)
        self.assertAlmostEqual(rereadcheck.cer("ABC1", "ABC2"), 0.25)
        self.assertTrue(rereadcheck.digit_exact("φ60cm", "Ø60cm"))
        self.assertFalse(rereadcheck.digit_exact("DN25", "DN52"))


class Summary(unittest.TestCase):
    def test_p1_passes_and_p3_passes(self):
        recs = [rec(ocr=True, vlm=True), rec(ocr=False, vlm=True, n=2, crop="2.png")]
        recs += [rec("A", ["tank_dims"], "agree", n=3 + i) for i in range(9)] + [rec("A", ["tank_dims"], "disagree", n=12)]
        s = rereadcheck.summarize(recs, unmatched=3)
        self.assertTrue(s["P1"])          # VLM 2 >= OCR 1 digit-exact, 0 OCR-only lines
        self.assertTrue(s["P3"])          # 9 of 10 tank targets agree
        self.assertEqual(s["B"]["unmatched"], 3)
        self.assertEqual(s["status"], {"agree": 11, "disagree": 1, "rejected": 0})

    def test_p1_fails_on_ocr_only_lines(self):
        recs = [rec(ocr=True, vlm=False, n=1), rec(ocr=True, vlm=False, n=2), rec(ocr=False, vlm=True, n=3)]
        s = rereadcheck.summarize(recs)
        self.assertFalse(s["P1"])         # two lines OCR gets right and the reader gets wrong
        self.assertEqual(s["B"]["ocr_right_vlm_wrong"], 2)

    def test_p3_fails_and_bore_crops_listed(self):
        recs = [rec("A", ["tank_dims"], "agree", n=1)] + [rec("A", ["tank_dims"], "disagree", n=2 + i) for i in range(9)]
        recs.append(rec("A", ["bore"], "agree", n=20, crop="20.png"))
        s = rereadcheck.summarize(recs)
        self.assertFalse(s["P3"])         # 1 of 10 tank targets agree
        self.assertEqual(s["P2_bore_crops"], ["20.png"])

    def test_empty_b_fails_rules(self):
        s = rereadcheck.summarize([rec("A", ["bore"], "agree")])
        self.assertFalse(s["P1"])
        self.assertFalse(s["P3"])


class Precheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.page = self.tmp / "page.png"
        Image.new("RGB", (300, 100), "white").save(self.page)
        self.docs = {"C.D系統": {"image": str(self.page)}}
        self.chunks = [{"id": "s1", "doc": "C.D系統", "kind": "block",
                        "lines": [{"text": "DN25", "bbox": [10, 10, 60, 30]}]}]
        self.patches = [
            mock.patch.object(config, "REREAD", "openai"),
            mock.patch.object(config, "WORK", self.tmp / "work"),
            mock.patch.object(rereadcheck, "_load_index", return_value=(self.chunks, self.docs)),
            mock.patch.object(rereadcheck.facts, "extract", return_value=[]),
            mock.patch.object(rereadcheck, "GT", [("sys", "legend", "DN25", 0)]),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self._tmp.cleanup()

    def test_writes_crops_report_and_summary(self):
        with mock.patch.object(reread, "_openai_read", return_value="DN25"):
            s = rereadcheck.precheck(self.tmp / "out")
        self.assertEqual(s["crops"], 1)
        self.assertEqual(s["status"]["agree"], 1)
        self.assertTrue(s["P1"])
        self.assertTrue((self.tmp / "out" / "1.png").exists())
        data = json.loads((self.tmp / "out" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["backend"], "openai")
        self.assertIn("reader:", (self.tmp / "out" / "report.md").read_text(encoding="utf-8"))

    def test_missing_page_raises_with_path(self):
        self.docs["C.D系統"]["image"] = str(self.tmp / "gone.png")
        with self.assertRaises(FileNotFoundError) as cm:
            rereadcheck.precheck(self.tmp / "out")
        self.assertIn("gone.png", str(cm.exception))

    def test_unreachable_server_raises_with_url(self):
        url = "http://127.0.0.1:9/v1"
        with mock.patch.object(config, "REREAD_URL", url), \
                mock.patch.object(reread.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(RuntimeError) as cm:
                rereadcheck.precheck(self.tmp / "out")
        self.assertIn(url, str(cm.exception))

    def test_refuses_when_reader_off(self):
        with mock.patch.object(config, "REREAD", "0"), self.assertRaises(SystemExit):
            rereadcheck.precheck(self.tmp / "out")


class Fill(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bench = Path(self._tmp.name) / "bench.json"
        self.bench.write_text(json.dumps({"題目": [{"編號": "Q1", "問題": SECRET_Q}, {"編號": "Q2", "問題": "other"}]},
                                         ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _engine(self):
        class Reader:
            def run(self, ev, fx):
                return {"records": [{"status": "disagree", "seconds": 0.5}, {"status": "agree", "seconds": 0.25}]}

        class Engine:
            def prepare(self, question):
                assert question == SECRET_Q or question == "other"
                return {"ev": [], "fx": None}

            def rereader(self):
                return Reader()

        return Engine()

    def test_fill_prints_ids_counts_seconds_not_question(self):
        out = io.StringIO()
        with mock.patch.object(config, "REREAD", "openai"), \
                mock.patch("docsearch.cli._engine", return_value=self._engine()), \
                contextlib.redirect_stdout(out):
            rereadcheck.fill(self.bench, ["Q1", "Q2"])
        text = out.getvalue()
        self.assertNotIn("SECRET", text)
        self.assertIn("Q1\tcrops=2\tdisagree=1\tseconds=0.75", text)
        self.assertIn("Q2\tcrops=2", text)

    def test_unknown_id_raises_naming_only_ids(self):
        with mock.patch.object(config, "REREAD", "openai"), \
                mock.patch("docsearch.cli._engine", return_value=self._engine()):
            with self.assertRaises(SystemExit) as cm:
                rereadcheck.fill(self.bench, ["Q9"])
        self.assertIn("Q9", str(cm.exception))
        self.assertNotIn("SECRET", str(cm.exception))

    def test_cli_ids_file_splits_commas_and_lines(self):
        ids_file = Path(self._tmp.name) / "ids.csv"
        ids_file.write_text("Q1,Q2\nQ3 Q4\n", encoding="utf-8")
        with mock.patch.object(rereadcheck, "fill") as fill:
            rereadcheck.main(["fill", str(self.bench), "--ids-file", str(ids_file)])
        fill.assert_called_once_with(self.bench, ["Q1", "Q2", "Q3", "Q4"])


if __name__ == "__main__":
    unittest.main()
