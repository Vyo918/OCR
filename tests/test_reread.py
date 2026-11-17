"""Crop re-read core on synthetic chunks, rows and images: no model, no network."""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image

from docsearch import config, reread

ROOT = Path(__file__).resolve().parent.parent

LINE_25 = {"text": "25", "bbox": [140, 100, 170, 120]}
C1 = {"id": "sheet#1", "doc": "sheet", "kind": "block", "lines": [
    {"text": "HEAD", "bbox": [100, 100, 130, 120]}, LINE_25, {"text": "D20", "bbox": [600, 100, 640, 120]}]}
C2 = {"id": "sheet#2", "doc": "sheet", "kind": "block", "lines": [{"text": "W=1.85m", "bbox": [100, 300, 170, 320]}]}
BY_ID = {c["id"]: c for c in (C1, C2)}
ROW = {"id": "F001", "src": [{"chunk": "sheet#1", "bbox": [140, 100, 170, 120]}]}


class Config(unittest.TestCase):
    def _flags(self, value):
        env = {k: v for k, v in os.environ.items() if k != "DOCSEARCH_REREAD"}
        if value is not None:
            env["DOCSEARCH_REREAD"] = value
        code = "from docsearch import config as c; print(c.REREAD, c.REREAD_URL, c.REREAD_MODEL, c.REREAD_MAX)"
        return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)

    def test_defaults(self):
        p = self._flags(None)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.split(), ["0", "http://127.0.0.1:8082/v1", "mlx-community/Qwen3-VL-4B-Instruct-4bit", "12"])

    def test_unknown_value_raises(self):
        p = self._flags("bogus")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ValueError", p.stderr)


class Select(unittest.TestCase):
    def test_fact_row_seed_first_then_block_lines(self):
        targets = reread.select([C2, C1], {"rows": [ROW]}, BY_ID, 10)
        self.assertEqual([t["text"] for t in targets], ["HEAD 25", "W=1.85m", "D20"])
        self.assertEqual(targets[0]["rows"], ["F001"])
        self.assertEqual(targets[0]["box"], [100, 100, 170, 120])
        self.assertEqual(targets[1]["rows"], [])

    def test_cap(self):
        self.assertEqual([t["text"] for t in reread.select([C2, C1], None, BY_ID, 1)], ["W=1.85m"])

    def test_dedupe_same_doc_and_bbox(self):
        twin = {"id": "sheet#4", "doc": "sheet", "kind": "block", "lines": [LINE_25]}
        self.assertEqual([t["text"] for t in reread.select([C1, twin], None, {}, 10)], ["HEAD 25", "D20"])

    def test_same_row_merge_joins_adjacent_cell(self):
        # "HEAD" ends 10 px before "25" on the same row (gap <= 1.5 line heights): one target
        self.assertEqual(reread.select([C1], None, {}, 10)[0]["text"], "HEAD 25")

    def test_same_row_merge_skips_other_row(self):
        far = {"id": "sheet#5", "doc": "sheet", "kind": "block", "lines": [
            {"text": "A", "bbox": [100, 300, 130, 320]}, LINE_25]}
        self.assertEqual([t["text"] for t in reread.select([far], None, {}, 10)], ["25"])

    def test_only_short_block_digit_lines(self):
        long = {"id": "sheet#3", "doc": "sheet", "kind": "block", "lines": [{"text": "1" * 41, "bbox": [0, 0, 50, 20]}]}
        section = {"id": "sheet#6", "doc": "sheet", "kind": "section", "lines": [{"text": "25", "bbox": [0, 0, 20, 20]}]}
        self.assertEqual(reread.select([long, section], None, {}, 10), [])


class Crop(unittest.TestCase):
    def setUp(self):
        self.page = Image.new("RGB", (1000, 800), "white")

    def test_pad_and_upscale(self):
        # h 20 -> pad 12 each side (64x44 region), scale 2.4
        self.assertEqual(reread.crop(self.page, [100, 100, 140, 120], 20).size, (154, 106))

    def test_clip_to_page_and_scale_cap_4x(self):
        # pad 8 clipped at the page corner (18x18 region), scale min(4, 4.8) = 4
        self.assertEqual(reread.crop(self.page, [0, 0, 10, 10], 10).size, (72, 72))

    def test_no_upscale_for_large_text(self):
        self.assertEqual(reread.crop(self.page, [100, 100, 200, 160], 60).size, (172, 132))

    def test_long_side_capped_at_1280(self):
        img = reread.crop(Image.new("RGB", (4000, 3000), "white"), [0, 0, 1000, 100], 20)
        self.assertEqual(max(img.size), 1280)
        self.assertEqual(img.mode, "RGB")


class Reconcile(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(reread.normalise("“2×3”"), "2×3")
        self.assertEqual(reread.normalise("2 x 3 * 4"), "2×3×4")
        self.assertEqual(reread.normalise('6″ PVC'), '6" PVC')
        self.assertEqual(reread.normalise("```\n12\n```"), "12")
        self.assertEqual(reread.normalise("  D20   mm "), "D20 mm")

    def test_digits(self):
        self.assertEqual(reread.digits("A790×B352.5×C350"), ["790", "352.5", "350"])

    def test_agree(self):
        self.assertEqual(reread.reconcile("D=2.5×H350", "D=2.5 x H350"), "agree")

    def test_disagree(self):
        self.assertEqual(reread.reconcile("D20", "D25"), "disagree")
        self.assertEqual(reread.reconcile("D250", "D258"), "disagree")

    def test_rejected(self):
        self.assertEqual(reread.reconcile("D20", ""), "rejected")                                # empty
        self.assertEqual(reread.reconcile("D20", "D"), "rejected")                               # no digit
        self.assertEqual(reread.reconcile("1111111111111111", "1111111111111111"), "rejected")   # looped
        self.assertEqual(reread.reconcile("D20", "D20" + "x" * 20), "rejected")                  # over-long
        self.assertEqual(reread.reconcile("D20", "D1 2 3 4"), "rejected")                        # number count
        self.assertEqual(reread.reconcile("AAAA 20", "BBBB 20"), "rejected")                     # text skeleton differs


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        Image.new("RGB", (1000, 800), "white").save(root / "sheet.png")
        self.docs = {"sheet": {"image": str(root / "sheet.png"), "size": [1000, 800]}}
        self.work = root / "work"
        for name, value in (("WORK", self.work), ("REREAD", "openai"), ("REREAD_MODEL", "m1")):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.target = reread.select([C1], None, {}, 1)[0]

    def test_hit_skips_backend_and_key_has_backend_and_model(self):
        with mock.patch.object(reread.Rereader, "_read", return_value="D25") as read:
            first = reread.Rereader(self.docs, [C1]).read_target(self.target)
            second = reread.Rereader(self.docs, [C1]).read_target(self.target)
        self.assertEqual(read.call_count, 1)
        self.assertEqual((first["cached"], second["cached"], second["text"]), (False, True, "D25"))
        cache = json.loads((self.work / "reread" / "cache.json").read_text(encoding="utf-8"))
        self.assertEqual(list(cache), ["openai|m1|sheet|100,100,170,120"])
        self.assertFalse((self.work / "reread" / "cache.json.tmp").exists())

    def test_model_change_misses(self):
        with mock.patch.object(reread.Rereader, "_read", return_value="D25") as read:
            reread.Rereader(self.docs, [C1]).read_target(self.target)
            with mock.patch.object(config, "REREAD_MODEL", "m2"):
                reread.Rereader(self.docs, [C1]).read_target(self.target)
        self.assertEqual(read.call_count, 2)

    def test_hunyuan_model_is_config_hunyuan_id(self):
        with mock.patch.object(config, "REREAD", "hunyuan"):
            self.assertEqual(reread.Rereader(self.docs, [C1]).model, config.HUNYUAN_ID)

    def test_run_block_only_for_disagree(self):
        # targets: "HEAD 25" (reader agrees) and "D20" (reader says D25)
        with mock.patch.object(reread.Rereader, "_read", side_effect=["25", "D25"]):
            out = reread.Rereader(self.docs, [C1]).run([C1], None)
        self.assertEqual([r["status"] for r in out["records"]], ["agree", "disagree"])
        self.assertEqual([r["id"] for r in out["records"]], [None, "R1"])
        self.assertEqual(out["evidence"], ["D25"])
        self.assertEqual(out["text"], reread.HEADER + "\n[R1] sheet#1 | OCR: D20 | re-read: D25")

    def test_run_none_when_nothing_disagrees(self):
        with mock.patch.object(reread.Rereader, "_read", side_effect=["25", "D20"]):
            out = reread.Rereader(self.docs, [C1]).run([C1], None)
        self.assertIsNone(out["text"])
        self.assertEqual(out["evidence"], [])

    def test_format_block_rows_and_sequential_ids(self):
        recs = [{"id": "R1", "chunk": "sheet#1", "rows": ["F013"], "ocr": "HEAD 25", "vlm": "HEAD 35"},
                {"id": None, "chunk": "sheet#2", "rows": [], "ocr": "x", "vlm": "x"},
                {"id": "R2", "chunk": "sheet#2", "rows": [], "ocr": "W=1.85m", "vlm": "W=1.35m"}]
        self.assertEqual(reread.format_block(recs), reread.HEADER + "\n"
                         "[R1] sheet#1 | rows F013 | OCR: HEAD 25 | re-read: HEAD 35\n"
                         "[R2] sheet#2 | OCR: W=1.85m | re-read: W=1.35m")
        self.assertIsNone(reread.format_block([recs[1]]))


class Openai(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(reread.urllib.request, "urlopen")
        self.urlopen = patch.start()
        self.addCleanup(patch.stop)

    def _reply(self, content):
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.urlopen.return_value.__enter__.return_value.read.return_value = body

    def test_request_body(self):
        self._reply("<think>hm</think>D25")
        with mock.patch.object(config, "REREAD_MODEL", "m1"), mock.patch.object(config, "REREAD_URL", "http://h:1/v1"):
            out = reread._openai_read(Image.new("RGB", (10, 10), "white"))
        self.assertEqual(out, "D25")
        req = self.urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "http://h:1/v1/chat/completions")
        self.assertEqual(self.urlopen.call_args.kwargs["timeout"], 120)
        body = json.loads(req.data)
        self.assertEqual(body["model"], "m1")
        self.assertEqual((body["temperature"], body["max_tokens"]), (0, 64))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        content = body["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1], {"type": "text", "text": reread.PROMPT})

    def test_http_error_names_url(self):
        url = "http://h:1/v1/chat/completions"
        self.urlopen.side_effect = urllib.error.HTTPError(url, 500, "boom", {}, io.BytesIO(b"bad"))
        with mock.patch.object(config, "REREAD_URL", "http://h:1/v1"):
            with self.assertRaises(RuntimeError) as cm:
                reread._openai_read(Image.new("RGB", (10, 10)))
        self.assertIn(url, str(cm.exception))
        self.assertIn("500", str(cm.exception))

    def test_unreachable_names_url(self):
        self.urlopen.side_effect = urllib.error.URLError("refused")
        with self.assertRaises(RuntimeError) as cm:
            reread._openai_read(Image.new("RGB", (10, 10)))
        self.assertIn(config.REREAD_URL, str(cm.exception))


class Resolver(unittest.TestCase):
    def test_given_path_then_pages_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sheet.png").write_bytes(b"x")
            pages = root / "pages"
            pages.mkdir()
            (pages / "sheet.native-000.jpg").write_bytes(b"x")
            with mock.patch.object(config, "PAGES", pages):
                self.assertEqual(reread.resolve_image(str(root / "sheet.png")), root / "sheet.png")
                self.assertEqual(reread.resolve_image("/data/work/pages/sheet.native-000.jpg"), pages / "sheet.native-000.jpg")
                with self.assertRaises(FileNotFoundError) as cm:
                    reread.resolve_image("/data/work/pages/missing.jpg")
        self.assertIn("/data/work/pages/missing.jpg", str(cm.exception))
        self.assertIn(str(pages / "missing.jpg"), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
