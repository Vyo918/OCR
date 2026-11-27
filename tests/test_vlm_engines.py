"""OpenAI-compatible VLM engine: the Qwen and dots.mocr parsers, OpenAIVLM against a fake local server, and the sample
helpers of modelsel. No model, no network beyond 127.0.0.1."""
import base64
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from PIL import Image

from docsearch.engines import OpenAIVLM, parse_dots_quads, parse_qwen_spotting
from docsearch.modelsel.__main__ import _crop, _inside, locate, pick_regions, sample_metrics, scored

PROMPT = "Detect and recognize the text in the image."


class ParseQwen(unittest.TestCase):
    def test_keys_scale_and_truncation(self):
        raw = ('```json\n[{"bbox_2d": [0, 0, 500, 250], "text_content": "管径 DN50"}, '
               '{"bbox": [100, 100, 200, 200], "text": "P5"}, {"bbox_2d": [1, 2')
        self.assertEqual(parse_qwen_spotting(raw, (1000, 800)), [
            ("管径 DN50", [0.0, 0.0, 500.0, 200.0]),
            ("P5", [100.0, 80.0, 200.0, 160.0]),
        ])

    def test_empty_text_and_bad_numbers_dropped(self):
        raw = '[{"bbox_2d": [0, 0, 10, 10], "text_content": "  "}, {"bbox_2d": ["x", 0, 1, 1], "text_content": "a"}]'
        self.assertEqual(parse_qwen_spotting(raw, (1000, 1000)), [])


class ParseDots(unittest.TestCase):
    def test_quads_reduced_to_boxes_and_clipped(self):
        raw = ("[(10, 20), (110, 20), (110, 40), (10, 40)] 管径\n"
               "(0, 0), (50, 0), (50, 10), (0, 10) DN50\n"
               "(0, 0), (300, 0), (300, 10), (0, 10) wide\n"
               "noise without quad")
        self.assertEqual(parse_dots_quads(raw, (200, 100)), [
            ("管径", [10.0, 20.0, 110.0, 40.0]),
            ("DN50", [0.0, 0.0, 50.0, 10.0]),
            ("wide", [0.0, 0.0, 200.0, 10.0]),
        ])

    def test_box_ref_token_form(self):
        raw = ("<|box_start|>(53, 16), (163, 16), (163, 47), (53, 47)<|box_end|><|ref_start|>-1431<|ref_end|>"
               "<|box_start|>(35, 152), (300, 152), (300, 183), (35, 183)<|box_end|><|ref_start|>60X60cm<|ref_end|>"
               "<|box_start|>(1, 2), (3, 2), (3, 4)")   # truncated entry is dropped
        self.assertEqual(parse_dots_quads(raw, (200, 200)), [
            ("-1431", [53.0, 16.0, 163.0, 47.0]),
            ("60X60cm", [35.0, 152.0, 200.0, 183.0]),
        ])


def fake_server(reply: str):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def sent_png(body) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(body["messages"][0]["content"][0]["image_url"]["url"].split(",", 1)[1])))


class OpenAIVLMSpot(unittest.TestCase):
    def engine(self, reply, model_env=None, parser=parse_dots_quads, **kw):
        srv, seen = fake_server(reply)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        patcher = mock.patch.dict(os.environ, {"DOCSEARCH_VLM_URL": f"http://127.0.0.1:{srv.server_port}/v1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("DOCSEARCH_VLM_MODEL", None)
        if model_env:
            os.environ["DOCSEARCH_VLM_MODEL"] = model_env
        return OpenAIVLM("org/model", PROMPT, parser, **kw), seen

    def test_request_shape_and_think_stripped(self):
        eng, seen = self.engine("<think>plan the layout</think>(0, 0), (10, 0), (10, 5), (0, 5) 管径", upscale=1.0)
        out = eng.spot(Image.new("RGB", (20, 10), "white"))
        self.assertEqual(out, [("管径", [0, 0, 10, 5], None)])
        path, body = seen[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "org/model")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["messages"][0]["content"][1], {"type": "text", "text": PROMPT})
        self.assertTrue(body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(sent_png(body).size, (20, 10))

    def test_env_model_overrides_entry_model(self):
        eng, seen = self.engine("", model_env="other/model", upscale=1.0)
        eng.spot(Image.new("RGB", (8, 8), "white"))
        self.assertEqual(seen[0][1]["model"], "other/model")

    def test_multiple_maps_boxes_back_by_exact_scale(self):
        # 100x50 tile, upscale 2 -> 200x100, snapped to multiples of 28 -> 196x112 (the sent image)
        eng, seen = self.engine("(98, 56), (196, 56), (196, 112), (98, 112) 管", upscale=2.0, multiple=28)
        out = eng.spot(Image.new("RGB", (100, 50), "white"))
        self.assertEqual(sent_png(seen[0][1]).size, (196, 112))
        (text, box, _), = out
        self.assertEqual(text, "管")
        for got, want in zip(box, [50.0, 25.0, 100.0, 50.0]):
            self.assertAlmostEqual(got, want, places=6)

    def test_qwen_boxes_scale_with_upscale(self):
        eng, _ = self.engine('[{"bbox_2d": [0, 0, 500, 500], "text_content": "A"}]', parser=parse_qwen_spotting, upscale=2.0)
        out = eng.spot(Image.new("RGB", (30, 20), "white"))
        self.assertEqual(out, [("A", [0.0, 0.0, 15.0, 10.0], None)])


class SampleHelpers(unittest.TestCase):
    def test_inside_is_half_open_and_crop_clamps(self):
        self.assertTrue(_inside((0, 0), (0, 0, 10, 10)))
        self.assertFalse(_inside((10, 5), (0, 0, 10, 10)))
        self.assertEqual(_crop((10, 10, 100, 100), (120, 200)), (0, 0, 120, 164))

    def test_locate_takes_best_line_centre_and_skips_absent(self):
        gts = [("sys", "x", "DN50 管", False), ("sys", "x", "absent text zz", False)]
        lines = {"sys": [{"text": "DN50 管", "bbox": [0, 0, 10, 4]}, {"text": "other", "bbox": [50, 50, 60, 60]}]}
        self.assertEqual(locate(gts, lines), {("sys", "DN50 管"): (5, 2)})

    def test_pick_regions_ranks_by_located_gt_per_sheet(self):
        grids = {"sys": [(0, 0, 500, 640), (500, 0, 1140, 640), (0, 500, 640, 1140)],
                 "raft": [(0, 0, 640, 640), (600, 0, 1240, 640)]}
        located = {("sys", "a"): (550, 100), ("sys", "b"): (560, 200), ("sys", "c"): (50, 600), ("raft", "d"): (100, 100)}
        picked = pick_regions(2, grids, located)
        self.assertEqual(picked["sys"], [(500, 0, 1140, 640)])
        self.assertEqual(picked["raft"], [(0, 0, 640, 640)])

    def test_scored_is_the_located_gt_inside_a_sampled_region(self):
        gts = [("sys", "x", "in", False), ("sys", "x", "out", False), ("sys", "x", "unlocated", False)]
        located = {("sys", "in"): (100, 100), ("sys", "out"): (900, 900)}
        self.assertEqual(scored(gts, located, {"sys": [(0, 0, 640, 640)]}), [gts[0]])

    def test_metrics_hit_digit_fold_and_seconds(self):
        gts = [("sys", "x", "DN50", False), ("sys", "x", "管径 3.5m", False),
               ("sys", "x", "風机", False), ("sys", "x", "absent", False)]
        lines = {"sys": [{"text": "管径 3.5m", "bbox": [0, 0, 1, 1]}, {"text": "DN50", "bbox": [0, 0, 1, 1]},
                         {"text": "风机", "bbox": [0, 0, 1, 1]}, {"text": "國", "bbox": [0, 0, 1, 1]}]}
        m = sample_metrics(gts, lines, [2.0, 4.0])
        self.assertEqual(m["GT"], 4)
        self.assertEqual(m["hit"], 50.0)      # DN50 and 管径3.5m; 風机 is not folded, so it misses 风机
        self.assertEqual(m["digit"], 100.0)   # both GT strings with numbers have them in a hit line
        self.assertEqual(m["fold"], 20.0)     # 1 of 5 CJK output characters (國) is Traditional
        self.assertEqual(m["lines"], 4)
        self.assertEqual(m["sec"], 3.0)


if __name__ == "__main__":
    unittest.main()
