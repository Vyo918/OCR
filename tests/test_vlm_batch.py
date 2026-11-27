"""Batched VLM re-reads: cascade grouping, offsets, progress and fallback with a fake slow engine; Hunyuan.spot_batch
decoding with fake processor and model. No model, no network."""
import contextlib
import io
import unittest
from unittest import mock

from PIL import Image

from docsearch import cascade as cascade_mod
from docsearch import config
from docsearch.cascade import PASS1_SHARE, Cascade
from docsearch.engines import Hunyuan

# seven 100 px tiles side by side; each tile is painted with its own red level so a fake engine can tell the crops apart
TILES = [(i * 100, 0, i * 100 + 100, 100) for i in range(7)]
PAGE = Image.new("RGB", (700, 100), (255, 255, 255))
for i, t in enumerate(TILES):
    PAGE.paste((30 * i, 0, 0), t)


def tile_of(img: Image.Image) -> int:
    return img.getpixel((0, 0))[0] // 30


def base_lines():
    """Two unsure PP-OCR lines per tile, so every tile goes to the VLM."""
    return [{"doc": "p", "text": "junk", "bbox": [i * 100 + 40, 40, i * 100 + 60, 60], "score": 0.5}
            for i in range(7) for _ in range(2)]


class SpotOnly:
    def __init__(self):
        self.spot_calls = []

    def spot(self, img):
        self.spot_calls.append(tile_of(img))
        return [(f"v{tile_of(img)}", [0, 0, 10, 10], None)]


class FakeVLM(SpotOnly):
    def __init__(self, fail_batches=0):
        super().__init__()
        self.batch_calls, self.fail_batches = [], fail_batches

    def spot_batch(self, imgs):
        self.batch_calls.append([tile_of(i) for i in imgs])
        if self.fail_batches:
            self.fail_batches -= 1
            raise RuntimeError("out of memory")
        return [[(f"v{tile_of(i)}", [0, 0, 10, 10], None)] for i in imgs]


def run_page(slow, batch, on_tile=None, on_fraction=None):
    c = Cascade.__new__(Cascade)   # skip __init__: it would load PP-OCR and the VLM
    c.fast, c._slow_name, c.slow = object(), "fake", slow
    c.min_score, c.min_lines, c.tile, c.overlap, c.stats = 0.75, 2, 640, 120, {}
    with mock.patch.object(config, "VLM_BATCH", batch), \
            mock.patch.object(cascade_mod, "read_page", side_effect=lambda *a, **k: base_lines()), \
            mock.patch.object(cascade_mod, "tiles", return_value=TILES), \
            mock.patch("docsearch.timing.stage", lambda *a: contextlib.nullcontext()), \
            mock.patch.object(cascade_mod, "empty_cache", lambda: None), \
            contextlib.redirect_stdout(io.StringIO()):
        out = c.read_page(PAGE, "p", on_tile=on_tile, on_fraction=on_fraction)
    return sorted((l for l in out if l.get("src") == "vlm"), key=lambda l: l["bbox"][0])


def expected_vlm():
    return [{"doc": "p", "text": f"v{t}", "bbox": [t * 100, 0, t * 100 + 10, 10], "src": "vlm"} for t in range(7)]


class CascadeBatching(unittest.TestCase):
    def test_groups_of_three_then_remainder_with_right_offsets(self):
        slow = FakeVLM()
        got = run_page(slow, 3)
        self.assertEqual(slow.batch_calls, [[0, 1, 2], [3, 4, 5], [6]])
        self.assertEqual(slow.spot_calls, [])
        self.assertEqual(got, expected_vlm())

    def test_progress_counts_tiles_per_group_and_is_monotonic(self):
        ticks, fractions = [], []
        run_page(FakeVLM(), 3, on_tile=lambda n, total, count: ticks.append((n, total, count)),
                 on_fraction=fractions.append)
        self.assertEqual([t[0] for t in ticks], [3, 6, 7])
        self.assertTrue(all(t[1] == 7 for t in ticks))
        self.assertEqual([t[2] for t in ticks], sorted(t[2] for t in ticks))
        self.assertEqual(ticks[-1][2], 7)
        self.assertAlmostEqual(fractions[0], PASS1_SHARE)
        self.assertTrue(all(a < b for a, b in zip(fractions, fractions[1:])))
        self.assertAlmostEqual(fractions[-1], 1.0)

    def test_batch_of_one_uses_spot_per_tile(self):
        slow = SpotOnly()
        ticks = []
        got = run_page(slow, 1, on_tile=lambda n, total, count: ticks.append(n))
        self.assertEqual(slow.spot_calls, list(range(7)))
        self.assertEqual(ticks, list(range(1, 8)))
        self.assertEqual(got, expected_vlm())

    def test_engine_without_spot_batch_uses_spot(self):
        slow = SpotOnly()
        self.assertEqual(run_page(slow, 4), expected_vlm())
        self.assertEqual(slow.spot_calls, list(range(7)))

    def test_failed_batch_falls_back_to_one_tile_at_a_time(self):
        slow = FakeVLM(fail_batches=1)
        got = run_page(slow, 3)
        self.assertEqual(slow.batch_calls, [[0, 1, 2], [3, 4, 5], [6]])   # the failed group is not retried batched
        self.assertEqual(slow.spot_calls, [0, 1, 2])
        self.assertEqual(got, expected_vlm())


class HunyuanSpotBatch(unittest.TestCase):
    PROMPT_TOKEN = 1   # chr(1) shows up in the output if a row is decoded from column 0 instead of the prompt end

    class Tok:
        padding_side = "right"

    class Rows(list):   # stands in for the batch tensor: spot_batch reads only .shape
        @property
        def shape(self):
            return (len(self), len(self[0]))

    class Batch(dict):
        def to(self, _dev):
            return self

    class Out:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):   # out[:, start:]
            _, sl = key
            return [r[sl] for r in self.rows]

    class Processor:
        def __init__(self):
            self.tokenizer = HunyuanSpotBatch.Tok()
            self.msgs = None

        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, return_dict, return_tensors, padding):
            assert add_generation_prompt and tokenize and return_dict and padding and return_tensors == "pt"
            assert self.tokenizer.padding_side == "left"   # generated tokens must start at one column for every row
            self.msgs = msgs
            # prompt length grows with the image width, so the rows differ and get padded
            rows = [[HunyuanSpotBatch.PROMPT_TOKEN] * (3 + m[0]["content"][0]["image"].width // 100) for m in msgs]
            width = max(len(r) for r in rows)
            ids = [[0] * (width - len(r)) + r for r in rows]
            mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
            return HunyuanSpotBatch.Batch(input_ids=HunyuanSpotBatch.Rows(ids), attention_mask=mask)

        def batch_decode(self, rows, skip_special_tokens):
            assert skip_special_tokens
            return ["".join(chr(t) for t in r if t != 0) for r in rows]

    class Model:
        def __init__(self, answers):
            self.answers, self.kwargs = answers, None

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, repetition_penalty):
            self.kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=repetition_penalty)
            gen = [[ord(c) for c in a] for a in self.answers]
            n = max(len(g) for g in gen)
            return HunyuanSpotBatch.Out([row + g + [0] * (n - len(g)) for row, g in zip(input_ids, gen)])

    class Torch:
        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    def test_spot_batch_slices_each_row_after_the_common_prompt(self):
        answers = [
            '[{"box": [0, 0, 500, 500], "text": "甲"}]',
            '[{"box": [100, 200, 300, 400], "text": "B2"}, {"box": [0, 0, 1000, 1000], "text": "3"}]',
            "[]",
        ]
        h = Hunyuan.__new__(Hunyuan)   # skip __init__: it loads the 1B model
        h.torch, h.upscale = self.Torch, 2.0
        h.processor, h.model = self.Processor(), self.Model(answers)
        # boxes are scaled by the ORIGINAL size (the upscaled image is only what the model sees)
        imgs = [Image.new("RGB", (1000, 500)), Image.new("RGB", (150, 250)), Image.new("RGB", (60, 100))]
        with mock.patch("docsearch.engines.device", return_value="cpu"):
            got = h.spot_batch(imgs)

        self.assertEqual(got, [
            [("甲", [0.0, 0.0, 500.0, 250.0], None)],
            [("B2", [15.0, 50.0, 45.0, 100.0], None), ("3", [0.0, 0.0, 150.0, 250.0], None)],
            [],
        ])
        self.assertEqual(h.model.kwargs, dict(max_new_tokens=3000, do_sample=False, repetition_penalty=1.08))
        sizes = [m[0]["content"][0]["image"].size for m in h.processor.msgs]
        self.assertEqual(sizes, [(2000, 1000), (300, 500), (120, 200)])
        self.assertTrue(all(m[0]["content"][1]["text"] == Hunyuan.PROMPT for m in h.processor.msgs))

    def test_empty_batch_makes_no_call(self):
        h = Hunyuan.__new__(Hunyuan)
        self.assertEqual(h.spot_batch([]), [])


if __name__ == "__main__":
    unittest.main()
