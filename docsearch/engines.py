"""OCR engines. Every engine reads one image tile and returns text lines with boxes and, when it has one, a confidence:

    engine.spot(PIL tile) -> [(text, [x0, y0, x1, y1], score | None)]      (pixels of that tile; score 0..1)

and declares how the page has to be cut for it: `tile`, `overlap` (page px) and `upscale` (applied inside `spot`).

  PPOCR      PP-OCR family: text detector + text-line recogniser, ONNX Runtime via RapidOCR (v5 server, v6 tiny/small/medium)
  PaddleVL   PaddleOCR-VL-1.6, 0.9B document-parsing VLM, native "Spotting:" task
  Hunyuan    HunyuanOCR, 1B end-to-end OCR VLM, native text-spotting task
  OpenAIVLM any OpenAI-compatible vision server (Qwen-VL spotting JSON, dots.mocr quads); DOCSEARCH_VLM_URL points at it
  Cascade    (docsearch/cascade.py) PP-OCR on the whole page, a VLM only where PP-OCR is unsure; reads whole pages itself

Which one is the default, and why, is decided by the bake-off in `docsearch.modelsel` (see MODEL_SELECTION.md).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from . import config
from .device import device, empty_cache

LOC_RE = re.compile(r"<\|LOC_(\d+)\|>")

_TEX = [(r"\s*\^\{\\(?:ast|prime)\}\s*", '"'),   # HunyuanOCR writes the inch mark of `6"PVC` as 6 ^{\ast} PVC
        (r"\\phi", "φ"), (r"\\varphi", "φ"), (r"\\geq?", "≧"), (r"\\leq?", "≦"), (r"\\times", "×"), (r"\\div", "÷"),
        (r"\\approx", "≒"), (r"\\fallingdotseq", "≒"), (r"\\%", "%"), (r"\\circ", "°"), (r"\\mathrm\{([^}]*)\}", r"\1"),
        (r"\\text\{([^}]*)\}", r"\1"), (r"\^\{?(\d)\}?", r"\1"), (r"_\{?(\d)\}?", r"\1")]


def clean_text(t: str) -> str:
    """VLMs trained for document parsing write units and Greek letters as inline LaTeX (\\( m^3 \\)); undo it."""
    if "\\" not in t and "^" not in t:
        return t.strip()
    t = t.replace("\\(", " ").replace("\\)", " ").replace("$", "")
    for pat, rep in _TEX:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


def parse_spotting(raw: str):
    """'text<|LOC_1|>...<|LOC_8|>text<|LOC_..|>' -> [(text, [x0,y0,x1,y1])] on the model's 0..1000 grid."""
    out = []
    parts = re.split(r"((?:<\|LOC_\d+\|>){4,})", raw)
    for i in range(0, len(parts) - 1, 2):
        text = clean_text(parts[i])
        locs = [int(v) for v in LOC_RE.findall(parts[i + 1])]
        if text and len(locs) >= 4:
            xs, ys = locs[0::2], locs[1::2]
            out.append((text, [min(xs), min(ys), max(xs), max(ys)]))
    return out


def parse_json_spotting(raw: str, size):
    """Tolerant parse of a (possibly truncated) JSON array of {"box":[...], "text": "..."} on a 0..1000 grid."""
    items = []
    for m in re.finditer(r'\{\s*"box"\s*:\s*\[([^\]]*)\]\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
        try:
            b = [float(v) for v in m.group(1).split(",")]
            if len(b) == 4:
                items.append((clean_text(json.loads(f'"{m.group(2)}"')), [b[0] * size[0] / 1000, b[1] * size[1] / 1000, b[2] * size[0] / 1000, b[3] * size[1] / 1000]))
        except Exception:
            continue
    return items


_QWEN_OBJ = re.compile(r"\{[^{}]*\}")


def parse_qwen_spotting(raw: str, size):
    """Qwen-VL grounding JSON: [{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}, ...] on a 0..1000 grid of the sent image.
    Works on truncated output: only complete objects are kept."""
    items = []
    for m in _QWEN_OBJ.finditer(re.sub(r"```(?:json)?", "", raw)):
        try:
            o = json.loads(m.group())
            b = [float(v) for v in o.get("bbox_2d", o.get("bbox"))]
            text = clean_text(str(o.get("text_content", o.get("text")) or ""))
        except (ValueError, TypeError):
            continue
        if len(b) == 4 and text:
            items.append((text, [b[0] * size[0] / 1000, b[1] * size[1] / 1000, b[2] * size[0] / 1000, b[3] * size[1] / 1000]))
    return items


_N = r"(\d+(?:\.\d+)?)"
_PAIR = rf"\(\s*{_N}\s*,\s*{_N}\s*\)"
_QUAD = rf"{_PAIR}\s*,\s*{_PAIR}\s*,\s*{_PAIR}\s*,\s*{_PAIR}"
_DOTS_LINE = re.compile(rf"^[ \t]*\[?{_QUAD}\s*\]?[ \t]*(.*)$", re.M)
# what mlx_vlm's dots.mocr actually writes: special tokens instead of newlines
_DOTS_TOKENS = re.compile(rf"<\|box_start\|>\s*{_QUAD}\s*<\|box_end\|>\s*<\|ref_start\|>(.*?)<\|ref_end\|>", re.S)


def parse_dots_quads(raw: str, size):
    """dots.mocr: one text line per entry in the sent image's pixels, either
    `<|box_start|>(x1, y1), (x2, y2), (x3, y3), (x4, y4)<|box_end|><|ref_start|>text<|ref_end|>` (the model's own tokens)
    or `(x1, y1), (x2, y2), (x3, y3), (x4, y4) text` per line. Each quad is reduced to its axis-aligned box, clipped to the image."""
    items = []
    for m in (_DOTS_TOKENS if "<|box_start|>" in raw else _DOTS_LINE).finditer(raw):
        g = m.groups()
        text = clean_text(g[8])
        if text:
            xs, ys = [float(v) for v in g[0:8:2]], [float(v) for v in g[1:8:2]]
            items.append((text, [max(0, min(xs)), max(0, min(ys)), min(size[0], max(xs)), min(size[1], max(ys))]))
    return items


class OpenAIVLM:
    """Any OpenAI-compatible vision server (mlx_vlm.server, llama-server, vLLM): one tile per chat request, the answer
    parsed by `parser`. The server is DOCSEARCH_VLM_URL; DOCSEARCH_VLM_MODEL, when set, overrides the model id."""
    # ponytail: tiles go one at a time; ceiling = the server's single-stream speed. Upgrade: send tiles concurrently if the server batches.

    def __init__(self, model: str, prompt: str, parser, tile: int = 640, overlap: int = 120, upscale: float = 2.0, multiple: int = 0):
        self.model = os.environ.get("DOCSEARCH_VLM_MODEL") or model
        self.url = os.environ.get("DOCSEARCH_VLM_URL", "http://127.0.0.1:8082/v1").rstrip("/")
        self.prompt, self.parser = prompt, parser
        self.tile, self.overlap, self.upscale, self.multiple = tile, overlap, upscale, multiple

    def spot(self, img: Image.Image):
        big = _up(img, self.upscale)
        if self.multiple:   # dots.mocr wants both sides on its patch grid; boxes map back by the exact scale
            m = self.multiple
            big = big.resize((max(m, round(big.width / m) * m), max(m, round(big.height / m) * m)), Image.LANCZOS)
        sx, sy = img.width / big.width, img.height / big.height
        return [(t, [a * sx, b * sy, c * sx, d * sy], None) for t, (a, b, c, d) in self.parser(self._chat(big), big.size)]

    def _chat(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": self.model,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": self.prompt}]}],
                "temperature": 0, "max_tokens": 3000, "chat_template_kwargs": {"enable_thinking": False}}
        url = f"{self.url}/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"VLM server {url} returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot reach VLM server at {url}: {exc}") from exc
        content = payload["choices"][0]["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()

    def close(self):
        """Nothing to free here: the server owns the weights."""


QWEN_SPOT = ("Spotting all the text in the image with line-level, and output in JSON format as "
             "[{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...].")


def _up(img: Image.Image, f: float) -> Image.Image:
    return img if f == 1.0 else img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)


# --------------------------------------------------------------------------------------------- PP-OCR
class PPOCR:
    def __init__(self, version="v6", size="medium", tile=1280, overlap=160, upscale=1.0, unclip=None, box_thresh=None, det_limit=2600):
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        self.tile, self.overlap, self.upscale = tile, overlap, upscale
        ver = {"v5": OCRVersion.PPOCRV5, "v6": OCRVersion.PPOCRV6}[version]
        mt = {"server": ModelType.SERVER, "tiny": ModelType.TINY, "small": ModelType.SMALL, "medium": ModelType.MEDIUM}[size]
        root = config.CACHE / "rapidocr"
        root.mkdir(parents=True, exist_ok=True)
        params = {
            "Global.model_root_dir": str(root),
            "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangDet.CH, "Det.model_type": mt, "Det.ocr_version": ver,
            "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.lang_type": LangRec.CH if version == "v5" else LangRec.CHINESE_CHT, "Rec.model_type": mt, "Rec.ocr_version": ver,
            "Global.use_cls": False, "Det.limit_side_len": det_limit, "Det.limit_type": "max",
        }
        if unclip:
            params["Det.unclip_ratio"] = unclip
        if box_thresh:
            params["Det.box_thresh"] = box_thresh
        self.engine = RapidOCR(params=params)

    def spot(self, img: Image.Image):
        r = self.engine(np.asarray(_up(img, self.upscale).convert("RGB"))[:, :, ::-1])
        if r.boxes is None or r.txts is None:
            return []
        out = []
        for box, txt, score in zip(r.boxes, r.txts, r.scores):
            b = np.asarray(box) / self.upscale
            out.append((txt, [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())], float(score)))
        return out

    def warm(self):
        """RapidOCR loads (and, first time, downloads) its ONNX models lazily; do it now so `prefetch` really fetches them."""
        self.engine._load_det_model()
        self.engine._load_rec_model()

    def close(self):
        self.engine = None


# --------------------------------------------------------------------------------------------- VLMs
class PaddleVL:
    tile, overlap, upscale = 640, 120, 2.0   # model card: spotting inputs < 1500 px are upscaled 2x

    def __init__(self, model_id: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = model_id or config.PADDLEVL_ID
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device()).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, image: Image.Image, prompt: str = "Spotting:", max_new_tokens: int = 2048) -> str:
        max_pixels = 2048 * 28 * 28   # model card: spotting uses ~1.6 Mpx instead of the 1 Mpx default
        ip = self.processor.image_processor
        min_pixels = getattr(ip, "min_pixels", None) or ip.size.shortest_edge   # attribute moved between transformers 5.x releases
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": max_pixels}},
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][inputs["input_ids"].shape[-1]:-1])

    def spot(self, img: Image.Image):
        sx, sy = img.width / 1000, img.height / 1000
        return [(t, [a * sx, b * sy, c * sx, d * sy], None) for t, (a, b, c, d) in parse_spotting(self.run(_up(img, self.upscale)))]

    def close(self):
        import gc
        del self.model
        gc.collect()
        empty_cache()


class Hunyuan:
    tile, overlap, upscale = 640, 120, 2.0
    PROMPT = ("检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 输出格式为 JSON 数组，每个元素必须包含："
              '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；"text": "识别出的文字内容"。 '
              "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。")

    def __init__(self, model_id: str = "tencent/HunyuanOCR"):
        import torch
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True).to(device()).eval()

    def spot(self, img: Image.Image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": _up(img, self.upscale)}, {"type": "text", "text": self.PROMPT}]}]
        inp = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device())
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=3000, do_sample=False, repetition_penalty=1.08)
        raw = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return [(t, box, None) for t, box in parse_json_spotting(raw, img.size)]

    def spot_batch(self, imgs: list[Image.Image]):
        """`spot` for several tiles in one generate() call. Prompts are LEFT-padded, so every row's generated tokens start
        at the same column (the padded prompt length); a row that stops early is padded with pad tokens, which decode away."""
        if not imgs:
            return []
        msgs = [[{"role": "user", "content": [{"type": "image", "image": _up(img, self.upscale)}, {"type": "text", "text": self.PROMPT}]}] for img in imgs]
        self.processor.tokenizer.padding_side = "left"
        inp = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to(device())
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=3000, do_sample=False, repetition_penalty=1.08)
        raws = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return [[(t, box, None) for t, box in parse_json_spotting(raw, img.size)] for raw, img in zip(raws, imgs)]

    def read(self, img: Image.Image, prompt: str = "提取图中的文字。", max_new_tokens: int = 64) -> str:
        """Plain text recognition of one line crop (HunyuanOCR's own `structured_parse` prompt), not the spotting JSON."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inp = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device())
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.08)
        return clean_text(self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])

    def close(self):
        import gc
        del self.model
        gc.collect()
        empty_cache()


def _cascade(fast: str, slow: str):
    from .cascade import Cascade
    return Cascade(fast, slow)


# Named configurations = the candidates compared in `docsearch.modelsel`; DOCSEARCH_OCR_ENGINE picks one for ingestion.
REGISTRY = {
    "paddlevl": PaddleVL,
    "hunyuan": Hunyuan,
    "ppocrv5": lambda: PPOCR("v5", "server"),
    # same model with tiles up-scaled 2x like the VLMs get, and a looser detector (boxes were clipping the first glyph)
    "ppocrv5-x2": lambda: PPOCR("v5", "server", tile=640, overlap=120, upscale=2.0, unclip=2.0),
    "ppocrv6-m": lambda: PPOCR("v6", "medium"),
    "ppocrv6-m-x2": lambda: PPOCR("v6", "medium", tile=640, overlap=120, upscale=2.0, unclip=2.0),
    "ppocrv6-s": lambda: PPOCR("v6", "small"),
    # detector box padding (RapidOCR default 1.6): thin CAD strokes get their first/last glyph clipped by tight boxes
    "ppocrv6-m-u20": lambda: PPOCR("v6", "medium", unclip=2.0),
    "ppocrv6-m-u25": lambda: PPOCR("v6", "medium", unclip=2.5),
    "ppocrv5-u20": lambda: PPOCR("v5", "server", unclip=2.0),
    # PP-OCR on the whole page; HunyuanOCR re-reads only the tiles where PP-OCR's line confidence is low (cascade.py)
    "hybrid": lambda: _cascade("ppocrv6-m-u20", "hunyuan"),
    # OpenAI-compatible servers (OpenAIVLM): DOCSEARCH_VLM_URL / DOCSEARCH_VLM_MODEL select the running server and model
    "qwen3vl4b": lambda: OpenAIVLM("mlx-community/Qwen3-VL-4B-Instruct-8bit", QWEN_SPOT, parse_qwen_spotting),
    "qwen35-9b": lambda: OpenAIVLM("mlx-community/Qwen3.5-9B-4bit", QWEN_SPOT, parse_qwen_spotting),
    "dotsmocr": lambda: OpenAIVLM("mlx-community/dots.mocr-8bit", "Detect and recognize the text in the image.", parse_dots_quads, tile=644, multiple=28),
}


def make(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"unknown OCR engine {name!r}; choose from {', '.join(REGISTRY)}")
    return REGISTRY[name]()
