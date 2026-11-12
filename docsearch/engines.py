"""OCR engines. Every engine reads one image tile and returns text lines with boxes and, when it has one, a confidence:

    engine.spot(PIL tile) -> [(text, [x0, y0, x1, y1], score | None)]      (pixels of that tile; score 0..1)

and declares how the page has to be cut for it: `tile`, `overlap` (page px) and `upscale` (applied inside `spot`).

  PPOCR      PP-OCR family: text detector + text-line recogniser, ONNX Runtime via RapidOCR (v5 server, v6 tiny/small/medium)
  PaddleVL   PaddleOCR-VL-1.6, 0.9B document-parsing VLM, native "Spotting:" task
  Hunyuan    HunyuanOCR, 1B end-to-end OCR VLM, native text-spotting task
  Cascade    (docsearch/cascade.py) PP-OCR on the whole page, a VLM only where PP-OCR is unsure; reads whole pages itself

Which one is the default, and why, is decided by the bake-off in `docsearch.modelsel` (see MODEL_SELECTION.md).
"""
from __future__ import annotations

import json
import re

import numpy as np
from PIL import Image

from . import config

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
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
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
        self.torch.cuda.empty_cache()


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
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    def spot(self, img: Image.Image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": _up(img, self.upscale)}, {"type": "text", "text": self.PROMPT}]}]
        inp = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=3000, do_sample=False, repetition_penalty=1.08)
        raw = self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return [(t, box, None) for t, box in parse_json_spotting(raw, img.size)]

    def close(self):
        import gc
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


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
}


def make(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"unknown OCR engine {name!r}; choose from {', '.join(REGISTRY)}")
    return REGISTRY[name]()
