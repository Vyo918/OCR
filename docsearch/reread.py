"""Crop re-read: a vision reader looks again at the OCR lines a question's evidence rests on, on the drawing image.

Only lines whose digits the reader and OCR disagree on reach the prompt (`format_block`); everything else leaves the prompt
byte-identical. Off unless DOCSEARCH_REREAD is `openai` or `hunyuan` (config.REREAD).

  select    fact-row source lines first, then short digit lines of the evidence blocks; same-row neighbours merged
  crop      page pixels around a target, scaled so small CAD text reads
  reconcile normalised OCR vs reader text -> agree | disagree | rejected (guards against hallucinated digits)
  Rereader  disk cache keyed backend|model|doc|bbox, then the backend (openai-compatible server or in-process Hunyuan)
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import statistics
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from rapidfuzz import fuzz

from . import config, index
from .cascade import looped

PROMPT = ("This image is a crop of one text line from a scanned Taiwanese construction drawing (Traditional Chinese, CAD font). "
          "Transcribe exactly the text you see, left to right. Keep every digit, decimal point, inch mark (\"), ×, φ, units and "
          "brackets as printed. Output only the text. Write ? for a character you cannot read.")
HEADER = ("Re-read of drawing crops (a vision model looked at these OCR lines on the drawing image again; listed only where its "
          "digits differ from the OCR text):")
DIGIT = re.compile(r"\d")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
_QUOTES = str.maketrans({"″": '"', "”": '"', "“": '"'})
_PAIRED = ("\"", "'", "「", "」")
_TIMES = re.compile(r"(?<=\d)\s*[xX*]\s*(?=\d)")
_SKELETON = re.compile(r"[\d\W_]+")


# ----------------------------------------------------------------------------------------------- selection
def _cx(b) -> float:
    return (b[0] + b[2]) / 2


def _near(a, b) -> bool:
    """Same text row (vertical overlap >= half the shorter line) and a horizontal gap of at most 1.5 line heights."""
    ha, hb = a[3] - a[1], b[3] - b[1]
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    gap = max(b[0] - a[2], a[0] - b[2], 0)
    return overlap >= 0.5 * min(ha, hb) and gap <= 1.5 * max(ha, hb)


def _row_group(seed: dict, lines: list[dict]) -> list[dict]:
    """The seed plus same-row lines of its chunk, chained outwards to the left and right (header cell + value)."""
    group = [seed]
    grew = True
    while grew:
        grew = False
        left = min(group, key=lambda l: _cx(l["bbox"]))
        right = max(group, key=lambda l: _cx(l["bbox"]))
        for ln in lines:
            if any(ln is g for g in group):
                continue
            if (_cx(ln["bbox"]) < _cx(left["bbox"]) and _near(ln["bbox"], left["bbox"])) or \
               (_cx(ln["bbox"]) > _cx(right["bbox"]) and _near(ln["bbox"], right["bbox"])):
                group.append(ln)
                grew = True
                break
    return sorted(group, key=lambda l: l["bbox"][0])


def select(ev: list[dict], fx: dict | None, by_id: dict, limit: int) -> list[dict]:
    """Up to `limit` targets, deterministic: fact-row sources (row order), then short digit lines of block chunks (rank order)."""
    rows = (fx or {}).get("rows", [])
    rows_of: dict = {}
    seeds = []
    for row in rows:
        for src in row.get("src", []):
            key = (src["chunk"], tuple(src["bbox"]))
            rows_of.setdefault(key, []).append(row["id"])
            chunk = by_id.get(src["chunk"])
            for ln in (chunk or {}).get("lines", []):
                if tuple(ln["bbox"]) == key[1] and DIGIT.search(ln["text"]):
                    seeds.append((chunk, ln))
    for chunk in ev:
        if chunk.get("kind") == "block":
            for ln in chunk.get("lines", []):
                if DIGIT.search(ln["text"]) and len(ln["text"].strip()) <= 40:
                    seeds.append((chunk, ln))

    out, used = [], set()
    for chunk, ln in seeds:
        if len(out) >= limit:
            break
        if (chunk["doc"], tuple(ln["bbox"])) in used:
            continue
        group = _row_group(ln, chunk.get("lines", []))
        used.update((chunk["doc"], tuple(g["bbox"])) for g in group)
        out.append(_target(chunk, group, rows_of))
    return out


def _target(chunk: dict, group: list[dict], rows_of: dict) -> dict:
    rows = list(dict.fromkeys(r for l in group for r in rows_of.get((chunk["id"], tuple(l["bbox"])), [])))
    box = [min(l["bbox"][0] for l in group), min(l["bbox"][1] for l in group),
           max(l["bbox"][2] for l in group), max(l["bbox"][3] for l in group)]
    return {"doc": chunk["doc"], "chunk": chunk["id"], "rows": rows, "box": box,
            "h": statistics.median(l["bbox"][3] - l["bbox"][1] for l in group),
            "lines": [{"text": l["text"], "bbox": list(l["bbox"])} for l in group],
            "text": " ".join(l["text"] for l in group)}


# ----------------------------------------------------------------------------------------------- crop
def crop(page: Image.Image, box, h: float) -> Image.Image:
    """Target box plus padding, scaled so the median line is at least 48 px (max 4x), long side capped at 1280 px."""
    h = max(float(h), 1.0)
    pad = max(8, round(0.6 * h))
    W, H = page.size
    x0, y0 = max(0, math.floor(box[0]) - pad), max(0, math.floor(box[1]) - pad)
    x1, y1 = min(W, math.ceil(box[2]) + pad), min(H, math.ceil(box[3]) + pad)
    img = page.crop((x0, y0, x1, y1))
    f = min(4.0, max(1.0, 48 / h))
    img = img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
    long = max(img.size)
    if long > 1280:
        s = 1280 / long
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img.convert("RGB")


# ----------------------------------------------------------------------------------------------- reconcile
def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").translate(_QUOTES)).strip()   # quotes first: NFKC would split ″ into two primes
    s = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in _PAIRED:
        s = s[1:-1]
    s = _TIMES.sub("×", s)
    return re.sub(r"\s+", " ", s).strip()


def digits(s: str) -> list[str]:
    return NUMBER.findall(s)


def reconcile(ocr: str, vlm: str) -> str:
    """'agree' | 'disagree' (both readings shown, the reader's digits are used) | 'rejected' (reader output not trusted)."""
    o, v = normalise(ocr), normalise(vlm)
    if not v or looped(v) or len(v) > 2 * len(o) + 8:
        return "rejected"
    do, dv = digits(o), digits(v)
    if not dv:
        return "rejected"
    if abs(sum(map(len, do)) - sum(map(len, dv))) > 2 or abs(len(do) - len(dv)) > 1:
        return "rejected"
    so, sv = _SKELETON.sub("", o), _SKELETON.sub("", v)
    if len(so) >= 4 and len(sv) >= 4 and fuzz.ratio(index.fold(so), index.fold(sv)) < 50:
        return "rejected"
    return "agree" if do == dv else "disagree"


# ----------------------------------------------------------------------------------------------- backends
def _openai_read(img: Image.Image) -> str:
    """One non-streaming OpenAI-compatible chat call (mlx_vlm.server / llama-server with an image part)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": config.REREAD_MODEL,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": PROMPT}]}],
            "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}
    url = f"{config.REREAD_URL}/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"re-read server {url} returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot reach re-read server at {url}: {exc}") from exc
    content = payload["choices"][0]["message"].get("content") or ""
    return re.sub(r"^\s*<think>.*?</think>", "", content, flags=re.S).strip()


def resolve_image(p: str) -> Path:
    """Index paths are container paths (/data/work/pages/x.jpg); locally the page sits in config.PAGES by name."""
    path = Path(p)
    if path.exists():
        return path
    alt = config.PAGES / path.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"page image not found: {p} (also tried {alt})")


def format_block(records: list[dict]) -> str | None:
    lines = [f"[{r['id']}] {r['chunk']} | " + (f"rows {', '.join(r['rows'])} | " if r["rows"] else "")
             + f"OCR: {r['ocr']} | re-read: {r['vlm']}" for r in records if r["id"]]
    return "\n".join([HEADER, *lines]) if lines else None


class Rereader:
    def __init__(self, docs: dict, chunks: list[dict]):
        self.docs = docs
        self.by_id = {c["id"]: c for c in chunks}
        self.backend = config.REREAD
        self.model = config.HUNYUAN_ID if self.backend == "hunyuan" else config.REREAD_MODEL
        self._pages: dict = {}
        self._engine = None
        self._cache = self._load()

    @staticmethod
    def _cache_path() -> Path:
        return config.WORK / "reread" / "cache.json"

    def _load(self) -> dict:
        path = self._cache_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _save(self) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def _page(self, doc: str) -> Image.Image:
        if doc not in self._pages:
            with Image.open(resolve_image(self.docs[doc]["image"])) as im:
                self._pages[doc] = im.convert("RGB")
        return self._pages[doc]

    def _read(self, img: Image.Image) -> str:
        if self.backend == "openai":
            return _openai_read(img)
        if self._engine is None:
            from . import engines
            self._engine = engines.Hunyuan()
        return self._engine.read(img)

    def read_target(self, target: dict) -> dict:
        """-> {text, seconds, cached}. The backend (and its model) is touched only on a cache miss."""
        key = f"{self.backend}|{self.model}|{target['doc']}|{','.join(f'{v:g}' for v in target['box'])}"
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        img = crop(self._page(target["doc"]), target["box"], target["h"])
        t0 = time.time()
        entry = {"text": self._read(img), "seconds": round(time.time() - t0, 2)}
        self._cache[key] = entry
        self._save()
        return {**entry, "cached": False}

    def run(self, ev: list[dict], fx: dict | None) -> dict:
        records = []
        for t in select(ev, fx, self.by_id, config.REREAD_MAX):
            r = self.read_target(t)
            records.append({"id": None, "doc": t["doc"], "chunk": t["chunk"], "rows": t["rows"], "box": t["box"],
                            "ocr": t["text"], "vlm": r["text"], "status": reconcile(t["text"], r["text"]),
                            "seconds": r["seconds"], "cached": r["cached"]})
        n = 0
        for rec in records:
            if rec["status"] == "disagree":
                n += 1
                rec["id"] = f"R{n}"
        return {"text": format_block(records), "records": records,
                "evidence": [rec["vlm"] for rec in records if rec["status"] == "disagree"]}
