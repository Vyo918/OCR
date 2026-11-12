"""PP-OCR on the whole page, a VLM only where PP-OCR is unsure.

Why (MODEL_SELECTION.md): PP-OCRv6 is fast, has the best digit accuracy and never invents text, but it garbles the tiny coloured
pipe labels (`4"PVC連通管x1支(壁上方)` -> `4PvC延载警x1文(信上方)`); HunyuanOCR reads those labels best but needs ~15 s per
640 px tile and is weak on tables and formulas. The recogniser's own line confidence tells the two situations apart (garbled
labels score 0.58-0.73, correct ones 0.83-0.97), so:

  1. PP-OCR reads every tile of the page -> lines with a confidence score.
  2. VLM tiles that hold >= `min_lines` lines below `min_score` are re-read by the VLM.
  3. Arbitration per VLM line: no PP-OCR line under it -> add it (PP-OCR missed the text); only unsure PP-OCR lines
     under it -> replace them; a confident PP-OCR line under it -> keep PP-OCR, drop the VLM line.
"""
from __future__ import annotations

import time

from PIL import Image

from .engines import make
from .ocr_ingest import inside, iou, merge_overlap, read_page, tiles


def looped(text: str) -> bool:
    """A VLM stuck in a repetition loop writes the same few characters over and over."""
    return len(text) >= 12 and len(set(text.replace(" ", ""))) <= max(2, len(text) // 8)


class Cascade:
    def __init__(self, fast: str = "ppocrv6-m-u20", slow: str = "hunyuan", min_score: float = 0.75, min_lines: int = 2):
        self.fast, self._slow_name = make(fast), slow
        self.slow = None                      # loaded on first use: pages PP-OCR reads confidently never touch the GPU model
        self.min_score, self.min_lines = min_score, min_lines
        self.tile, self.overlap = self.fast.tile, self.fast.overlap
        self.stats: dict = {}

    def read_page(self, page: Image.Image, name: str = "", limit: int | None = None, on_tile=None) -> list[dict]:
        t0 = time.time()
        base = read_page(page, self.fast, name, limit)
        shaky = [l for l in base if l.get("score", 1.0) < self.min_score]
        W, H = page.size
        grid = list(tiles(W, H, 640, 120))   # the geometry HunyuanOCR was benchmarked with
        todo = []
        for t in grid:
            n = sum(t[0] <= (l["bbox"][0] + l["bbox"][2]) / 2 < t[2] and t[1] <= (l["bbox"][1] + l["bbox"][3]) / 2 < t[3] for l in shaky)
            if n >= self.min_lines:
                todo.append(t)
        self.stats = {"pp_lines": len(base), "unsure_lines": len(shaky), "vlm_tiles": len(todo), "grid_tiles": len(grid),
                      "pp_seconds": round(time.time() - t0)}
        print(f"  [{name}] cascade: {len(base)} PP-OCR lines, {len(shaky)} below {self.min_score} -> VLM re-reads {len(todo)} of {len(grid)} tiles "
              f"(PP-OCR {self.stats['pp_seconds']}s)", flush=True)
        if not todo:
            return base
        if self.slow is None:
            self.slow = make(self._slow_name)
        vlm: list[dict] = []
        for n, (x0, y0, x1, y1) in enumerate(todo, 1):
            for text, (a, b, c, d), _ in self.slow.spot(page.crop((x0, y0, x1, y1))):
                if not looped(text):
                    vlm.append({"doc": name, "text": text, "bbox": [round(x0 + a), round(y0 + b), round(x0 + c), round(y0 + d)]})
            if on_tile:
                on_tile(n, len(todo), len(vlm))
        out = self.arbitrate(base, merge_overlap(vlm))
        self.stats["seconds"] = round(time.time() - t0)
        return out

    def arbitrate(self, base: list[dict], vlm: list[dict]) -> list[dict]:
        drop: set[int] = set()
        add: list[dict] = []
        n_new = n_swap = n_kept = 0
        for v in vlm:
            under = [i for i, p in enumerate(base) if inside(p["bbox"], v["bbox"]) >= 0.5 or iou(p["bbox"], v["bbox"]) >= 0.3]
            if not under:
                add.append({**v, "src": "vlm"})
                n_new += 1
            elif all(base[i].get("score", 1.0) < self.min_score for i in under):
                drop.update(under)
                add.append({**v, "src": "vlm", "was": " ".join(base[i]["text"] for i in under)})
                n_swap += 1
            else:
                n_kept += 1
        self.stats.update(vlm_lines=len(vlm), vlm_added=n_new, vlm_replaced=n_swap, vlm_dropped=n_kept)
        keep = [p for i, p in enumerate(base) if i not in drop]
        return sorted(keep + add, key=lambda l: (l["bbox"][1], l["bbox"][0]))

    def warm(self):
        self.fast.warm()

    def close(self):
        for e in (self.fast, self.slow):
            if e is not None and hasattr(e, "close"):
                e.close()
