"""Stage 1 - OCR ingest.

Scanned engineering drawings (A3 @ 300 dpi, ~5000x3500 px) are cut into overlapping tiles; every non-blank tile goes
through the configured OCR engine (`docsearch.engines`, chosen by the bake-off in MODEL_SELECTION.md), which returns
text lines *with* boxes. Tile boxes are mapped back to page space and lines duplicated by the overlap are merged.

Output: <WORK>/ocr/<doc>.lines.json = {doc, image, size, lines:[{doc, text, bbox:[x0,y0,x1,y1]}]}
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, engines, progress

Image.MAX_IMAGE_PIXELS = None

MIN_INK = 0.004     # skip tiles that are (almost) blank paper


# --------------------------------------------------------------------------- pages
def _native_scan(src: Path, page: int, pages_dir: Path, name: str) -> Path | None:
    """A scanned PDF page is one embedded image. Extracting it avoids a lossy re-rasterisation; None if the page is not that."""
    rows = subprocess.run(["pdfimages", "-list", "-f", str(page), "-l", str(page), str(src)], capture_output=True, text=True).stdout.splitlines()[2:]
    imgs = [r.split() for r in rows if len(r.split()) >= 14 and r.split()[2] == "image"]
    if len(imgs) != 1 or float(imgs[0][12]) < 150 or imgs[0][8] not in ("jpeg", "image", "ccitt", "jpx"):
        return None
    prefix = pages_dir / f"{name}.native"
    flag = "-j" if imgs[0][8] == "jpeg" else "-png"
    subprocess.run(["pdfimages", flag, "-f", str(page), "-l", str(page), str(src), str(prefix)], check=True)
    got = sorted(pages_dir.glob(f"{name}.native-*"))
    return got[0] if got else None


def page_images(src: Path, pages_dir: Path) -> list[tuple[str, Path]]:
    """[(doc_name, image_path)] - one entry per page. Images are used as is; PDFs give their embedded scan or a 300 dpi render."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in config.IMAGE_EXT:
        return [(src.stem, src)]
    info = subprocess.run(["pdfinfo", str(src)], capture_output=True, text=True, check=True).stdout
    n = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    out = []
    for p in range(1, n + 1):
        name = src.stem if n == 1 else f"{src.stem}-p{p}"
        done = sorted(pages_dir.glob(f"{name}.native-*")) or sorted(pages_dir.glob(f"{name}.png"))
        if done and done[0].stat().st_mtime < src.stat().st_mtime:   # the PDF was replaced: drop the stale extraction
            for f in done:
                f.unlink()
            done = []
        if not done:
            native = _native_scan(src, p, pages_dir, name)
            if native is None:
                png = pages_dir / f"{name}.png"
                subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "-f", str(p), "-l", str(p), str(src), str(png.with_suffix(""))], check=True)
                native = png
            done = [native]
        out.append((name, done[0]))
    return out


def tiles(w: int, h: int, tile: int, overlap: int):
    step = tile - overlap
    for y in range(0, max(h - overlap, 1), step):
        for x in range(0, max(w - overlap, 1), step):
            yield x, y, min(x + tile, w), min(y + tile, h)


def ink_ratio(tile: Image.Image) -> float:
    return float((np.asarray(tile.convert("L")) < 170).mean())


# ------------------------------------------------------------------------ overlap merge
def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(a, b) -> float:
    """Share of box `a` that lies inside box `b`."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def merge_overlap(lines: list[dict]) -> list[dict]:
    """Drop lines read twice because tiles overlap. The longer reading wins; a shorter one is a duplicate when its
    box coincides with a kept box (IoU > 0.5) or lies mostly inside it and its text is part of the kept text
    (a tile edge cut the line, e.g. '三號基地' next to '桃園市八德區三號基地')."""
    from rapidfuzz import fuzz

    kept: list[dict] = []
    squash = lambda s: re.sub(r"\s+", "", s).lower()
    for ln in sorted(lines, key=lambda l: -len(l["text"])):
        a, dup = squash(ln["text"]), False
        for k in kept:
            kb = k["bbox"]
            if ln["bbox"][2] < kb[0] or ln["bbox"][0] > kb[2] or ln["bbox"][3] < kb[1] or ln["bbox"][1] > kb[3]:
                continue
            if iou(ln["bbox"], kb) > 0.5 or (inside(ln["bbox"], kb) >= 0.6 and (a in squash(k["text"]) or fuzz.partial_ratio(a, squash(k["text"])) >= 85)):
                dup = True
                break
        if not dup:
            kept.append(ln)
    return sorted(kept, key=lambda l: (l["bbox"][1], l["bbox"][0]))


# ------------------------------------------------------------------------ driver
def read_page(page: Image.Image, engine, name: str = "", limit: int | None = None, on_tile=None) -> list[dict]:
    """All text lines of one page image (page pixel coordinates), overlap-merged."""
    if hasattr(engine, "read_page"):   # engines that organise the whole page themselves (the PP-OCR + VLM cascade)
        return engine.read_page(page, name, limit, on_tile)
    W, H = page.size
    grid = list(tiles(W, H, engine.tile, engine.overlap))
    todo = [t for t in grid if ink_ratio(page.crop(t)) >= MIN_INK][:limit]
    lines: list[dict] = []
    for n, (x0, y0, x1, y1) in enumerate(todo, 1):
        for text, (a, b, c, d), score in engine.spot(page.crop((x0, y0, x1, y1))):
            line = {"doc": name, "text": text, "bbox": [round(x0 + a), round(y0 + b), round(x0 + c), round(y0 + d)]}
            if score is not None:
                line["score"] = round(score, 3)
            lines.append(line)
        if on_tile:
            on_tile(n, len(todo), len(lines))
    return merge_overlap(lines)


def ingest_image(name: str, img_path: Path, engine, out_dir: Path, limit: int | None = None) -> Path:
    page = Image.open(img_path).convert("RGB")
    t0 = time.time()
    lines = read_page(page, engine, name, limit,
                      on_tile=lambda n, total, k: print(f"  [{name}] tile {n}/{total} -> {k} lines [{time.time() - t0:.0f}s]", flush=True) if n % 5 == 0 or n == total else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.lines.json"
    out.write_text(json.dumps({"doc": name, "engine": config.OCR_ENGINE, "image": str(img_path), "size": list(page.size), "lines": lines}, ensure_ascii=False, indent=1))
    print(f"[{name}] {page.size[0]}x{page.size[1]} -> {out} ({len(lines)} lines, {time.time() - t0:.0f}s)", flush=True)
    return out


def collect_inputs(paths: list[Path]) -> list[Path]:
    """Files as given; folders are scanned (non-recursive) for PDFs and images."""
    found: list[Path] = []
    for p in paths or [config.INPUT]:
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() == ".pdf" or f.suffix.lower() in config.IMAGE_EXT)
        elif p.exists():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return found


def is_current(out: Path, img: Path) -> bool:
    """OCR output is reusable when it is newer than the page image and came from the configured engine."""
    if not out.exists() or out.stat().st_mtime < img.stat().st_mtime:
        return False
    return json.loads(out.read_text()).get("engine") == config.OCR_ENGINE


def ingest(paths: list[Path], limit: int | None = None, force: bool = False) -> list[Path]:
    """OCR every page of the given PDFs / images (folders allowed). Pages already read by the same engine are reused."""
    srcs = collect_inputs(paths)
    if not srcs:
        raise SystemExit("no PDF/image found to ingest")
    engine, outs = None, []
    try:
        for src in srcs:
            for name, img in page_images(src, config.PAGES):
                out = config.OCR_DIR / f"{name}.lines.json"
                if not force and limit is None and is_current(out, img):
                    print(f"[{name}] OCR cached ({config.OCR_ENGINE}) -> {out}", flush=True)
                    outs.append(out)
                    continue
                if engine is None:
                    print(f"[ocr] engine = {config.OCR_ENGINE}", flush=True)
                    engine = engines.make(config.OCR_ENGINE)
                outs.append(ingest_image(name, img, engine, config.OCR_DIR, limit))
                progress.report('ocr', len(outs), len(srcs), 'pages')
    finally:
        if engine is not None:
            engine.close()
    return outs
