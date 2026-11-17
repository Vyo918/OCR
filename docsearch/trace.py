"""Step-by-step trace of one question through the pipeline, written as one self-contained HTML player.

`collect` runs the production retrieval and prompt code (Engine.expand / lexical / dense / novel, entities.anchor,
factquery.context, Engine.build_prompt) and keeps every stage's input and output. The LLM is never called: the answers
shown are the recorded ones from real runs. `render` writes the data into trace_template.html.

    DOCSEARCH_WORK=<dir with index/ and ocr/> python -m docsearch.trace --question-file Q.json \
        --result-file RUN_WITH_FACTS.json --result-base-file RUN_WITHOUT_FACTS.json --pdf SOURCE.pdf --out work/trace.html
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import calc, config, factquery, timing
from .index import fold, norm, tokens
from .ocr_ingest import MIN_INK, ink_ratio, tiles
from .entities import anchor, chunk_entities, question_entities

TEMPLATE = Path(__file__).with_name("trace_template.html")
PLACEHOLDER = "__TRACE_DATA__"
LEX_K = 60                  # per-ranker depth and candidate pool of Engine.hybrid (its defaults)
RRF_K = 60                  # RRF constant of Engine.hybrid
LIST_N = 10                 # ids shown per ranking
SHEET_BUDGET = 350 * 1024    # bytes of the WebP page image embedded per sheet
SHEET_WIDTH = 1800           # px: the page image is scaled to this width before encoding
SOURCE_PAD = 60              # page px around the source rows: the text rows of the OCR and chunk stages come from that area
VERDICT = {"pass": "correct", "fail": "incorrect"}
PASS_TILE, PASS_OVERLAP = 1280, 160      # PP-OCR pass grid (engines.py ppocrv6-m-u20)
REREAD_TILE, REREAD_OVERLAP = 640, 120   # VLM re-read grid (cascade.py)
UNSURE = 0.75              # cascade.Cascade.min_score: a PP-OCR line below it is unsure
MIN_UNSURE = 2             # cascade.Cascade.min_lines: unsure line centres a tile needs to be re-read
SHOW_RANK = 5              # ranked chunks shown per ranking
THINK_CAP, ANSWER_CAP = 7000, 1500   # Engine.chat defaults (tests check them against the signature)


# ------------------------------------------------------------------ helpers
def fuse(rankings: list[list[int]]) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion with the arithmetic of Engine.hybrid, keeping the scores (hybrid returns only ids)."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


def _text(label: str, text: str) -> dict:
    return {"t": "text", "label": label, "text": str(text)}


def _rows(label: str, rows: list[dict]) -> dict:
    return {"t": "rows", "label": label, "rows": rows or [_row("-", "(none)")]}


def _row(key, text="", meta="", bar: float | None = None, tone: str = "") -> dict:
    return {"key": str(key), "text": str(text), "meta": str(meta), "tone": tone,
            "bar": None if bar is None else round(min(max(float(bar), 0.0), 1.0), 3)}


def _stage(sid: str, short: str, title: str, what: str, inp: list, out: list, view: str = "blocks", *, phase: str = "answer",
           how=(), params=(), why: str = "") -> dict:
    return {"id": sid, "short": short, "title": title, "what": what, "input": inp, "output": out, "view": view,
            "phase": phase, "how": list(how), "params": [list(p) for p in params], "why": why, "ms": None}


def _ids(engine, idx: list[int]) -> str:
    return ", ".join(engine.chunks[i]["id"] for i in idx)


def _query_text(query) -> str:
    return "\n".join(f"{k}: {sorted(v) if isinstance(v, (set, frozenset)) else v}" for k, v in asdict(query).items())


def _diff(a: str, b: str) -> list[list[str]]:
    """Line diff as [tag, line], tag eq | del | ins. Reading the output top to bottom gives the prompt B text; deletions are A-only."""
    al, bl = a.split("\n"), b.split("\n")
    ops: list[list[str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, al, bl, autojunk=False).get_opcodes():
        if tag == "equal":
            ops += [["eq", x] for x in al[i1:i2]]
        else:
            ops += [["del", x] for x in al[i1:i2]]
            ops += [["ins", x] for x in bl[j1:j2]]
    return ops


def _centre_in(bbox: list, box: list) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _overlaps(bbox: list, box: list) -> bool:
    return bbox[0] < box[2] and box[0] < bbox[2] and bbox[1] < box[3] and box[1] < bbox[3]


def _source_area(size: list, boxes: list[list]) -> list[int]:
    """The source rows' box padded by SOURCE_PAD and clamped to the page: the text rows of the OCR and chunk stages come from it."""
    return [max(0, round(min(b[0] for b in boxes) - SOURCE_PAD)), max(0, round(min(b[1] for b in boxes) - SOURCE_PAD)),
            min(size[0], round(max(b[2] for b in boxes) + SOURCE_PAD)), min(size[1], round(max(b[3] for b in boxes) + SOURCE_PAD))]


def _page_path(engine, doc: str, pdfs: list[Path], pages_dir: Path) -> Path | None:
    """The page image the index was built from (docs[doc]["image"]) if it is on this machine, else the page rendered from a
    PDF given with --pdf (the OCR ingest helper, cached in pages_dir)."""
    recorded = Path(engine.docs[doc]["image"])
    if recorded.exists():
        return recorded
    from .ocr_ingest import page_images
    files: list[Path] = []
    for p in pdfs:
        files += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    for pdf in files:
        if doc == pdf.stem or doc.startswith(pdf.stem + "-p"):
            return dict(page_images(pdf, pages_dir)).get(doc)
    return None


def _page_url(path: Path, budget: int) -> str:
    """WebP data URL of the whole page, SHEET_WIDTH px wide (or its own width when smaller): the highest quality step whose
    file fits `budget` bytes, or the lowest step when none does."""
    im = Image.open(path).convert("RGB")
    if im.width > SHEET_WIDTH:
        im = im.resize((SHEET_WIDTH, round(im.height * SHEET_WIDTH / im.width)), Image.LANCZOS)
    for quality in (85, 75, 65, 55, 45, 35):
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=quality, method=4)
        if buf.tell() <= budget:
            break
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def _tile_ink(page: Path | None, size: list) -> list[float] | None:
    """Ink share of each PP-OCR pass tile (the ingest grid), or None when the page image is not on this machine."""
    if not page:
        return None
    img = Image.open(page).convert("L")
    return [ink_ratio(img.crop(t)) for t in tiles(size[0], size[1], PASS_TILE, PASS_OVERLAP)]


def _overlay(engine, doc: str, sources: list[list], page: Path | None, ocr_dir: Path, evid: set[str]) -> dict:
    """One sheet in page px: the whole page image (None when it is not on this machine), every OCR line and chunk of the sheet
    (evidence chunks keep their full text), the source boxes, and the PP-OCR pass tiles: each with its blank verdict (None when
    the page is not here) and the number of OCR lines whose top-left corner starts inside it."""
    size = engine.docs[doc]["size"]
    W, H = size
    rnd = lambda b: [round(v) for v in b]
    lines = _lines_file(ocr_dir, doc)
    ink = _tile_ink(page, size)
    grid = list(tiles(W, H, PASS_TILE, PASS_OVERLAP))
    return {"src": _page_url(page, SHEET_BUDGET) if page else None, "size": [W, H],
            "lines": [[*rnd(ln["bbox"]), ln["text"][:80]] for ln in lines],
            "chunks": [{"id": c["id"], "kind": c["kind"], "bbox": rnd(c["bbox"]), "ev": c["id"] in evid,
                        "text": c["body"] if c["id"] in evid else c["body"][:80]}
                       for c in engine.chunks if c["doc"] == doc],
            "sources": [rnd(b) for b in sources],
            "tiles": [[i, *t, None if ink is None else int(ink[i] < MIN_INK),
                       sum(t[0] <= ln["bbox"][0] < t[2] and t[1] <= ln["bbox"][1] < t[3] for ln in lines)]
                      for i, t in enumerate(grid)]}


def _sheet_blocks(docs: list[str], overlays: dict, layers: list[str]) -> list[dict]:
    """One overlay block per sheet of `docs` that has an overlay; `layers` are the layers the stage shows."""
    return [{"t": "overlay", "label": d, "doc": d, "layers": layers,
             "caption": f"{d} · {overlays[d]['size'][0]} × {overlays[d]['size'][1]} px"
                        + ("" if overlays[d]["src"] else " · page image not on this machine (pass --pdf): boxes only")}
            for d in docs if d in overlays]


def _prompt_segments(engine, system: str, user: str, ev: list[dict], fx: dict | None) -> list[list]:
    """Characters of each part of one prompt: system message, evidence (one per chunk in chunks mode), fact rows, and the rest
    of the user message. The parts add up to len(system) + len(user)."""
    if config.CONTEXT_MODE == "chunks":
        evidence = [[c["id"], len(engine.evidence_text([c], "chunks")), "evidence"] for c in ev]
    else:
        evidence = [["evidence (sheet mode)", len(engine.evidence_text(ev, config.CONTEXT_MODE)), "evidence"]]
    facts = [["fact rows", len(fx["text"]), "facts"]] if fx else []
    segs = [["system message", len(system), "system"]] + evidence + facts
    return segs + [["outline, question, separators", len(system) + len(user) - sum(s[1] for s in segs), "other"]]


def _answer(label: str, rec: dict | None) -> dict:
    if not rec:
        return {"label": label, "answer": None, "grade": None, "verdict": None, "evidence": [], "facts": [], "computed": []}
    grade = rec.get("grade")
    return {"label": label, "answer": rec.get("answer", ""), "grade": grade, "verdict": VERDICT.get(grade),
            "evidence": list(rec.get("evidence_ids") or []),
            "facts": [f["id"] if isinstance(f, dict) else str(f) for f in rec.get("facts") or []],
            "computed": list(rec.get("computed") or [])}


# ------------------------------------------------------------------ detail helpers
@contextmanager
def _timed(ms: dict, sid: str):
    """Wall-clock milliseconds of the block, stored as ms[sid]."""
    t0 = time.perf_counter()
    yield
    ms[sid] = round((time.perf_counter() - t0) * 1000, 1)


INDEX_STAGES = ["prepare", "pp_ocr", "vlm_reread", "seam_merge", "chunking", "embedding"]   # tiling runs inside pp_ocr
QUESTION_STAGES = ["expand", "retrieve", "rrf", "rerank", "anchor", "facts", "prompt"]


def timing_summary(indexing: dict | None, docs: list[str], ms: dict, seconds: float | None) -> dict:
    """The two phases of the stage timeline. Indexing: each stage from timing.json summed over the question's documents
    (the embedding covers the whole file set). Question: the measured retrieval stages, and the LLM stage derived as the
    recorded run's seconds minus them."""
    indexing = indexing or {}
    index = []
    for name in INDEX_STAGES:
        if name == "embedding":
            parts = {"whole file set": indexing.get(timing.INDEX, {}).get(name)}
        else:
            parts = {d: indexing.get(d, {}).get(name) for d in docs}
        parts = {k: v for k, v in parts.items() if v is not None}
        if parts:
            index.append({"name": name, "seconds": round(sum(parts.values()), 3), "per_doc": parts})
    question = [{"name": k, "seconds": round(ms[k] / 1000, 4)} for k in QUESTION_STAGES if k in ms]
    measured = round(sum(s["seconds"] for s in question), 4)
    llm = round(max(0.0, seconds - measured), 3) if seconds is not None else None
    meta = indexing.get(timing.META, {})
    return {"index_recorded": bool(index), "device": meta.get("device"), "engine": meta.get("engine"), "recorded": meta.get("recorded"),
            "docs": docs, "missing_docs": [d for d in docs if d not in indexing],
            "index": index, "index_total": round(sum(s["seconds"] for s in index), 3),
            "question": question, "llm": llm, "result_seconds": seconds,
            "question_total": round(measured + (llm or 0), 3) if llm is not None else None}


def novelty(chunks: list[dict], ranked: list[int], top: int, min_new: float = 0.25) -> dict[int, tuple]:
    """Engine.novel's walk, kept per chunk for display: {chunk: (share of bigrams not yet seen or None, status)}."""
    def grams(t: str) -> set[str]:
        s = fold(norm(t)).replace(" ", "").replace("\n", "")
        return {s[k:k + 2] for k in range(len(s) - 1)}

    out: dict[int, tuple] = {}
    kept, seen = 0, set()
    for i in ranked:
        if kept == top:
            out[i] = (None, "not reached: top-k full")
            continue
        g = grams(chunks[i]["body"])
        share = len(g - seen) / len(g) if g else 0.0
        if g and (kept == 0 or share >= min_new):
            kept += 1
            seen |= g
            out[i] = (share, "kept")
        else:
            out[i] = (share, "near copy" if g else "no bigrams")
    return out


def _nov_text(st: tuple) -> str:
    share, status = st
    return status if share is None else f"{status} · {share:.0%} new bigrams"


def _lines_file(ocr_dir: Path, doc: str) -> list[dict]:
    path = ocr_dir / f"{doc}.lines.json"
    return json.loads(path.read_text(encoding="utf-8"))["lines"] if path.exists() else []


def _texts(chunk: dict | None) -> list[str]:
    return [str(ln.get("text", "")) if isinstance(ln, dict) else str(ln) for ln in (chunk or {}).get("lines") or []]


PARSER = {"calc": "facts_calc.py", "legend": "facts_tables.py", "pump_table": "facts_tables.py",
          "plan_label": "facts_labels.py", "plan_volume": "facts_labels.py", "check": "facts_labels.py"}


def index_stages(engine, docs: list[str], regions: dict, ocr_dir: Path, page_files: dict, overlays: dict, by_id: dict) -> tuple[list[dict], list[dict]]:
    """The eight indexing stages for the sheet this question reads: its stored OCR lines, its page image (when it is on this
    machine), the index and the fact rows. Returns (stages, overview nodes)."""
    doc = docs[0] if docs else next(iter(engine.docs))
    W, H = engine.docs[doc]["size"]
    box = regions.get(doc) or [0, 0, W, H]
    page = page_files.get(doc)
    lines = _lines_file(ocr_dir, doc)
    pp = [ln for ln in lines if ln.get("src") != "vlm"]
    unsure = [ln for ln in pp if ln.get("score", 1.0) < UNSURE]
    replaced = [ln for ln in lines if "was" in ln]
    added = [ln for ln in lines if ln.get("src") == "vlm" and "was" not in ln]
    grid = list(tiles(W, H, PASS_TILE, PASS_OVERLAP))
    reread_grid = list(tiles(W, H, REREAD_TILE, REREAD_OVERLAP))
    reread_tiles = [t for t in reread_grid if sum(_centre_in(ln["bbox"], list(t)) for ln in unsure) >= MIN_UNSURE]
    flags = [t[5] for t in overlays[doc]["tiles"]] if doc in overlays else []
    kept = sum(f == 0 for f in flags) if flags and None not in flags else None
    tile_blocks = _sheet_blocks([doc], overlays, ["tiles"])
    ex_rep = [ln for ln in replaced if _centre_in(ln["bbox"], box)][:2] or replaced[:2]
    ex_pp = next((ln for ln in pp if _centre_in(ln["bbox"], box)), None)
    ex_block = next((c for c in engine.chunks if c["doc"] == doc and c["kind"] == "block" and len(c.get("lines") or []) >= 2
                     and _overlaps(c["bbox"], box)), None)
    ex_nb = next((c for c in engine.chunks if c["doc"] == doc and c["kind"] == "neighbourhood" and _overlaps(c["bbox"], box)), None)
    kinds = Counter(c["kind"] for c in engine.chunks if c["doc"] == doc)
    ex_fact = next((f for f in engine.facts if f["doc"] == doc and any(_overlaps(s["bbox"], box) for s in f["src"])), None)
    src_chunk = by_id.get(ex_fact["src"][0]["chunk"]) if ex_fact else None
    by_src = Counter(f["source"] for f in engine.facts if f["doc"] == doc)
    emb = getattr(engine, "emb", None)
    score_txt = lambda ln: f"score {ln['score']:.3f}" if "score" in ln else "no score"
    tiles_txt = f"{kept} of {len(grid)} tiles have ink (grey < 170 share ≥ {MIN_INK})" if kept is not None else f"{len(grid)} tiles (page image not on this machine: ink filter not applied)"

    stages = [
        _stage("render", "PDF render", "PDF page render",
               "The drawing is a PDF. Ingest turns each page into a bitmap at 300 dpi; OCR reads pixels only, so this bitmap is the whole input of the indexing pipeline.",
               [_rows("Sheet", [_row(doc, f"{W} × {H} px as indexed")])],
               [_rows("Page", [_row("size", f"{W} × {H} px = {W * H:,} pixels"),
                               _row("file", page.name if page else "not on this machine; pass --pdf to render it")])],
               phase="index",
               how=["pdftoppm -r 300 -png -singlefile renders the page into the pages folder. A native scan is used instead only when the page holds one embedded image of at least 150 ppi in jpeg, image, ccitt or jpx format.",
                    "The index keeps the image path and the page size; the OCR file is named after the sheet."],
               params=[("dpi", "300 (pdftoppm -r 300)"), ("page size", f"{W} × {H} px")],
               why="Every later step reads this bitmap, so its size sets how many tiles the OCR has to run on."),
        _stage("tiling", "Tiling", "Overlapping tiles",
               "The page is cut into square tiles that overlap, so a line cut by one tile edge is whole in the next tile. Blank tiles are skipped before OCR runs.",
               [_rows("Page", [_row(doc, f"{W} × {H} px")])],
               tile_blocks + [_rows("PP-OCR pass", [_row(f"{PASS_TILE} px", tiles_txt, f"overlap {PASS_OVERLAP} px · step {PASS_TILE - PASS_OVERLAP} px")]),
                _rows("VLM re-read grid", [_row(f"{REREAD_TILE} px", f"{len(reread_grid)} tiles", f"overlap {REREAD_OVERLAP} px · step {REREAD_TILE - REREAD_OVERLAP} px")])],
               phase="index",
               how=["ingest tiles() yields (x0, y0, x1, y1) boxes on a grid with step = tile − overlap; tiles at the right and bottom edges are clipped to the page.",
                    "A tile is kept only if its ink share (pixels darker than grey 170) is at least MIN_INK; a tile below it is skipped as blank and not read. The squares in the image are translucent, so where two tiles overlap the fill is darker."],
               params=[("PP-OCR tile / overlap", f"{PASS_TILE} / {PASS_OVERLAP} px"), ("re-read tile / overlap", f"{REREAD_TILE} / {REREAD_OVERLAP} px"),
                       ("MIN_INK", str(MIN_INK))],
               why="The overlap is why a line can be read twice; the merge stage removes those copies. The blank-tile skip saves OCR time on empty paper."),
        _stage("ocr_pass", "PP-OCR", "PP-OCR detect + recognise",
               "PP-OCR finds text boxes (detector) and reads each box (recogniser) on every kept tile. Every recognised line keeps the recogniser's confidence.",
               [_rows("Tiles", [_row("PP-OCR pass", f"{kept if kept is not None else len(grid)} tiles")])],
               tile_blocks + [_rows("Stored lines of this sheet", [_row(f"y{ex_pp['bbox'][1]}", ex_pp["text"], f"{score_txt(ex_pp)} · bbox {ex_pp['bbox']}")] if ex_pp else [_row("-", "no line in this region")]),
                _rows("Totals", [_row("PP-OCR lines", str(len(pp)), "non-VLM lines in the stored file"), _row("all lines", str(len(lines)))])],
               phase="index",
               how=["RapidOCR runs the PP-OCRv6 medium detector and recogniser (ONNX).",
                    "Detector: unclip ratio 2.0 (boxes grown to take the whole glyph); the longest side is limited to 2600 px.",
                    "Each line is stored with its bbox, text and score (0..1)."],
               params=[("engine", "ppocrv6-m-u20 (RapidOCR ONNX, PP-OCRv6 medium)"), ("tile / overlap", f"{PASS_TILE} / {PASS_OVERLAP} px"),
                       ("unclip", "2.0"), ("det limit", "2600 px")],
               why="The score is what the next stage judges. A low-score line is the only thing that sends a tile to the second engine."),
        _stage("reread", "VLM re-read", "Unsure tiles re-read by the second engine",
               f"A PP-OCR line scoring below {UNSURE} is unsure. A {REREAD_TILE}-px tile holding at least {MIN_UNSURE} unsure line centres is read again by HunyuanOCR (a vision-language model). Its lines are then arbitrated against the PP-OCR lines under them.",
               [_rows("Judging rule", [_row("unsure line", f"score < {UNSURE}"), _row("tile", f"≥ {MIN_UNSURE} unsure line centres inside it")])],
               [_rows("Counts (stored lines)", [_row("unsure lines", str(len(unsure)), f"of {len(pp)} PP-OCR lines"),
                                               _row("tiles meeting the rule", str(len(reread_tiles)), f"of {len(reread_grid)} grid tiles · estimate: the run does not record re-reads"),
                                               _row("replaced", str(len(replaced)), "PP-OCR text kept in 'was'"),
                                               _row("added", str(len(added)), "VLM line with no PP-OCR line under it")]),
                _rows("Before (PP-OCR) → after (VLM)", [r for ln in ex_rep for r in (_row("before", ln["was"], f"PP-OCR · bbox {ln['bbox']}"),
                                                                                 _row("after", ln["text"], "VLM replacement"))]
                      or [_row("-", "no replaced line on this sheet")])],
               phase="index",
               how=["The VLM reads the tile upscaled 2× and returns boxes and text, mapped back to page pixels.",
                    "Arbitration per VLM line: the PP-OCR lines under it are those with centre inside ≥ 0.5 or IoU ≥ 0.3. None → the VLM line is added. All of them below 0.75 → they are replaced by the VLM text, and 'was' keeps their PP-OCR text. Any one at or above 0.75 → the PP-OCR lines are kept and the VLM line is dropped.",
                    "A repetition loop in VLM output (≥ 12 characters with at most max(2, len/8) distinct ones) is dropped before arbitration.",
                    "The re-read grid is not drawn: the stored lines carry no tile id, so the file cannot show which tiles the second engine actually read. The count above applies the trigger rule to the stored scores."],
               params=[("trigger", f"score < {UNSURE}, ≥ {MIN_UNSURE} per {REREAD_TILE}-px tile"), ("VLM tile / overlap", f"{REREAD_TILE} / {REREAD_OVERLAP} px"),
                       ("upscale", "2.0×"), ("max new tokens", "3000"), ("repetition penalty", "1.08")],
               why="Most of the page is read by the fast engine; the slow one only runs where the fast one is unsure, which is what keeps indexing affordable."),
        _stage("merge", "Seam merge", "Seam duplicates and line merge",
               "Tiles overlap, so a line on a seam is read twice. Duplicates are dropped (the longer reading wins); the surviving lines are then joined into blocks when they sit close together.",
               [_rows("Lines of one block (before)", [_row(f"line {n}", t) for n, t in enumerate(_texts(ex_block))]
                      or [_row("-", "no block with two lines in this region")])],
               [_rows("Block (after)", [_row(ex_block["id"], ex_block["body"], f"{len(_texts(ex_block))} lines joined")] if ex_block else [_row("-", "none")])],
               phase="index",
               how=["merge_overlap() sorts lines longest first; a line is a duplicate of a kept one if its box IoU > 0.5, or it is ≥ 60 % inside a kept box and its text is part of the kept text (partial ratio ≥ 85).",
                    "group_blocks() joins two lines when they sit in one printed row (centres within 0.5 of the taller height, gap < 1.2 × height) or stack (vertical gap < 0.7 × height, overlapping in x), the gap is ≤ 1.0 text height, and no third line sits between them. A block may not grow past 45 × 16 text heights.",
                    "Lines are then sorted top to bottom, then left to right."],
               params=[("duplicate", "IoU > 0.5, or ≥ 60 % inside with text inside"), ("block gap", "≤ 1.0 text height"), ("block cap", "45 × 16 text heights")],
               why="Without this a seam would give two copies of one line, and a label split over lines would reach the index as separate pieces."),
        _stage("chunking", "Layout chunks", "Layout chunks",
               "Lines are grouped into three chunk kinds, so a question can hit one label, a whole section, or a neighbourhood of labels. The retrieval ranks chunks, not raw lines.",
               [_text("Sheet", f"{doc}: {len(lines)} stored lines")],
               [_rows("Chunks of this sheet", [_row(k, str(v), "kind") for k, v in sorted(kinds.items())] + [_row("index total", str(len(engine.chunks)))]),
                _rows("Neighbourhood example", [_row(ex_nb["id"], ex_nb["body"][:200], f"neighbourhood · {len(_texts(ex_nb))} lines")] if ex_nb else [_row("-", "none in this region")])],
               phase="index",
               how=["block: one label cluster from the block grouping.",
                    "section: every block under one level-2 heading (e.g. 三、處理單元功能計算 > 1. 初沉池), in reading order, so a unit's design values stay together.",
                    "neighbourhood: for each block, the blocks whose centres lie within R = 14 text heights (at most 14, at least 2); prose blocks do not start one; identical member sets are kept once.",
                    "reading_order() sorts by vertical centre; blocks whose centres are within 0.6 text heights and overlap vertically form one row, read left to right."],
               params=[("neighbourhood radius", "14 text heights"), ("neighbourhood size", "2–14 blocks"), ("row tolerance", "0.6 text heights"), ("kinds", "block · section · neighbourhood")],
               why="Plan labels only mean something next to the labels around them (neighbourhood); a unit's values are read together (section)."),
        _stage("index_build", "Embed + BM25", "Embedding and BM25 index",
               "Every chunk is embedded once (a dense vector) and tokenised once (for BM25). At answer time the question is searched both ways.",
               [_text("Chunks", f"{len(engine.chunks)} in the index")],
               [_rows("Dense index", [_row("shape", f"{emb.shape[0]} × {emb.shape[1]}" if emb is not None else "not loaded here", "chunk × embedding dimension")]),
                _rows("BM25 tokens of one chunk", [_row(ex_nb["id"] if ex_nb else "-", " · ".join(tokens(ex_nb["text"])[:16]) if ex_nb else "none",
                                                        f"{len(tokens(ex_nb['text']))} tokens" if ex_nb else "")])],
               phase="index",
               how=[f"Embedder {config.EMBED_ID} encodes the chunk text, normalised, in batches of 64 (inner batch 16). Stored as float16; search loads float32 and takes a dot product (cosine on unit vectors).",
                    "BM25: rank_bm25 BM25Okapi over index.tokens(): CJK characters as uni- and bi-grams, ASCII words and numbers whole (2HP, 0.25m3/min)."],
               params=[("embedder", config.EMBED_ID), ("batch", "64 texts, inner 16"), ("stored dtype", "float16"), ("BM25", "BM25Okapi, library defaults")],
               why="Dense search finds paraphrases; BM25 finds exact unit numbers and codes. RRF merges the two at answer time."),
        _stage("fact_parse", "Fact parsers", "Fact rows parsed by code",
               "Before any question, the OCR text is parsed into printed values with unit and source chunk: calculation sheets, the legend and pump table, plan labels and volumes. The answer reads these rows as evidence.",
               [_rows("Source chunk lines (before parsing)", [_row(f"line {n}", t) for n, t in enumerate(_texts(src_chunk))]
                      or [_row("-", "no fact row from this region")])],
               [_rows("Rows of this sheet by source", [_row(s, str(n), PARSER.get(s, "")) for s, n in by_src.most_common()] or [_row("-", "none")]),
                _rows("One row (after parsing)", [_row(ex_fact["id"], f"{ex_fact['label']} = {ex_fact['value']} {ex_fact['unit']}",
                                                       f"{ex_fact['source']} · {ex_fact['text']}")] if ex_fact else [_row("-", "none")])],
               phase="index",
               how=["facts.dedupe() drops a line ≥ 60 % inside a longer line; facts.stitch() joins two lines in one row when the end of one and the start of the next overlap by at least 3 characters with fuzz.ratio ≥ 85 (row tolerance 0.5 text height).",
                    "facts_calc.py reads calculation sheets; facts_tables.py reads the legend and the pump table; facts_labels.py reads plan labels, volumes and their checks.",
                    "Rows are added to the prompt unless DOCSEARCH_FACTS=0 (on by default)."],
               params=[("dedupe", "≥ 60 % inside a longer line"), ("stitch", "fuzzy ratio ≥ 85, row tolerance 0.5 text height"),
                       ("parsers", "facts_calc · facts_tables · facts_labels"), ("rows in index", str(len(engine.facts)))],
               why="A printed number split over two OCR lines only becomes a value once the fragments are stitched; each row keeps its source chunk so the answer can cite it."),
    ]
    nodes = [
        {"label": "page pixels", "value": f"{W * H:,}", "stage": "render"},
        {"label": "PP-OCR tiles", "value": kept if kept is not None else len(grid), "stage": "tiling"},
        {"label": "OCR lines", "value": len(lines), "stage": "ocr_pass"},
        {"label": "VLM re-read tiles (estimate)", "value": len(reread_tiles), "stage": "reread"},
        {"label": "chunks", "value": len(engine.chunks), "stage": "chunking"},
        {"label": "embedded", "value": len(engine.chunks), "stage": "index_build"},
        {"label": "fact rows", "value": len(engine.facts), "stage": "fact_parse"},
    ]
    return stages, nodes


# ------------------------------------------------------------------ collect
def collect(engine, question: str, *, top: int = config.TOP_K, question_id: str = "", category: str = "", difficulty: str = "",
            reference: str | None = None, recorded: dict | None = None, pdfs: list[Path] = (), pages_dir: Path | None = None,
            ocr_dir: Path | None = None, indexing: dict | None = None) -> dict:
    """Run the production retrieval and prompt code on `question`, keeping each stage's input, output and wall-clock time.
    `recorded`: {"base": result of the run without fact rows, "facts": result of the run with them} (either may be None)."""
    recorded = recorded or {}
    chunks = engine.chunks
    by_id = {c["id"]: c for c in chunks}
    ocr_dir = ocr_dir or config.OCR_DIR
    pages_dir = pages_dir or config.WORK / "trace-pages"
    ms: dict[str, float] = {}

    # question
    ents = question_entities(question)
    wanted = {e for e in ents if e[1] is not None}
    query = factquery.parse(question)

    # query expansion, BM25 and dense per query, RRF -> candidate pool (the calls Engine.hybrid makes)
    with _timed(ms, "expand"):
        queries = engine.expand(question)
    with _timed(ms, "retrieve"):
        lexical = [engine.lexical(q, LEX_K) for q in queries]
        dense = engine.dense(queries, LEX_K)
    rankings = lexical + dense
    with _timed(ms, "rrf"):
        fused = fuse(rankings)
    cand = [i for i, _ in fused][:LEX_K]

    # cross-encoder rerank, novelty filter, entity anchoring (the branches of Engine.answer; the rerank is the same call)
    with _timed(ms, "rerank"):
        scores = np.asarray(engine.reranker().predict([(question, chunks[i]["text"]) for i in cand], batch_size=8))
        score_of = {cand[j]: float(scores[j]) for j in range(len(cand))}
        order = [cand[j] for j in np.argsort(-scores)]
        novel = engine.novel(order, top)
    nov = novelty(chunks, order, top)
    with _timed(ms, "anchor"):
        best = anchor(question, novel, order, chunks, top) if wanted else novel
    ev = [chunks[i] for i in best]

    # fact rows and both prompts (A: flag off, B: fact rows added)
    with _timed(ms, "facts"):
        fx = factquery.context(question, engine.facts)
    fact_rows = fx["rows"] if fx else []
    with _timed(ms, "prompt"):
        system_a, user_a = engine.build_prompt(question, ev, None)
        system_b, user_b = engine.build_prompt(question, ev, fx)

    # source rows: the rows the fact rows were parsed from, else the top evidence chunks
    boxes = [(by_id[s["chunk"]]["doc"], tuple(s["bbox"])) for r in fact_rows for s in r["src"] if s["chunk"] in by_id]
    if not boxes:
        boxes = [(c["doc"], tuple(c["bbox"])) for c in ev[:3]]
    boxes = list(dict.fromkeys(boxes))[:8]
    docs = list(dict.fromkeys(d for d, _ in boxes))

    regions = {doc: _source_area(engine.docs[doc]["size"], [list(b) for d, b in boxes if d == doc]) for doc in docs}
    page_files = {doc: _page_path(engine, doc, list(pdfs), pages_dir) for doc in docs}
    evid = {c["id"] for c in ev}
    overlays = {d: _overlay(engine, d, [b for dd, b in boxes if dd == d], page_files[d], ocr_dir, evid) for d in docs}
    rec_b = recorded.get("facts") or {}
    rec_any = rec_b or recorded.get("base") or {}
    compare = {"a": {"label": "A · flag off, no fact rows", "chars": len(system_a) + len(user_a)},
               "b": {"label": "B · fact rows added", "chars": len(system_b) + len(user_b)},
               "diff": {"system": _diff(system_a, system_b), "user": _diff(user_a, user_b)}}
    stack = {"t": "stack", "label": "Prompt composition (characters)", "rows": [
        {"label": "A", "total": compare["a"]["chars"], "segs": _prompt_segments(engine, system_a, user_a, ev, None)},
        {"label": "B", "total": compare["b"]["chars"], "segs": _prompt_segments(engine, system_b, user_b, ev, fx)}]}
    stages: list[dict] = []
    index_st, index_nodes = index_stages(engine, docs, regions, ocr_dir, page_files, overlays, by_id)
    stages += index_st

    # 1 question
    stages.append(_stage("question", "Question", "Question",
                         "The question is read before any retrieval: the drawing units it names (entities) and the kind of fact it asks for.",
                         [_text("Question", question)],
                         [_rows("Entities", [_row(fam, ident or "no unit number", "anchoring key" if ident else "") for fam, ident in ents]),
                          _text("Fact intent", _query_text(query))],
                         how=["question_entities() finds drawing unit names and their numbers (e.g. 調整池 1).", "factquery.parse() reads the kind of value the question asks for."],
                         params=[("anchoring key", "unit family + number"), ("fact intent", "factquery.parse")],
                         why="The unit number is what anchoring uses to pull that unit's chunks into the evidence; without one, anchoring is skipped."))

    # 2 input page
    stages.append(_stage("page", "Input page", "Input page",
                         "The sheet the answer lives on, shown whole. Red boxes mark the source rows.",
                         [_rows("Source rows", [_row(d, f"bbox {list(b)}") for d, b in boxes])],
                         _sheet_blocks(docs, overlays, ["src"]) or [_text("Page image", "no source rows")],
                         how=[f"The whole page is scaled to {SHEET_WIDTH} px wide and embedded as WebP, at the highest quality that keeps the sheet under {SHEET_BUDGET // 1024} KiB. Boxes are in page pixels over the image (SVG viewBox = page size), so they line up at any zoom.",
                              "Only the sheets that hold source rows are embedded. The zoom button switches between fit width and 2× with scrolling."],
                         params=[("image", f"WebP, {SHEET_WIDTH} px wide"), ("sheet budget", f"{SHEET_BUDGET // 1024} KiB per sheet"),
                                 ("sheets shown", "those holding source rows"), ("source boxes", "up to 8")],
                         why="It shows where on the sheet the answer comes from; the OCR lines and chunks below are the text of the source area."))

    # 3 OCR lines (every line of the sheet is on the image; the table lists the source area)
    ocr_rows, ocr_note = [], []
    for doc in docs:
        path = ocr_dir / f"{doc}.lines.json"
        if not path.exists():
            ocr_note.append(f"no OCR file for {doc}")
            continue
        lines = [ln for ln in json.loads(path.read_text(encoding="utf-8"))["lines"] if _centre_in(ln["bbox"], regions[doc])]
        lines.sort(key=lambda ln: (ln["bbox"][1] // 30, ln["bbox"][0]))
        ocr_rows += [_row(f"y{ln['bbox'][1]}", ln["text"], f"{doc[:40]} · bbox {ln['bbox']}") for ln in lines[:40]]
    stages.append(_stage("ocr", "OCR lines", "OCR lines",
                         "OCR (ingest) turned the page into text lines with boxes. The image shows every line of the sheet; the table lists the lines whose centre is inside the source area.",
                         [_text("Source area", "; ".join(f"{d}: {regions[d]}" for d in docs) or "no source rows")],
                         _sheet_blocks(docs, overlays, ["src", "lines"]) + [_rows("Lines in the source area", ocr_rows)] + ([_text("Note", "; ".join(ocr_note))] if ocr_note else []),
                         how=[f"Stored OCR lines of the sheet are drawn grey. The table holds the lines whose centre is inside the source area (the source rows padded by {SOURCE_PAD} px), sorted in 30-px row bands, then left to right."],
                         params=[("row band", "30 px"), ("source pad", f"{SOURCE_PAD} px"), ("lines in table", "40 per sheet")],
                         why="These are the raw lines the chunks are built from; a wrong reading here cannot be fixed by retrieval."))

    # 4 chunks (the units the retrieval ranks)
    inregion = [c for c in chunks if c["doc"] in regions and _overlaps(c["bbox"], regions[c["doc"]])][:12]
    stages.append(_stage("chunks", "Chunks", "Chunks",
                         "Lines are grouped into block chunks (one label cluster), section chunks (a headed group) and neighbourhood chunks (a window of blocks). BM25 and the embedder index their text. The image shows every chunk of the sheet; evidence chunks are drawn with their full text.",
                         [_text("Index", f"{len(chunks)} chunks over {len(engine.docs)} document(s)")],
                         _sheet_blocks(docs, overlays, ["lines", "chunks", "ev", "src"])
                         + [_rows("Chunks overlapping the source area", [_row(c["id"], c["body"], f"{c['kind']} · {c.get('section') or '-'}"
                                                                         + (" · in evidence" if c["id"] in evid else ""))
                                                                        for c in inregion])],
                         how=["The image shows every chunk of the sheet (text cut to 80 characters, evidence chunks in full). The table lists the chunks overlapping the source area, up to 12, with their kind and section heading."],
                         params=[("kinds", "block · section · neighbourhood"), ("table", "12 chunks"), ("image text", "80 characters")],
                         why="Retrieval ranks chunks, so the answer can only use text that sits inside one of them."))

    # 5 expansion
    stages.append(_stage("expand", "Query expansion", "Query expansion",
                         "Engineers and users name the same things differently. Every glossary group the question touches adds one lexical query.",
                         [_text("Question", question)],
                         [_rows("Search queries", [_row(f"query {n}", q, "question" if n == 0 else "glossary group") for n, q in enumerate(queries)])],
                         how=["Engine.expand() adds one query per glossary group the question touches; the question itself is query 0."],
                         params=[("queries", str(len(queries)))],
                         why="The same thing is written differently on drawings and in questions; each name is searched."))

    # 6 BM25 + dense (top SHOW_RANK per ranking; the bar is the rank, 1 = top)
    rank_rows = []
    for n, r in enumerate(lexical):
        rank_rows += [_row(f"BM25 · q{n} · #{k + 1}", f"{chunks[i]['id']} · {chunks[i]['body'][:60]}", f"top {LEX_K} of BM25",
                           bar=1 - k / LEX_K, tone="rank") for k, i in enumerate(r[:SHOW_RANK])]
    for n, r in enumerate(dense):
        rank_rows += [_row(f"dense · q{n} · #{k + 1}", f"{chunks[i]['id']} · {chunks[i]['body'][:60]}", f"top {LEX_K} by dot product",
                           bar=1 - k / LEX_K, tone="rank") for k, i in enumerate(r[:SHOW_RANK])]
    stages.append(_stage("retrieve", "BM25 + dense", "BM25 + dense",
                        "Each query runs BM25 over the tokenised chunk text and a BGE-M3 dense search over the chunk embeddings.",
                         [_rows("Queries", [_row(f"query {n}", q[:80]) for n, q in enumerate(queries)])],
                         [_rows("Rankings (top 5 each)", rank_rows)],
                         how=[f"BM25 (rank_bm25 over index.tokens) takes the top {LEX_K} per query; dense search takes the top {LEX_K} by dot product with the query embedding.",
                              "The bar is 1 − (rank − 1) / 60: the top chunk is full, the fifth is 1 − 4/60 ≈ 93 %."],
                         params=[("per ranking", f"top {LEX_K}"), ("rankings", str(len(rankings)))],
                         why="BM25 wins on exact codes and numbers, dense search on paraphrase; both lists go into the fusion."))

    # rank of every candidate in each list (1 = top): the slope chart of the rrf, rerank and anchor stages
    def best_rank(lists: list[list[int]], i: int) -> int | None:
        found = [r.index(i) + 1 for r in lists if i in r]
        return min(found) if found else None
    fused_pos = {i: k + 1 for k, (i, _) in enumerate(fused)}
    order_pos = {i: k + 1 for k, i in enumerate(order)}
    final_pos = {i: k + 1 for k, i in enumerate(best)}
    src_ids = {s["chunk"] for r in fact_rows for s in r["src"]}

    def status(i: int) -> str:
        if i in final_pos:
            return "kept" if i in novel else "anchored"
        return "removed" if i in novel else "dropped"
    slope_rows = [{"id": chunks[i]["id"], "text": chunks[i]["body"][:80], "status": status(i), "src": chunks[i]["id"] in src_ids,
                   "note": _nov_text(nov[i]) if i in nov else "not in the candidate pool",
                   "ranks": [best_rank(lexical, i), best_rank(dense, i), fused_pos.get(i), order_pos.get(i), final_pos.get(i)]}
                  for i in dict.fromkeys([i for i, _ in fused[:15]] + best)]
    slope = {"t": "slope", "label": "Rank movement (top candidates)", "cols": ["BM25", "dense", "RRF", "reranker", "final evidence"],
             "depth": LEX_K, "rows": slope_rows}
    # 7 RRF
    top_rrf = fused[0][1] if fused else 1.0
    stages.append(_stage("rrf", "RRF fusion", "RRF fusion",
                         f"Reciprocal-rank fusion: each chunk scores sum(1 / ({RRF_K} + rank)) over the {len(rankings)} rankings. The top {LEX_K} go on to the reranker.",
                         [_text("Input", f"{len(rankings)} rankings")],
                         [_rows("Fused ranking (top 20)", [_row(i, chunks[i]["body"], f"rrf {s:.4f} · in {sum(i in r for r in rankings)} of {len(rankings)} lists",
                                                                bar=s / top_rrf, tone="rrf")
                                                          for i, s in fused[:20]]), {**slope, "upto": 2}],
                         how=[f"Each chunk's score is the sum over the rankings of 1 / ({RRF_K} + rank), rank counted from 1.", f"The top {LEX_K} candidates go to the reranker."],
                         params=[("RRF constant", str(RRF_K)), ("candidates", str(LEX_K))],
                         why="Raw scores of different rankers cannot be added; RRF uses only positions. The bar is the score relative to the top score."))

    # 8 rerank
    lo, hi = (float(scores.min()), float(scores.max())) if len(scores) else (0.0, 1.0)
    span = (hi - lo) or 1.0
    strip = {"t": "strip", "label": "Reranker scores (all candidates)", "top": top, "lo": lo, "hi": hi,
             "items": [[chunks[i]["id"], score_of[i], nov[i][1] == "kept"] for i in order]}
    stages.append(_stage("rerank", "Rerank", "Cross-encoder rerank",
                         f"{config.RERANK_ID} scores each (question, chunk) pair. Then near-copies are dropped: a chunk is kept only if 25 % of its bigrams are new.",
                         [_text("Candidates", f"{len(cand)} chunks")],
                         [_rows("Reranked (top 20 by score)", [_row(i, chunks[i]["body"], f"score {score_of[i]:.3f} · " + _nov_text(nov[i]),
                                                                bar=(score_of[i] - lo) / span, tone="rerank")
                                                             for i in order[:20]]), strip, {**slope, "upto": 3}],
                         how=[f"{config.RERANK_ID} scores each (question, chunk) pair in batches of 8; candidates are sorted by that score.",
                              "The novelty filter walks the sorted list: a chunk is kept if at least 25 % of its character bigrams (after fold and norm) are new; the walk stops at top-k."],
                         params=[("reranker", config.RERANK_ID), ("batch", "8"), ("novelty", "≥ 25 % new bigrams"), ("top-k", str(top))],
                         why="Near-copy neighbourhood chunks would fill the top-k with one label repeated. The bar is the score relative to the candidates."))

    # 9 entity anchoring
    dropped = [i for i in novel if i not in best]
    matched = lambda i: sorted(f"{f}{n}" for f, n in wanted if (f, n) in chunk_entities(chunks[i]["text"]))
    anchor_rows = [_row(i, chunks[i]["body"], ("kept" if i in novel else "added by anchoring") + (f" · matches {matched(i)}" if matched(i) else ""))
                   for i in best]
    anchor_rows += [_row(i, chunks[i]["body"], "removed to make room") for i in dropped]
    anchor_out = [_rows(f"Evidence for the prompt ({len(best)})", anchor_rows)]
    rec_base = (recorded.get("base") or {}).get("evidence_ids")
    if rec_base is not None:
        same = set(rec_base) == evid
        anchor_out.append(_text("Recorded run", f"the run without fact rows cited {len(rec_base)} chunks; this trace selects {len(ev)}: "
                                + ("same set" if same else f"differs ({len(set(rec_base) - evid)} only in the recorded run)")))
    anchor_out.append({**slope, "upto": 4})
    stages.append(_stage("anchor", "Entity anchoring", "Entity anchoring",
                         "If the question names a unit (e.g. 調整池 1), chunks that carry that unit are pulled into the evidence, up to 3 of them.",
                         [_rows("Entities", [_row(f, n) for f, n in wanted] or [_row("-", "no unit number: anchoring skipped")])],
                         anchor_out,
                         how=["anchor() adds chunks that carry the named unit; the lowest novel chunks are removed to make room."],
                         params=[("anchored chunks", "up to 3")],
                         why="A question about 調整池 1 must read 調整池 1's chunks, not the nearest similar unit."))

    # 10 fact rows
    rec_facts = rec_b.get("facts")
    fact_out = [_rows(f"Selected rows ({len(fact_rows)})",
                      [_row(r["id"], f"{r.get('entity', '')} | {r.get('label', '')} | printed: {r.get('text', '')}",
                            f"{r.get('source', '')} · {', '.join(s['chunk'] for s in r['src'])}") for r in fact_rows])]
    if fx and fx["computed"]:
        fact_out.append(_rows("Computed by code", [_row("exact", line) for line in fx["computed"]]))
    if rec_facts is not None:
        same = [r["id"] for r in fact_rows] == list(rec_facts)
        fact_out.append(_text("Recorded run", f"fact row ids {list(rec_facts)}: " + ("same rows" if same else "differ from this trace")))
    stages.append(_stage("facts", "Fact rows", "Fact rows parsed by code",
                         "The OCR text was parsed into printed values (value, unit, source chunk) before any question. The question selects rows; computed lines are exact sums, counts and comparisons.",
                         [_text("Parsed", f"{len(engine.facts)} rows over {len({r['doc'] for r in engine.facts})} sheet(s)")],
                         fact_out if fact_rows else [_text("Selected rows", "no row matched: prompt B is prompt A")],
                         how=["factquery.context() selects the fact rows for the question and computes exact sums, counts and comparisons in code.",
                              "Rows reach the prompt unless DOCSEARCH_FACTS=0 (on by default)."],
                         params=[("FACTS flag", "on by default (config.FACTS); DOCSEARCH_FACTS=0 turns it off"), ("rows appended as evidence", "up to 8")],
                         why="The model is not asked to find or add the numbers; code selects and computes them."))

    # 11 prompts A vs B
    stages.append(_stage("prompt", "Prompts A vs B", "Messages sent to the LLM",
                         "The exact system and user messages. B is A plus the fact rows and the rule for using them; the diff is highlighted.",
                         [_rows("Evidence", [_row(f"{len(ev)} chunks", _ids(engine, best))]),
                          _rows("Fact rows", [_row(f"{len(fact_rows)} rows", ", ".join(r["id"] for r in fact_rows) or "none")])],
                         [stack], view="prompt",
                         how=["Engine.build_prompt() puts the evidence chunks into the messages as <chunk id kind> blocks; B adds the fact rows and the rule for using them."],
                         params=[("evidence", f"{config.TOP_K} chunks at most"), ("prompt A / B", f"{compare['a']['chars']} / {compare['b']['chars']} chars")],
                         why="A and B differ only in the fact rows, so any change in the answer comes from them."))

    # 12 LLM stage: what runs between the prompt and the answer
    seed_rec = rec_any.get("seed", "not recorded")
    example = re.search(r"\{\{[^{}]+\}\}", system_b)
    ex_expr = example.group(0) if example else None
    stages.append(_stage("llm", "LLM", "LLM stage",
                         f"{config.LLM_ID} reads the messages from the previous step and writes the answer. It runs on the backend set by DOCSEARCH_LLM_BACKEND (here: {config.LLM_BACKEND}). The thinking text is returned by chat() but is not saved in the recorded result files, so it is not shown.",
                         [_rows("Model", [_row("model", config.LLM_ID), _row("backend", config.LLM_BACKEND, "transformers (in-process, default) or openai (any OpenAI-compatible server)"), _row("messages", "system + user, from the previous step")])],
                         [_rows("Limits and sampling", [_row("thinking cap", f"{THINK_CAP} tokens", "max_new_tokens of Engine.chat"),
                                                        _row("answer cap", f"{ANSWER_CAP} tokens", "answer_tokens of Engine.chat"),
                                                        _row("sampling", "temperature 0.6 · top_p 0.95 · top_k 20", "transformers path: do_sample"),
                                                        _row("seed", str(config.SEED), f"config.SEED (DOCSEARCH_SEED) · recorded run: {seed_rec}")]),
                          _rows("Thinking", [_row("recorded", "not saved", "chat() returns it, but the result files hold only the final answer (no reasoning or seed field)")]),
                          _rows("Calculator step", [_row("trigger", "the model writes {{expression}} as the prompt asks; calc.fill runs on the answer"),
                                                    _row("example in prompt", ex_expr or "none found", "the instruction example in the messages"),
                                                    _row("evaluated", calc.fill(ex_expr) if ex_expr else "-", "exact arithmetic, rounded to 2 decimals")]),
                          _rows("After the answer", [_row("1", "calc.fill: {{…}} → 'expression = value'; a written 'a × b = c' chain is recomputed and corrected when off by more than rounding"),
                                                    _row("2", "citations: a '#' followed by a number not followed by another digit; search.locate() reads them. benchjson strips trailing '#n' before grading"),
                                                    _row("3", "with fact rows on: up to 24 rows are selected, and up to 8 computed-check lines are added")])],
                         how=["transformers path: the model thinks up to max_new_tokens (7000); if no </think> appears, the thought is closed (budget forcing) and the answer is written with answer_tokens (1500). The thinking is returned as text.",
                              "openai path: one streamed request with seed and max_tokens = 7000 + 1500; if the thinking runs out, one follow-up call writes the answer from the tail of that thinking.",
                              "Sampling: temperature 0.6, top_p 0.95, top_k 20 on the transformers path; the openai path sends the seed."],
                         params=[("model", config.LLM_ID), ("backend", config.LLM_BACKEND), ("thinking cap", f"{THINK_CAP}"), ("answer cap", f"{ANSWER_CAP}"),
                                 ("seed", f"{config.SEED} (DOCSEARCH_SEED)")],
                         why="The answer is what the grade reads, so the caps and sampling set how long the model may think and how repeatable a run is."))

    # 13 answers
    stages.append(_stage("answers", "Answers vs reference", "Answers vs reference",
                         "The answering LLM read prompt A and prompt B. No model runs here: these are the recorded final answers. The timeline under the answers has two phases on one time axis: the indexing of the file set, then the question, from its submission to its answer.",
                         [_text("Prompts", "A and B, see the previous step")], [], view="answers",
                         how=["The recorded answers and their grades are shown next to the reference; no grading runs here.",
                              "Indexing seconds are measured by the build (the CLI and the web upload share this code path) and read from timing.json; each stage is wall-clock time on the device in its _meta. A stage is summed over this question's documents; the embedding covers the whole file set. Tiling runs inside the PP-OCR pass, so it is one stage; the VLM re-read time depends on how many tiles it re-reads. Fact rows and BM25 are built when the index loads, so they are not in the indexing phase.",
                              "The question phase measures the retrieval stages here. The LLM stage is derived (not measured): the recorded run's seconds minus those stages. It includes answer post-processing, which is not timed on its own."],
                         params=[("grade", "pass · fail"), ("timing", "<work>/timing.json, written by the build (--timing)")],
                         why="It shows whether the fact rows changed the verdict on this question."))

    rec_answer = rec_b.get("answer")
    answer_nodes = [
        {"label": "rankings", "value": len(rankings), "stage": "retrieve"},
        {"label": "candidates", "value": len(cand), "stage": "rrf"},
        {"label": "after novelty", "value": len(novel), "stage": "rerank"},
        {"label": "evidence", "value": len(ev), "stage": "anchor"},
        {"label": "fact rows", "value": len(fact_rows), "stage": "facts"},
        {"label": "prompt chars", "value": compare["b"]["chars"], "stage": "prompt"},
        {"label": "LLM", "value": config.LLM_ID.split("/")[-1], "stage": "llm"},
        {"label": "answer chars", "value": len(rec_answer) if rec_answer is not None else "not recorded", "stage": "answers"},
    ]
    for st in stages:
        st["ms"] = ms.get(st["id"])
    return {
        "title": f"Pipeline trace · {question_id or 'question'}",
        "meta": {"question_id": question_id, "category": category, "difficulty": difficulty,
                 "llm": rec_any.get("llm") or config.LLM_ID, "top": top, "context": config.CONTEXT_MODE},
        "reference": reference or rec_any.get("reference") or "",
        "overview": {"index": {"label": "Done once per drawing (indexing)", "nodes": index_nodes},
                     "answer": {"label": "Per question (answering)", "nodes": answer_nodes}},
        "stages": stages,
        "compare": compare,
        "overlays": overlays,
        "answers": {"a": _answer("A · no fact rows", recorded.get("base")), "b": _answer("B · fact rows", recorded.get("facts"))},
        "timing": timing_summary(indexing, list(engine.docs), ms, rec_b.get("seconds")),
    }


# ------------------------------------------------------------------ render
def render(data: dict, out_path: Path) -> Path:
    """Write the player: the template with `data` embedded as JSON. `<`, `>` and `&` are escaped so no text can close the script."""
    html = TEMPLATE.read_text(encoding="utf-8")
    if html.count(PLACEHOLDER) != 1:
        raise ValueError(f"{TEMPLATE} must contain {PLACEHOLDER} exactly once")
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(PLACEHOLDER, payload), encoding="utf-8")
    return out_path


def question_item(doc: dict) -> dict:
    """The benchmark item from {"題目": [item]} (the q058.json shape) or a bare item object."""
    if isinstance(doc, dict) and isinstance(doc.get("題目"), list):
        return doc["題目"][0]
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m docsearch.trace", description="Step-by-step HTML trace of one question through the pipeline. Never calls the LLM.")
    ap.add_argument("--question-file", type=Path, required=True, help="benchmark JSON with the question (題目 list or one item)")
    ap.add_argument("--result-file", type=Path, required=True, help="recorded run WITH fact rows")
    ap.add_argument("--result-base-file", type=Path, help="recorded run WITHOUT fact rows")
    ap.add_argument("--pdf", type=Path, action="append", default=[], help="source PDF or folder, used when the index page image is not on this machine (repeatable)")
    ap.add_argument("--top", type=int, default=config.TOP_K)
    ap.add_argument("--out", type=Path, required=True, help="HTML file to write")
    ap.add_argument("--timing", type=Path, help="timing.json written by the build (default: <work>/timing.json when it exists)")
    a = ap.parse_args(argv)

    item = question_item(json.loads(a.question_file.read_text(encoding="utf-8")))
    facts_run = json.loads(a.result_file.read_text(encoding="utf-8"))
    base_run = json.loads(a.result_base_file.read_text(encoding="utf-8")) if a.result_base_file else None
    if not (config.INDEX_DIR / "chunks.json").exists():
        sys.exit(f"no index at {config.INDEX_DIR}: set DOCSEARCH_WORK to a directory with index/ and ocr/")
    from .search import Engine

    data = collect(Engine(), item.get("問題") or facts_run["question"], top=a.top,
                   question_id=str(item.get("編號", facts_run.get("id", ""))), category=item.get("類別", ""), difficulty=item.get("難度", ""),
                   reference=item.get("答案"), recorded={"facts": facts_run, "base": base_run}, pdfs=a.pdf,
                   pages_dir=a.out.parent / "trace-pages", indexing=timing.load(a.timing or config.WORK / timing.FILE))
    render(data, a.out)
    print(f"wrote {a.out} ({len(data['stages'])} stages)")


if __name__ == "__main__":
    main()
