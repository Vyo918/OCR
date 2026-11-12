"""Stage 2 - layout-aware contextual index.

OCR lines (text + page coordinates) are turned into chunks that keep their *context*:

  * block chunks     - lines grouped by spatial proximity (paragraphs, table rows, labels);
  * section path     - calculation-sheet headings (一、 / 1. / A. / a.) inherited by the prose below
                       them, so "V=84.93x0.70" is stored as "...> 4.終沉池 > B.規劃計算 > b.實際容量";
  * neighbourhood    - for scattered drawing labels, the labels around them (a tank name, its
                       height and the pipes written next to it only mean something together);
  * document header  - drawing title + position on the sheet are prefixed to every chunk
                       (Anthropic-style "contextual retrieval").

Retrieval is hybrid: char n-gram BM25 (exact keyword hits, units, model numbers) + BGE-M3 dense
vectors, fused with reciprocal-rank fusion, then re-ranked by a cross-encoder (see search.py).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from . import config


H1 = re.compile(r"^\s*[一二三四五六七八九十]+\s*[、,.]")
H2 = re.compile(r"^\s*\d{1,2}\s*[.．]\s*\S")
H3 = re.compile(r"^\s*[A-ZＡ-Ｚ]\s*[.．]\s*\S")
H4 = re.compile(r"^\s*[a-zａ-ｚ]\s*[.．]\s*\S")


_T2S = None
_VARIANTS = str.maketrans({"沈": "沉", "联": "连"})   # 沈澱/沉澱 and 聯通/連通 are both in use on Taiwanese drawings


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("³", "3").replace("²", "2").replace("×", "x").replace("＊", "*")
    return re.sub(r"\s+", " ", s).strip()


def fold(s: str) -> str:
    """Traditional -> Simplified, for matching only (never for display): OCR engines and users mix the two scripts freely,
    so 風機 / 风机 and 連通管 / 连通管 must hit each other."""
    global _T2S
    if _T2S is None:
        from opencc import OpenCC
        _T2S = OpenCC("t2s")
    return _T2S.convert(s).translate(_VARIANTS)


def tokens(s: str) -> list[str]:
    """Char uni+bi-grams for CJK, whole tokens for ASCII words/numbers (units like 2HP, 0.25m3/min)."""
    s = fold(norm(s)).lower()
    out: list[str] = []
    for m in re.finditer(r"[a-z0-9][a-z0-9.\-/\"']*|[\u3400-\u9fff]+", s):
        w = m.group()
        if "\u3400" <= w[0] <= "\u9fff":
            out += list(w) + [w[i:i + 2] for i in range(len(w) - 1)]
        else:
            out.append(w)
            parts = re.findall(r"\d+(?:\.\d+)?|[a-z]+", w)
            if len(parts) > 1:
                out += parts
    return out


def heading_level(text: str) -> int | None:
    t = norm(text)
    for lvl, rx in enumerate((H1, H2, H3, H4), 1):
        if rx.match(t):
            return lvl
    return None


class UF:
    """Union-find that also tracks each group's bounding box, so a merge can be refused when the group gets too big."""

    def __init__(self, boxes):
        self.p = list(range(len(boxes)))
        self.box = [list(b) for b in boxes]

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b, max_w, max_h) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        x0, y0 = min(self.box[ra][0], self.box[rb][0]), min(self.box[ra][1], self.box[rb][1])
        x1, y1 = max(self.box[ra][2], self.box[rb][2]), max(self.box[ra][3], self.box[rb][3])
        if x1 - x0 > max_w or y1 - y0 > max_h:
            return
        self.p[ra] = rb
        self.box[rb] = [x0, y0, x1, y1]


def group_blocks(lines: list[dict]) -> list[list[int]]:
    """Union lines that read as one unit: pieces of one printed row, or stacked rows of one paragraph / one label.
    Distances are in text heights, and a block may not grow past 45 x 16 text heights, so a sparse drawing becomes many
    small label blocks instead of a few sheet-wide chains."""
    n = len(lines)
    hs = np.array([max(l["bbox"][3] - l["bbox"][1], 1) for l in lines], float)
    h = float(np.median(hs))
    uf = UF([l["bbox"] for l in lines])
    order = sorted(range(n), key=lambda i: lines[i]["bbox"][1])
    for ai, i in enumerate(order):
        a = lines[i]["bbox"]
        for j in order[ai + 1:]:
            b = lines[j]["bbox"]
            if b[1] - a[3] > 1.0 * h:
                break
            ay, by = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
            hmax = max(a[3] - a[1], b[3] - b[1], 1)
            hgap = max(b[0] - a[2], a[0] - b[2], 0)
            xover = min(a[2], b[2]) - max(a[0], b[0])
            same_row = abs(ay - by) < 0.5 * hmax and hgap < 1.2 * hmax
            stacked = 0 <= (b[1] - a[3]) < 0.7 * hmax and xover > 0 and (abs(a[0] - b[0]) < 1.5 * hmax or abs((a[0] + a[2]) - (b[0] + b[2])) / 2 < 1.5 * hmax)
            if same_row or stacked:
                uf.union(i, j, 45 * h, 16 * h)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def block_text(lines: list[dict], idx: list[int]) -> str:
    """Reading order: rows top->bottom, left->right inside a row."""
    h = float(np.median([lines[i]["bbox"][3] - lines[i]["bbox"][1] for i in idx])) or 1
    rows: list[list[int]] = []
    for i in sorted(idx, key=lambda k: (lines[k]["bbox"][1] + lines[k]["bbox"][3]) / 2):
        yc = (lines[i]["bbox"][1] + lines[i]["bbox"][3]) / 2
        if rows and abs(yc - np.mean([(lines[k]["bbox"][1] + lines[k]["bbox"][3]) / 2 for k in rows[-1]])) < 0.6 * h:
            rows[-1].append(i)
        else:
            rows.append([i])
    return "\n".join(" ".join(lines[k]["text"] for k in sorted(r, key=lambda k: lines[k]["bbox"][0])) for r in rows)


def union_box(boxes):
    b = np.array(boxes)
    return [int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())]


def where(bb, size) -> str:
    cx, cy = (bb[0] + bb[2]) / 2 / size[0], (bb[1] + bb[3]) / 2 / size[1]
    v = "upper" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "lower"
    hz = "left" if cx < 1 / 3 else "centre" if cx < 2 / 3 else "right"
    return f"{v}-{hz}"


def build_doc(doc: dict) -> tuple[list[dict], list[str]]:
    lines, size, title = doc["lines"], doc["size"], doc["doc"]
    if not lines:
        return [], []
    h = float(np.median([l["bbox"][3] - l["bbox"][1] for l in lines]))
    # ---- section headings: for each line the nearest heading above, per level, in the same text column
    heads = [(i, heading_level(l["text"])) for i, l in enumerate(lines)]
    heads = [(i, lv) for i, lv in heads if lv]

    def path_for(bb) -> str:
        best: dict[int, tuple[float, str]] = {}
        for i, lv in heads:
            hb = lines[i]["bbox"]
            dy = bb[1] - hb[1]
            dx = bb[0] - hb[0]
            if 0 <= dy < 60 * h and -2 * h <= dx < 25 * h:
                if lv not in best or dy < best[lv][0]:
                    best[lv] = (dy, norm(lines[i]["text"]))
        out, last = [], -1.0
        for lv in sorted(best):  # a deeper heading only counts if it is below the shallower one
            if best[lv][0] <= (last if last >= 0 else 1e9):
                out.append(best[lv][1])
                last = best[lv][0]
        return " > ".join(out)

    groups = group_blocks(lines)
    blocks = []
    for g in groups:
        bb = union_box([lines[i]["bbox"] for i in g])
        txt = block_text(lines, g)
        xs0 = [lines[i]["bbox"][0] for i in g]
        prose = len(g) >= 3 and (np.std(xs0) < 15 * h)
        blocks.append({"lines": g, "bbox": bb, "text": txt, "prose": prose})
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    chunks: list[dict] = []

    def add(kind, bb, body, section="", members=None):
        cid = f"{title}#{len(chunks)}"
        head = f"[{title} | {where(bb, size)} of sheet" + (f" | {section}" if section else "") + "]"
        chunks.append({"id": cid, "doc": title, "kind": kind, "bbox": bb, "section": section, "body": body, "text": head + "\n" + body,
                       "lines": [{"text": lines[i]["text"], "bbox": lines[i]["bbox"]} for i in members or []]})

    for b in blocks:
        sec = path_for(b["bbox"]) if (b["prose"] or heading_level(b["text"].split("\n")[0])) else ""
        add("block", b["bbox"], b["text"], sec, b["lines"])

    # ---- neighbourhood chunks: labels only make sense next to the labels around them
    R = 14 * h
    cents = np.array([[(b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2] for b in blocks])
    seen: set[tuple] = set()
    for k, b in enumerate(blocks):
        if b["prose"]:
            continue
        d = np.hypot(*(cents - cents[k]).T)
        near = [m for m in np.argsort(d) if d[m] < R][:14]
        key = tuple(sorted(near))
        if len(near) < 2 or key in seen:
            continue
        seen.add(key)
        near.sort(key=lambda m: (blocks[m]["bbox"][1] // (1.5 * h), blocks[m]["bbox"][0]))
        bb = union_box([blocks[m]["bbox"] for m in near])
        add("neighbourhood", bb, "\n".join(blocks[m]["text"].replace("\n", " ") for m in near))

    outline = [f"{'  ' * ((heading_level(l['text']) or 1) - 1)}{norm(l['text'])}" for l in sorted(lines, key=lambda l: (l["bbox"][0] // (30 * h), l["bbox"][1])) if heading_level(l["text"]) and heading_level(l["text"]) <= 2]
    return chunks, outline


def build_index(ocr_files: list[Path], out: Path, embed_id: str | None = None) -> None:
    from sentence_transformers import SentenceTransformer

    chunks, outlines, docs = [], {}, {}
    for f in ocr_files:
        d = json.loads(Path(f).read_text())
        c, o = build_doc(d)
        chunks += c
        outlines[d["doc"]] = o
        docs[d["doc"]] = {"image": d["image"], "size": d["size"]}
    model = SentenceTransformer(embed_id or config.EMBED_ID, device="cuda")
    model.half()
    emb = model.encode([c["text"] for c in chunks], batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "emb.npy", emb.astype(np.float16))
    (out / "chunks.json").write_text(json.dumps({"chunks": chunks, "outlines": outlines, "docs": docs}, ensure_ascii=False, indent=1))
    print(f"indexed {len(chunks)} chunks from {len(ocr_files)} document(s) -> {out}")
