"""python -m docsearch.modelsel run <candidate> [--docs sys raft] [--limit N]
   python -m docsearch.modelsel score [candidates...] [--docs sys raft] [--misses N]
   python -m docsearch.modelsel zone <candidate> [<candidate> ...]     # only the tiny pipe-label band of the raft sheet
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config
from ..engines import clean_text
from ..ocr_ingest import ink_ratio, merge_overlap, tiles
from .gt import GT

Image.MAX_IMAGE_PIXELS = None
PAGES = {"sys": Path("work/manual/sys-000.jpg"), "raft": Path("work/manual/raft-000.jpg")}
OUT = Path("work/bakeoff")

_MAP = str.maketrans({"×": "x", "≒": "=", "≈": "=", "ø": "φ", "Ø": "φ", "Φ": "φ", "、": ",", "，": ",", "。": ".", "．": ".", "‧": ".", "·": ".", "”": '"', "“": '"', "″": '"', "'": '"', "：": ":", "﹕": ":", "∶": ":"})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_MAP)
    s = s.replace("O", "0") if False else s
    return re.sub(r"[\s\u3000]+", "", s).lower()


# ------------------------------------------------------------------ run
def run(cand: str, docs: list[str], limit: int | None):
    """Resumable: raw (un-merged) lines + seconds are checkpointed every 5 tiles in <cand>.<doc>.part.json."""
    from ..engines import make

    OUT.mkdir(parents=True, exist_ok=True)
    m = make(cand)
    for doc in docs:
        final, part = OUT / f"{cand}.{doc}.json", OUT / f"{cand}.{doc}.part.json"
        if final.exists() and not limit:
            print(f"[{cand}/{doc}] already done"); continue
        page = Image.open(PAGES[doc]).convert("RGB")
        if hasattr(m, "read_page"):   # whole-page engine (cascade): no per-tile checkpointing
            t0 = time.time()
            lines = m.read_page(page, doc, limit, on_tile=lambda n, tot, k: print(f"[{cand}/{doc}] {n}/{tot} -> {k} lines {time.time() - t0:.0f}s", flush=True))
            if not limit:
                final.write_text(json.dumps({"seconds": time.time() - t0, "tiles": len(lines), "lines": lines, "stats": getattr(m, "stats", {})}, ensure_ascii=False))
            print(f"[{cand}/{doc}] {len(lines)} lines, {time.time() - t0:.0f}s {getattr(m, 'stats', '')}")
            continue
        W, H = page.size
        grid = [t for t in tiles(W, H, m.tile, m.overlap) if ink_ratio(page.crop(t)) >= 0.004]
        st = json.loads(part.read_text()) if part.exists() and not limit else {"done": 0, "seconds": 0.0, "lines": []}
        lines, t0 = st["lines"], time.time() - st["seconds"]
        for n, (x0, y0, x1, y1) in enumerate(grid[:limit], 1):
            if n <= st["done"]:
                continue
            for text, (a, b, c, d), score in m.spot(page.crop((x0, y0, x1, y1))):
                lines.append({"doc": doc, "text": text, "bbox": [round(x0 + a), round(y0 + b), round(x0 + c), round(y0 + d)], **({"score": round(score, 3)} if score is not None else {})})
            if n % 5 == 0 or n == len(grid):
                print(f"[{cand}/{doc}] tile {n}/{len(grid)} lines={len(lines)} {time.time() - t0:.0f}s", flush=True)
                if not limit:
                    part.write_text(json.dumps({"done": n, "seconds": time.time() - t0, "lines": lines}, ensure_ascii=False))
        dt = time.time() - t0
        lines = merge_overlap(lines)
        if limit:
            for l in lines[:25]:
                print("   ", l["bbox"], l["text"])
            print(f"[{cand}/{doc}] smoke run: {len(lines)} lines, {dt:.0f}s (not saved)")
            continue
        final.write_text(json.dumps({"seconds": dt, "tiles": len(grid), "lines": lines}, ensure_ascii=False))
        part.unlink(missing_ok=True)
        print(f"[{cand}/{doc}] {len(lines)} lines, {dt:.0f}s")


# ----------------------------------------------------------------- score
def rows(lines):
    """Individual lines + same-row concatenations (a model may split one printed line into several boxes)."""
    if not lines:
        return []
    h = float(np.median([l["bbox"][3] - l["bbox"][1] for l in lines])) or 1.0
    out = [norm(clean_text(l["text"])) for l in lines]
    ls = sorted(lines, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2)
    cur: list[dict] = []
    clusters = []
    for l in ls:
        yc = (l["bbox"][1] + l["bbox"][3]) / 2
        if cur and abs(yc - np.mean([(c["bbox"][1] + c["bbox"][3]) / 2 for c in cur])) < 0.7 * h:
            cur.append(l)
        else:
            if cur:
                clusters.append(cur)
            cur = [l]
    clusters.append(cur)
    for c in clusters:
        c = sorted(c, key=lambda l: l["bbox"][0])
        for i in range(len(c)):
            for j in range(i + 1, min(i + 4, len(c)) + 1):
                if j - i > 1:
                    out.append(norm("".join(x["text"] for x in c[i:j])))
    return out


def best_match(gt: str, texts: list[str]):
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    g = norm(gt)
    best = (0.0, "", False)
    for t in texts:
        if not t:
            continue
        if g in t:
            return 1.0, g, True
        al = fuzz.partial_ratio_alignment(g, t)
        if al is None or al.score < 40:
            continue
        w = t[al.dest_start:al.dest_end]
        acc = 1 - Levenshtein.distance(g, w) / max(len(g), 1)
        if acc > best[0]:
            best = (acc, w, False)
    return best


def numbers(s: str):
    return re.findall(r"\d+(?:\.\d+)?", s)


def score(cands: list[str], only_docs: list[str] | None = None, show_misses: int = 70):
    res, recs = {}, {}
    for cand in cands:
        data = {d: json.loads((OUT / f"{cand}.{d}.json").read_text()) for d in (only_docs or PAGES) if (OUT / f"{cand}.{d}.json").exists()}
        if not data:
            continue
        texts = {d: rows(v["lines"]) for d, v in data.items()}
        per = defaultdict(list)
        misses = []
        recs[cand] = {}
        for doc, cat, gt, crit in GT:
            if doc not in data:
                continue
            acc, win, exact = best_match(gt, texts[doc])
            nums = numbers(norm(gt))
            num_ok = (sum(1 for n in nums if n in numbers(win)) / len(nums)) if nums else None
            rec = {"acc": acc, "exact": exact, "hit": acc >= 0.85, "num_ok": num_ok, "crit": crit}
            recs[cand][(doc, gt)] = rec
            per[cat].append(rec)
            per["ALL"].append(rec)
            if crit:
                per["CRITICAL"].append(rec)
            if not rec["hit"]:
                misses.append((doc, cat, gt, win, round(acc, 2)))
        # phantom lines: boxes sitting on blank paper
        phantom = n_lines = 0
        for d, v in data.items():
            g = np.asarray(Image.open(PAGES[d]).convert("L"))
            for l in v["lines"]:
                x0, y0, x1, y1 = l["bbox"]
                patch = g[max(y0, 0):y1, max(x0, 0):x1]
                n_lines += 1
                if patch.size == 0 or (patch < 170).mean() < 0.004:
                    phantom += 1
        res[cand] = (per, misses, {"lines": n_lines, "phantom": phantom, "sec": sum(v["seconds"] for v in data.values()), "docs": sorted(data)})

    def pct(r, k):
        v = [x[k] for x in r if x[k] is not None]
        return f"{100 * np.mean(v):5.1f}" if v else "    -"

    print(f"{'candidate':9s} {'GT':>3s} | {'exact%':>6s} {'hit%':>6s} {'char%':>6s} {'digit%':>6s} | CRITICAL {'n':>3s} {'hit%':>6s} {'digit%':>6s} | {'lines':>5s} {'phantom%':>8s} {'sec':>5s}")
    for cand, (per, _, st) in res.items():
        a, c = per["ALL"], per["CRITICAL"]
        print(f"{cand:9s} {len(a):3d} | {pct(a, 'exact'):>6s} {pct(a, 'hit'):>6s} {pct(a, 'acc'):>6s} {pct(a, 'num_ok'):>6s} |          {len(c):3d} {pct(c, 'hit'):>6s} {pct(c, 'num_ok'):>6s} | {st['lines']:5d} {100 * st['phantom'] / max(st['lines'], 1):8.1f} {st['sec']:5.0f}")
    cats = ["legend", "calc", "label", "pump-table", "pipe-label"]
    print("\nhit% (>=85% of characters recovered)  |  digit% (numbers in the GT string found intact)")
    print(f"{'candidate':9s} " + " ".join(f"{c:>17s}" for c in cats))
    for cand, (per, _, _) in res.items():
        print(f"{cand:9s} " + " ".join(f"{(pct(per[c], 'hit') + ' | ' + pct(per[c], 'num_ok')) if per.get(c) else '-':>17s}" for c in cats))
    if len(recs) > 1:
        keys = set.intersection(*[set(r) for r in recs.values()])
        best = {key: max((recs[c][key] for c in recs), key=lambda r: r["acc"]) for key in keys}
        print(f"\noracle union of {sorted(recs)} on {len(keys)} GT strings: hit {100 * np.mean([r['hit'] for r in best.values()]):.1f}%  "
              f"char {100 * np.mean([r['acc'] for r in best.values()]):.1f}%")
    for cand, (_, misses, _) in res.items():
        print(f"\n--- {cand}: {len(misses)} GT strings not recovered (>=85% chars)")
        for m in misses[:show_misses]:
            print("  ", m)


# ----------------------------------------------------------------- zone
ZONE = ("raft", (0.28, 0.60, 0.52, 0.78))   # band holding the 4"/6" PVC connecting-pipe labels (benchmark question 3)


def zone(cands: list[str], from_run: bool = False):
    """GT scoring restricted to the hardest text on the sheets - tiny coloured pipe labels over the raft drawing's grid.
    Default: the candidate reads only the cropped band, through the same tiling/merge code as ingestion, so a slow VLM can be
    compared with the PP-OCR family on exactly the text that decides question 3.
    from_run: score the lines of the candidate's stored full-page run (`run`) that lie inside the band instead - what the
    shipped pipeline really produces for that region (needed for the hybrid engine, whose cascade works on whole pages)."""
    from ..engines import make
    from ..ocr_ingest import read_page

    doc, (fx0, fy0, fx1, fy1) = ZONE
    page = Image.open(PAGES[doc]).convert("RGB")
    W, H = page.size
    box = (int(fx0 * W), int(fy0 * H), int(fx1 * W), int(fy1 * H))
    crop = page.crop(box)
    keys = [g[2] for g in GT if g[0] == doc and g[1] == "pipe-label" and g[3]]
    print(f"zone {crop.size} of the {doc} sheet ({'stored full-page run' if from_run else 'band re-read'}), {len(keys)} critical GT strings: {keys}")
    print(f"{'candidate':14s} {'lines':>5s} {'hit':>5s} {'char%':>6s} {'digit%':>6s} {'sec':>5s}")
    for cand in cands:
        m, t0 = None, time.time()
        if from_run:
            data = json.loads((OUT / f"{cand}.{doc}.json").read_text())
            lines = [l for l in data["lines"]
                     if box[0] <= (l["bbox"][0] + l["bbox"][2]) / 2 < box[2] and box[1] <= (l["bbox"][1] + l["bbox"][3]) / 2 < box[3]]
            dt = data.get("seconds", 0)
        else:
            m = make(cand)
            lines = read_page(crop, m)
            dt = time.time() - t0
        texts = rows(lines)
        accs, nums_ok = [], []
        for g in keys:
            acc, win, _ = best_match(g, texts)
            accs.append(acc)
            ns = numbers(norm(g))
            nums_ok += [n in numbers(win) for n in ns]
        print(f"{cand:14s} {len(lines):5d} {sum(a >= 0.85 for a in accs):2d}/{len(keys)} {100 * np.mean(accs):6.1f} {100 * np.mean(nums_ok):6.1f} {dt:5.0f}", flush=True)
        if m is not None and hasattr(m, "close"):
            m.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("cand"); r.add_argument("--docs", nargs="+", default=["sys", "raft"]); r.add_argument("--limit", type=int)
    s = sub.add_parser("score"); s.add_argument("cands", nargs="*", default=None); s.add_argument("--docs", nargs="+"); s.add_argument("--misses", type=int, default=70)
    z = sub.add_parser("zone"); z.add_argument("cands", nargs="+"); z.add_argument("--from-run", action="store_true", help="score the stored full-page run instead of re-reading the band")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cand, a.docs, a.limit)
    elif a.cmd == "zone":
        zone(a.cands, a.from_run)
    else:
        score(a.cands or sorted({p.name.split(".")[0] for p in OUT.glob("*.json") if ".part." not in p.name}), a.docs, a.misses)


if __name__ == "__main__":
    main()
