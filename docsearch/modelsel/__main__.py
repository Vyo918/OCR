"""python -m docsearch.modelsel run <candidate> [--docs sys raft] [--limit N]
   python -m docsearch.modelsel score [candidates...] [--docs sys raft] [--misses N]
   python -m docsearch.modelsel zone <candidate> [<candidate> ...]     # only the tiny pipe-label band of the raft sheet
   python -m docsearch.modelsel sample <candidate> [<candidate> ...] [--k 16] [--from-run]   # K shared 640/120 regions, GT hit / digit / fold rates
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


# ----------------------------------------------------------------- sample
# Every candidate reads the same K regions (half per sheet) and is scored on the same GT strings inside them.
SAMPLE_TILE, SAMPLE_OVERLAP, SAMPLE_MARGIN = 640, 120, 64   # the margin lets a line cut by a region edge be read whole


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] < box[2] and box[1] <= pt[1] < box[3]


def _key(doc: str, box) -> str:
    return f"{doc} {box[0]} {box[1]} {box[2]} {box[3]}"


def _crop(box, size):
    W, H = size
    m = SAMPLE_MARGIN
    return (max(0, box[0] - m), max(0, box[1] - m), min(W, box[2] + m), min(H, box[3] + m))


def run_lines(doc: str) -> list[dict]:
    """Lines of every stored full-page run of one sheet (partial checkpoints excluded)."""
    out = []
    for p in sorted(OUT.glob(f"*.{doc}.json")):
        if ".part." not in p.name:
            out += json.loads(p.read_text())["lines"]
    return out


def locate(gts, lines_by_doc: dict) -> dict:
    """(sheet, GT text) -> centre of the best run line (rapidfuzz ratio >= 80 on norm()). Absent when no run reads it."""
    from rapidfuzz import fuzz, process

    texts = {d: [norm(l["text"]) for l in v] for d, v in lines_by_doc.items()}
    out = {}
    for doc, _, gt, _ in gts:
        if (doc, gt) in out or not texts.get(doc):
            continue
        hit = process.extractOne(norm(gt), texts[doc], scorer=fuzz.ratio, score_cutoff=80)
        if hit:
            x0, y0, x1, y1 = lines_by_doc[doc][hit[2]]["bbox"]
            out[(doc, gt)] = ((x0 + x1) / 2, (y0 + y1) / 2)
    return out


def pick_regions(k: int, grids: dict, located: dict) -> dict:
    """k // sheets regions per sheet: the tiles holding the most located GT centres (ties: top-left first)."""
    def count(doc, box):
        return sum(_inside(p, box) for (d, _), p in located.items() if d == doc)
    per = k // len(grids)
    return {doc: sorted(boxes, key=lambda b: (-count(doc, b), b[1], b[0]))[:per] for doc, boxes in grids.items()}


def scored(gts, located: dict, regions: dict) -> list:
    """GT strings whose located centre lies in a sampled region. Depends only on the regions, never on a candidate."""
    return [g for g in gts if (g[0], g[2]) in located and any(_inside(located[(g[0], g[2])], b) for b in regions.get(g[0], []))]


def spot_regions(m, pages: dict, regions: dict, path: Path) -> dict:
    """Runs one engine on each sampled region's crop (margin included). Checkpoints after every region in `path`, so a run resumes.
    Returns {region key: {"doc", "seconds", "lines"}} with boxes in page px."""
    OUT.mkdir(parents=True, exist_ok=True)
    done = json.loads(path.read_text())["regions"] if path.exists() else {}
    got = {}
    for doc, boxes in regions.items():
        for box in boxes:
            key = _key(doc, box)
            if key not in done:
                crop = _crop(box, pages[doc].size)
                t0 = time.time()
                found = m.spot(pages[doc].crop(crop))
                done[key] = {"doc": doc, "seconds": round(time.time() - t0, 2),
                             "lines": [{"text": t, "bbox": [round(crop[0] + a), round(crop[1] + b), round(crop[0] + c), round(crop[1] + d)]} for t, (a, b, c, d), _ in found]}
                path.write_text(json.dumps({"regions": done}, ensure_ascii=False))
            got[key] = done[key]
    return got


def run_regions(cand: str, pages: dict, regions: dict, grids: dict) -> dict:
    """The same regions read from a stored full-page run: lines whose centre lies in the crop. Seconds are the run's
    average per ink tile, since the run does not time regions."""
    got = {}
    for doc, boxes in regions.items():
        run = json.loads((OUT / f"{cand}.{doc}.json").read_text())
        per = run["seconds"] / len(grids[doc])
        for box in boxes:
            crop = _crop(box, pages[doc].size)
            lines = [l for l in run["lines"] if _inside(((l["bbox"][0] + l["bbox"][2]) / 2, (l["bbox"][1] + l["bbox"][3]) / 2), crop)]
            got[_key(doc, box)] = {"doc": doc, "seconds": round(per, 2), "lines": lines}
    return got


def _pct(a, b):
    return round(100 * a / b, 1) if b else None


def sample_metrics(gts, lines_by_doc: dict, secs: list) -> dict:
    """hit: GT string inside a line (partial ratio >= 90). digit: of the GT strings with numbers, those whose numbers all
    appear in a hit line. fold: share of CJK output characters that fold() rewrites (the Traditional forms)."""
    from rapidfuzz import fuzz, process
    from ..index import fold

    texts = {d: [norm(l["text"]) for l in v] for d, v in lines_by_doc.items()}
    hit = digit = with_num = 0
    for doc, _, gt, _ in gts:
        g, cands = norm(gt), texts.get(doc, [])
        found = process.extract(g, cands, scorer=fuzz.partial_ratio, score_cutoff=90, limit=None)
        hit += bool(found)
        nums = numbers(g)
        if nums:
            with_num += 1
            digit += any(all(n in numbers(cands[i]) for n in nums) for _, _, i in found)
    chars = [c for v in lines_by_doc.values() for l in v for c in l["text"] if "\u4e00" <= c <= "\u9fff"]
    changed = {c: fold(c) != c for c in set(chars)}
    return {"GT": len(gts), "hit": _pct(hit, len(gts)), "digit": _pct(digit, with_num),
            "fold": _pct(sum(changed[c] for c in chars), len(chars)),
            "lines": sum(len(v) for v in lines_by_doc.values()), "sec": round(sum(secs) / len(secs), 1) if secs else None}


def _fmt(v):
    return "-" if v is None else f"{v:.1f}"


def sample(cands: list[str], k: int, from_run: bool):
    """Scores each candidate on K sampled regions of the sheets (640/120 cut, most located GT first). --from-run reads the
    regions from the stored full-page runs instead of running the candidate."""
    from ..engines import make

    pages = {d: Image.open(p).convert("RGB") for d, p in PAGES.items()}
    located = locate(GT, {d: run_lines(d) for d in PAGES})
    grids = {d: [t for t in tiles(*page.size, SAMPLE_TILE, SAMPLE_OVERLAP) if ink_ratio(page.crop(t)) >= 0.004] for d, page in pages.items()}
    regions = pick_regions(k, grids, located)
    gts = scored(GT, located, regions)
    for d, boxes in regions.items():
        for b in boxes:
            print(f"[{d}] tile {b}: {sum(_inside(p, b) for (dd, _), p in located.items() if dd == d)} located GT")
    print(f"{len(gts)} scored GT strings (of {len(GT)})\n")
    rows = []
    for cand in cands:
        if from_run:
            got = run_regions(cand, pages, regions, grids)
        else:
            m = make(cand)
            try:
                got = spot_regions(m, pages, regions, OUT / f"sample.{cand}.json")
            finally:
                m.close()
        by_doc = defaultdict(list)
        for e in got.values():
            by_doc[e["doc"]] += e["lines"]
        rows.append((cand, sample_metrics(gts, by_doc, [e["seconds"] for e in got.values()])))
    print("| candidate | GT | hit % | digit exact % | fold % | lines | sec/region |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for cand, r in rows:
        print(f"| {cand} | {r['GT']} | {_fmt(r['hit'])} | {_fmt(r['digit'])} | {_fmt(r['fold'])} | {r['lines']} | {_fmt(r['sec'])} |")
    print("\nfold % = CJK output characters that fold() rewrites (Traditional forms)."
          + (" from-run sec/region = the run's average per ink tile." if from_run else ""))
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("cand"); r.add_argument("--docs", nargs="+", default=["sys", "raft"]); r.add_argument("--limit", type=int)
    s = sub.add_parser("score"); s.add_argument("cands", nargs="*", default=None); s.add_argument("--docs", nargs="+"); s.add_argument("--misses", type=int, default=70)
    z = sub.add_parser("zone"); z.add_argument("cands", nargs="+"); z.add_argument("--from-run", action="store_true", help="score the stored full-page run instead of re-reading the band")
    sm = sub.add_parser("sample"); sm.add_argument("cands", nargs="+"); sm.add_argument("--k", type=int, default=16)
    sm.add_argument("--from-run", action="store_true", help="read the regions from the stored full-page runs (hybrid, hunyuan) instead of running the candidate")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cand, a.docs, a.limit)
    elif a.cmd == "zone":
        zone(a.cands, a.from_run)
    elif a.cmd == "sample":
        sample(a.cands, a.k, a.from_run)
    else:
        score(a.cands or sorted({p.name.split(".")[0] for p in OUT.glob("*.json") if ".part." not in p.name and not p.name.startswith("sample.")}), a.docs, a.misses)


if __name__ == "__main__":
    main()
