"""Crop re-read checks (MODELS.md section 4.3; plan in local://crop-reread-plan.md, sections 4 and 5 T3).

  python -m docsearch.rereadcheck precheck [--out DIR]
      crops and reader readings for targets A (fact rows) and B (GT-matched drawing lines); writes <n>.png,
      report.json and report.md under DIR (default work/reread-precheck/<backend>) and prints the summary.
  python -m docsearch.rereadcheck fill BENCH.json (--ids A,B | --ids-file FILE)
      retrieval + re-read per question id, no LLM; prints id, crop count, disagreements and seconds only.

Both need the reader switched on: DOCSEARCH_REREAD=openai (server at DOCSEARCH_REREAD_URL) or hunyuan.
Nothing here prints or writes question, answer or GT text: only ids, categories, counts and seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from . import config, facts, reread
from .modelsel.__main__ import norm
from .modelsel.gt import GT

SHEET = {"sys": "系統", "raft": "筏基"}   # GT sheet key -> substring of the index document name
ATTRS_A = ("bore", "tank_dims", "label_depth", "label_width", "plan_volume")
TANK = ("tank_dims", "label_depth", "plan_volume")
GT_MIN = 70           # fuzz.ratio on norm() strings needed to take a block line as the GT string's line
P1_MAX_OCR_ONLY = 1   # lines where OCR digits are right and the reader's are wrong
P3_MIN = 0.9          # share of tank targets where the reader agrees with OCR


def backend() -> str:
    if config.REREAD == "0":
        raise SystemExit("the crop re-reader is off: set DOCSEARCH_REREAD=openai (or hunyuan)")
    return config.REREAD


def cer(ref: str, hyp: str) -> float:
    """Normalised Levenshtein distance on modelsel.norm (0 = identical)."""
    return Levenshtein.normalized_distance(norm(ref), norm(hyp))


def digit_exact(ref: str, hyp: str) -> bool:
    return reread.digits(norm(ref)) == reread.digits(norm(hyp))


# ----------------------------------------------------------------------------------------------- targets
def targets_a(rows: list[dict], by_id: dict) -> list[dict]:
    """Fact rows with an attr in ATTRS_A -> targets, built by reread.select (same-row merge, dedupe)."""
    wanted = [r for r in rows if r.get("attr") in ATTRS_A]
    return reread.select([], {"rows": wanted}, by_id, sys.maxsize)


def targets_b(chunks: list[dict]) -> tuple[list[dict], int]:
    """One item per GT string with a digit, matched to the best block line of its sheet; -> (items, unmatched)."""
    pool = {key: [] for key in SHEET}
    for c in chunks:
        if c.get("kind") == "block":
            for key, kw in SHEET.items():
                if kw in c["doc"]:
                    pool[key].extend((c, ln, norm(ln["text"])) for ln in c.get("lines", []))
    items, unmatched = [], 0
    for i, (sheet, cat, text, _critical) in enumerate(GT):
        if not reread.DIGIT.search(text):
            continue
        g = norm(text)
        best = None
        for c, ln, n in pool[sheet]:
            score = fuzz.ratio(g, n)
            if score >= GT_MIN and (best is None or score > best[0]):
                best = (score, c, ln)
        if best is None:
            unmatched += 1
            continue
        score, c, ln = best
        items.append({"kind": "B", "target": reread._target(c, [ln], {}), "attrs": [], "gt": text,
                      "gt_index": i, "category": cat, "score": round(score, 1)})
    return items, unmatched


def _items(chunks: list[dict], docs: dict) -> tuple[list[dict], int]:
    rows = facts.extract(chunks, docs)
    attr_of = {r["id"]: r.get("attr") for r in rows}
    a = [{"kind": "A", "target": t, "attrs": sorted({attr_of[i] for i in t["rows"]})}
         for t in targets_a(rows, {c["id"]: c for c in chunks})]
    b, unmatched = targets_b(chunks)
    return a + b, unmatched


# ----------------------------------------------------------------------------------------------- summary
def _p90(xs: list[float]) -> float:
    return xs[math.ceil(0.9 * len(xs)) - 1]


def summarize(recs: list[dict], unmatched: int = 0) -> dict:
    secs = sorted(r["seconds"] for r in recs)
    b = [r for r in recs if r["kind"] == "B"]
    tank = [r for r in recs if r["kind"] == "A" and set(r["attrs"]) & set(TANK)]
    ocr_ok = sum(r["digits_ocr_ok"] for r in b)
    vlm_ok = sum(r["digits_vlm_ok"] for r in b)
    ocr_only = sum(r["digits_ocr_ok"] and not r["digits_vlm_ok"] for r in b)
    tank_agree = sum(r["status"] == "agree" for r in tank)
    return {
        "crops": len(recs),
        "cached": sum(r["cached"] for r in recs),
        "mean_s": round(statistics.fmean(secs), 2) if secs else None,
        "p90_s": round(_p90(secs), 2) if secs else None,
        "status": {s: sum(r["status"] == s for r in recs) for s in ("agree", "disagree", "rejected")},
        "B": {"matched": len(b), "unmatched": unmatched, "ocr_digit_exact": ocr_ok, "vlm_digit_exact": vlm_ok,
              "ocr_right_vlm_wrong": ocr_only, "vlm_right_ocr_wrong": sum(r["digits_vlm_ok"] and not r["digits_ocr_ok"] for r in b),
              "cer_ocr": round(statistics.fmean(r["cer_ocr"] for r in b), 3) if b else None,
              "cer_vlm": round(statistics.fmean(r["cer_vlm"] for r in b), 3) if b else None},
        "P1": bool(b) and vlm_ok >= ocr_ok and ocr_only <= P1_MAX_OCR_ONLY,
        "tank": {"targets": len(tank), "agree": tank_agree},
        "P3": bool(tank) and tank_agree / len(tank) >= P3_MIN,
        "P2_bore_crops": [r["crop"] for r in recs if "bore" in r["attrs"]],   # for the human check
    }


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _report_md(summary: dict, recs: list[dict], model: str) -> str:
    head = ["# Crop re-read precheck", "", f"reader: {model}", "", "```json", json.dumps(summary, ensure_ascii=False, indent=1),
            "```", "", "| n | kind | attrs / category | doc | OCR | VLM | status | s | CER ocr / vlm | digits ocr / vlm |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    body = []
    for r in recs:
        tag = ",".join(r["attrs"]) if r["kind"] == "A" else r["category"]
        cers = f"{r['cer_ocr']} / {r['cer_vlm']}" if r["kind"] == "B" else ""
        dig = f"{r['digits_ocr_ok']} / {r['digits_vlm_ok']}" if r["kind"] == "B" else ""
        body.append(f"| {r['n']} | {r['kind']} | {tag} | {r['doc']} | {_cell(r['ocr'])} | {_cell(r['vlm'])} | {r['status']} "
                    f"| {r['seconds']} | {cers} | {dig} |")
    return "\n".join(head + body) + "\n"


# ----------------------------------------------------------------------------------------------- commands
def _load_index() -> tuple[list[dict], dict]:
    data = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return data["chunks"], data["docs"]


def precheck(out: Path | None = None) -> dict:
    name = backend()
    out = Path(out) if out else config.WORK / "reread-precheck" / name
    chunks, docs = _load_index()
    items, unmatched = _items(chunks, docs)
    if not items:
        raise SystemExit("no targets: the index has no fact rows with an A attr and no GT digit string matched a block line")
    out.mkdir(parents=True, exist_ok=True)
    rr = reread.Rereader(docs, chunks)
    recs = []
    for n, item in enumerate(items, 1):
        t = item["target"]
        reread.crop(rr._page(t["doc"]), t["box"], t["h"]).save(out / f"{n}.png")
        r = rr.read_target(t)
        rec = {"n": n, "kind": item["kind"], "doc": t["doc"], "chunk": t["chunk"], "rows": t["rows"], "attrs": item["attrs"],
               "crop": f"{n}.png", "ocr": t["text"], "vlm": r["text"], "status": reread.reconcile(t["text"], r["text"]),
               "seconds": r["seconds"], "cached": r["cached"]}
        if item["kind"] == "B":
            ref = item["gt"]
            rec.update({"gt_index": item["gt_index"], "category": item["category"], "score": item["score"],
                        "cer_ocr": round(cer(ref, t["text"]), 3), "cer_vlm": round(cer(ref, r["text"]), 3),
                        "digits_ocr_ok": digit_exact(ref, t["text"]), "digits_vlm_ok": digit_exact(ref, r["text"])})
        recs.append(rec)
    summary = summarize(recs, unmatched)
    (out / "report.json").write_text(json.dumps({"backend": name, "model": rr.model, "summary": summary, "targets": recs},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(_report_md(summary, recs, rr.model), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"report: {out / 'report.md'}")
    return summary


def fill(bench: Path, ids: list[str]) -> None:
    backend()
    from .cli import _engine   # the same Engine factory as the docsearch CLI
    questions = {q.get("編號"): q for q in json.loads(Path(bench).read_text(encoding="utf-8"))["題目"]}
    missing = [i for i in ids if i not in questions]
    if missing:
        raise SystemExit(f"ids not in {bench}: {', '.join(missing)}")
    eng = _engine()
    for qid in ids:
        p = eng.prepare(questions[qid]["問題"])
        recs = eng.rereader().run(p["ev"], p["fx"])["records"]
        print(f"{qid}\tcrops={len(recs)}\tdisagree={sum(r['status'] == 'disagree' for r in recs)}"
              f"\tseconds={sum(r['seconds'] for r in recs):.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m docsearch.rereadcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("precheck", help="crops and reader readings on targets A and B, with the pre-registered rules")
    pc.add_argument("--out", type=Path, default=None, help="default work/reread-precheck/<backend>")
    fl = sub.add_parser("fill", help="retrieval + re-read per question id (no LLM)")
    fl.add_argument("bench", type=Path, help="benchmark questions JSON (題目 list)")
    grp = fl.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ids", help="comma-separated question ids")
    grp.add_argument("--ids-file", type=Path, help="question ids separated by commas or whitespace")
    a = ap.parse_args(argv)
    if a.cmd == "precheck":
        precheck(a.out)
    else:
        text = a.ids if a.ids is not None else a.ids_file.read_text(encoding="utf-8")
        fill(a.bench, [i for i in re.split(r"[,\s]+", text) if i])


if __name__ == "__main__":
    main()
