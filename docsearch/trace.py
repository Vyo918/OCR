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
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config, factquery
from .entities import anchor, chunk_entities, question_entities

TEMPLATE = Path(__file__).with_name("trace_template.html")
PLACEHOLDER = "__TRACE_DATA__"
LEX_K = 60                  # per-ranker depth and candidate pool of Engine.hybrid (its defaults)
RRF_K = 60                  # RRF constant of Engine.hybrid
LIST_N = 10                 # ids shown per ranking
IMAGE_BUDGET = 400 * 1024   # bytes of base64-decoded PNG data in the page, all crops together
CROP_PAD = 60               # page pixels around the source rows
VERDICT = {"pass": "correct", "fail": "incorrect"}


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


def _row(key, text="", meta="") -> dict:
    return {"key": str(key), "text": str(text), "meta": str(meta)}


def _img(label: str, src: str, caption: str) -> dict:
    return {"t": "img", "label": label, "src": src, "caption": caption}


def _stage(sid: str, short: str, title: str, what: str, inp: list, out: list, view: str = "blocks") -> dict:
    return {"id": sid, "short": short, "title": title, "what": what, "input": inp, "output": out, "view": view}


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


def _region(size: list, boxes: list[list]) -> list[int]:
    return [max(0, round(min(b[0] for b in boxes) - CROP_PAD)), max(0, round(min(b[1] for b in boxes) - CROP_PAD)),
            min(size[0], round(max(b[2] for b in boxes) + CROP_PAD)), min(size[1], round(max(b[3] for b in boxes) + CROP_PAD))]


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


def _crop_url(path: Path, size: list, region: list[int], boxes: list[list], budget: int) -> str:
    """PNG data URL of `region` (page px in `size` coordinates) with the source rows boxed in red. Downscaled and
    colour-reduced until it fits `budget` bytes; if nothing fits, the smallest variant is returned."""
    from PIL import Image, ImageDraw

    src = Image.open(path).convert("RGB")
    sx, sy = src.width / size[0], src.height / size[1]
    x0, y0, x1, y1 = region
    im = src.crop((round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)))
    draw = ImageDraw.Draw(im)
    for b in boxes:
        draw.rectangle([(b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy],
                       outline=(220, 30, 30), width=max(2, round(3 * sx)))
    buf = io.BytesIO()
    for scale in (1.0, 0.8, 0.65, 0.5, 0.4, 0.3):
        out = im if scale == 1.0 else im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        for variant in (out, out.quantize(colors=64)):
            buf = io.BytesIO()
            variant.save(buf, format="PNG", optimize=True)
            if buf.tell() <= budget:
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _answer(label: str, rec: dict | None) -> dict:
    if not rec:
        return {"label": label, "answer": None, "grade": None, "verdict": None, "evidence": [], "facts": [], "computed": []}
    grade = rec.get("grade")
    return {"label": label, "answer": rec.get("answer", ""), "grade": grade, "verdict": VERDICT.get(grade),
            "evidence": list(rec.get("evidence_ids") or []),
            "facts": [f["id"] if isinstance(f, dict) else str(f) for f in rec.get("facts") or []],
            "computed": list(rec.get("computed") or [])}


# ------------------------------------------------------------------ collect
def collect(engine, question: str, *, top: int = config.TOP_K, question_id: str = "", category: str = "", difficulty: str = "",
            reference: str | None = None, recorded: dict | None = None, pdfs: list[Path] = (), pages_dir: Path | None = None,
            ocr_dir: Path | None = None) -> dict:
    """Run the production retrieval and prompt code on `question`, keeping each stage's input and output.
    `recorded`: {"base": result of the run without fact rows, "facts": result of the run with them} (either may be None)."""
    recorded = recorded or {}
    chunks = engine.chunks
    by_id = {c["id"]: c for c in chunks}
    ocr_dir = ocr_dir or config.OCR_DIR
    pages_dir = pages_dir or config.WORK / "trace-pages"

    # question
    ents = question_entities(question)
    wanted = {e for e in ents if e[1] is not None}
    query = factquery.parse(question)

    # query expansion, BM25 and dense per query, RRF -> candidate pool (the calls Engine.hybrid makes)
    queries = engine.expand(question)
    lexical = [engine.lexical(q, LEX_K) for q in queries]
    dense = engine.dense(queries, LEX_K)
    rankings = lexical + dense
    fused = fuse(rankings)
    cand = [i for i, _ in fused][:LEX_K]

    # cross-encoder rerank, novelty filter, entity anchoring (the branches of Engine.answer; the rerank is the same call)
    scores = np.asarray(engine.reranker().predict([(question, chunks[i]["text"]) for i in cand], batch_size=8))
    score_of = {cand[j]: float(scores[j]) for j in range(len(cand))}
    order = [cand[j] for j in np.argsort(-scores)]
    novel = engine.novel(order, top)
    best = anchor(question, novel, order, chunks, top) if wanted else novel
    ev = [chunks[i] for i in best]

    # fact rows and both prompts (A: flag off, B: fact rows added)
    fx = factquery.context(question, engine.facts)
    fact_rows = fx["rows"] if fx else []
    system_a, user_a = engine.build_prompt(question, ev, None)
    system_b, user_b = engine.build_prompt(question, ev, fx)

    # source rows: the rows the fact rows were parsed from, else the top evidence chunks
    boxes = [(by_id[s["chunk"]]["doc"], tuple(s["bbox"])) for r in fact_rows for s in r["src"] if s["chunk"] in by_id]
    if not boxes:
        boxes = [(c["doc"], tuple(c["bbox"])) for c in ev[:3]]
    boxes = list(dict.fromkeys(boxes))[:8]
    docs = list(dict.fromkeys(d for d, _ in boxes))

    regions = {doc: _region(engine.docs[doc]["size"], [list(b) for d, b in boxes if d == doc]) for doc in docs}
    page_files = {doc: _page_path(engine, doc, list(pdfs), pages_dir) for doc in docs}
    rendered = [d for d in docs if page_files[d]]
    stages: list[dict] = []

    # 1 question
    stages.append(_stage("question", "Question", "Question",
                         "The question is read before any retrieval: the drawing units it names (entities) and the kind of fact it asks for.",
                         [_text("Question", question)],
                         [_rows("Entities", [_row(fam, ident or "no unit number", "anchoring key" if ident else "") for fam, ident in ents]),
                          _text("Fact intent", _query_text(query))]))

    # 2 input page
    page_out = [_img(d, _crop_url(page_files[d], engine.docs[d]["size"], regions[d],
                                  [list(b) for dd, b in boxes if dd == d], (IMAGE_BUDGET * 3 // 4) // len(rendered)),
                     f"{d} · region {regions[d]} · red = source rows") for d in rendered]
    stages.append(_stage("page", "Input page", "Input page",
                         "The sheet the answer lives on. Only the region around the top source rows is cropped from the page image.",
                         [_rows("Source rows", [_row(d, f"bbox {list(b)}") for d, b in boxes])],
                         page_out or [_text("Page image", "not available: the index image is not on this machine; pass --pdf to render it")]))

    # 3 OCR lines inside the region
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
                         "OCR (ingest) turned the page into text lines with boxes. These are the lines whose centre is inside the crop.",
                         [_text("Crop", "; ".join(f"{d}: {regions[d]}" for d in docs) or "no source rows")],
                         [_rows("Lines in the region", ocr_rows)] + ([_text("Note", "; ".join(ocr_note))] if ocr_note else [])))

    # 4 chunks overlapping the region (the units the retrieval ranks)
    evid = {c["id"] for c in ev}
    inregion = [c for c in chunks if c["doc"] in regions and _overlaps(c["bbox"], regions[c["doc"]])][:12]
    stages.append(_stage("chunks", "Chunks", "Chunks",
                         "Lines are grouped into block chunks (one label cluster), section chunks (a headed group) and neighbourhood chunks (a window of blocks). BM25 and the embedder index their text.",
                         [_text("Index", f"{len(chunks)} chunks over {len(engine.docs)} document(s)")],
                         [_rows("Chunks overlapping the crop", [_row(c["id"], c["body"], f"{c['kind']} · {c.get('section') or '-'}"
                                                                 + (" · in evidence" if c["id"] in evid else ""))
                                                                for c in inregion])]))

    # 5 expansion
    stages.append(_stage("expand", "Query expansion", "Query expansion",
                         "Engineers and users name the same things differently. Every glossary group the question touches adds one lexical query.",
                         [_text("Question", question)],
                         [_rows("Search queries", [_row(f"query {n}", q, "question" if n == 0 else "glossary group") for n, q in enumerate(queries)])]))

    # 6 BM25 + dense
    rank_rows = [_row(f"BM25 · query {n}", _ids(engine, r[:LIST_N]), f"top {LIST_N} of {LEX_K}") for n, r in enumerate(lexical)]
    rank_rows += [_row(f"dense · query {n}", _ids(engine, r[:LIST_N]), f"top {LIST_N} of {LEX_K}") for n, r in enumerate(dense)]
    stages.append(_stage("retrieve", "BM25 + dense", "BM25 + dense",
                         "Each query runs a character-n-gram BM25 over the chunk text and a BGE-M3 dense search over the chunk embeddings.",
                         [_rows("Queries", [_row(f"query {n}", q[:80]) for n, q in enumerate(queries)])],
                         [_rows("Rankings", rank_rows)]))

    # 7 RRF
    stages.append(_stage("rrf", "RRF fusion", "RRF fusion",
                         f"Reciprocal-rank fusion: each chunk scores sum(1 / ({RRF_K} + rank)) over the {len(rankings)} rankings. The top {LEX_K} go on to the reranker.",
                         [_text("Input", f"{len(rankings)} rankings")],
                         [_rows("Fused ranking (top 20)", [_row(i, chunks[i]["body"], f"rrf {s:.4f} · in {sum(i in r for r in rankings)} of {len(rankings)} lists")
                                                           for i, s in fused[:20]])]))

    # 8 rerank
    stages.append(_stage("rerank", "Rerank", "Cross-encoder rerank",
                         "bge-reranker-v2-m3 scores each (question, chunk) pair. Then near-copies are dropped: a chunk is kept only if 25 % of its bigrams are new.",
                         [_text("Candidates", f"{len(cand)} chunks")],
                         [_rows("Reranked (top 20 by score)", [_row(i, chunks[i]["body"], f"score {score_of[i]:.3f} · "
                                                              + ("kept" if i in novel else "not kept (near copy or below top-k)"))
                                                             for i in order[:20]])]))

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
    stages.append(_stage("anchor", "Entity anchoring", "Entity anchoring",
                         "If the question names a unit (e.g. 調整池 1), chunks that carry that unit are pulled into the evidence, up to 3 of them.",
                         [_rows("Entities", [_row(f, n) for f, n in wanted] or [_row("-", "no unit number: anchoring skipped")])],
                         anchor_out))

    # 10 fact rows
    rec_facts = (recorded.get("facts") or {}).get("facts")
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
                         fact_out if fact_rows else [_text("Selected rows", "no row matched: prompt B is prompt A")]))

    # 11 prompts A vs B
    compare = {"a": {"label": "A · flag off, no fact rows", "chars": len(system_a) + len(user_a)},
               "b": {"label": "B · fact rows added", "chars": len(system_b) + len(user_b)},
               "diff": {"system": _diff(system_a, system_b), "user": _diff(user_a, user_b)}}
    stages.append(_stage("prompt", "Prompts A vs B", "Messages sent to the LLM",
                         "The exact system and user messages. B is A plus the fact rows and the rule for using them; the diff is highlighted.",
                         [_rows("Evidence", [_row(f"{len(ev)} chunks", _ids(engine, best))]),
                          _rows("Fact rows", [_row(f"{len(fact_rows)} rows", ", ".join(r["id"] for r in fact_rows) or "none")])],
                         [], view="prompt"))

    # 12 answers
    stages.append(_stage("answers", "Answers vs reference", "Answers vs reference",
                         "The answering LLM (Qwen3-4B-Thinking) read prompt A and prompt B. No model runs here: these are the recorded final answers.",
                         [_text("Prompts", "A and B, see the previous step")], [], view="answers"))

    rec_any = recorded.get("facts") or recorded.get("base") or {}
    return {
        "title": f"Pipeline trace · {question_id or 'question'}",
        "meta": {"question_id": question_id, "category": category, "difficulty": difficulty,
                 "llm": rec_any.get("llm") or config.LLM_ID, "top": top, "context": config.CONTEXT_MODE},
        "reference": reference or rec_any.get("reference") or "",
        "stages": stages,
        "compare": compare,
        "answers": {"a": _answer("A · no fact rows", recorded.get("base")), "b": _answer("B · fact rows", recorded.get("facts"))},
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
                   pages_dir=a.out.parent / "trace-pages")
    render(data, a.out)
    print(f"wrote {a.out} ({len(data['stages'])} stages)")


if __name__ == "__main__":
    main()
