"""docsearch - OCR + context-aware keyword search / QA over scanned drawings.

  docsearch ask -f FILE "question" ...   one step: OCR FILE (cached) -> index -> grounded answer + boxed evidence crops
  docsearch keyword -f FILE TERM         one step: OCR FILE (cached) -> keyword hits with sheet position + crops
  docsearch chat [-f FILE]               interactive; keeps models loaded ("/kw term" = keyword search)
  docsearch build  [PDF|DIR ...]         OCR -> index   (= ingest + index)
  docsearch ingest [PDF|DIR ...]         OCR only       (--limit N tiles for a quick test, --force to redo)
  docsearch index                        (re)build the index from existing OCR output
  docsearch bench [questions.json]       run the reference questions and compare with the human answers
  docsearch serve [--host HOST] [--port PORT] start the local browser interface
  docsearch prefetch                     download model weights (once, into $HF_HOME)

Without -f, ask / keyword / chat use whatever `build` indexed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import config


def _prepare(files: list[Path] | None) -> set[str] | None:
    """-f FILE: OCR the file(s) (cached per page + engine), (re)index when needed, return the document names to search."""
    if not files:
        return None
    from .index import build_index
    from .ocr_ingest import ingest

    outs = ingest(files)
    names = {json.loads(o.read_text())["doc"] for o in outs}
    idx = config.INDEX_DIR / "chunks.json"
    stale = (not idx.exists() or any(o.stat().st_mtime > idx.stat().st_mtime for o in outs)
             or not names <= set(json.loads(idx.read_text())["docs"]))
    if stale:
        build_index(sorted(config.OCR_DIR.glob("*.lines.json")), config.INDEX_DIR)
    return names


def _engine(only: set[str] | None = None):
    from .search import Engine
    if not (config.INDEX_DIR / "chunks.json").exists():
        sys.exit(f"no index at {config.INDEX_DIR}; run `docsearch build` (or pass -f FILE) first")
    return Engine(only=only)


def _print_hits(eng, hits, term):
    for n, h in enumerate(hits[:25], 1):
        print(f"[{n}] score={h['score']} {h['doc'][:40]} @ {h['bbox']} {h['section']}\n    {h['body'].replace(chr(10), ' | ')}")
    if hits:
        print("crops:", *eng.annotate(hits[:25], config.WORK / "hits" / re.sub(r"\W+", "_", term)), sep="\n  ")
    else:
        print(f"no hit for {term!r}")


def _print_answer(eng, r):
    print(f"\n### {r['question']}\nsearch terms: {r['queries']}\n\n{r['answer']}\n")
    if r.get("facts"):
        for line in r.get("computed", []):
            print(line)
        for row in r["facts"]:
            print(json.dumps(row, ensure_ascii=False))
    print("evidence on the sheets:", *eng.locate(r, config.WORK / "answers" / re.sub(r"\W+", "_", r["question"])[:40]), sep="\n  ")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docsearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    file_opt = lambda p: p.add_argument("-f", "--file", action="append", type=Path, metavar="FILE", help="PDF / image / folder to read first (repeatable)")
    p = sub.add_parser("prefetch"); p.add_argument("--only", nargs="*", choices=["ocr", *config.MODELS])
    for name in ("ingest", "build"):
        p = sub.add_parser(name); p.add_argument("paths", nargs="*", type=Path); p.add_argument("--limit", type=int); p.add_argument("--force", action="store_true")
    sub.add_parser("index")
    p = sub.add_parser("keyword"); p.add_argument("term"); file_opt(p)
    p = sub.add_parser("ask"); p.add_argument("question", nargs="+"); p.add_argument("--top", type=int, default=12); p.add_argument("--json", type=Path); file_opt(p)
    p = sub.add_parser("bench"); p.add_argument("questions", nargs="?", type=Path, default=Path(__file__).with_name("questions.json")); file_opt(p)
    p = sub.add_parser("chat"); file_opt(p)
    p = sub.add_parser("benchjson"); p.add_argument("questions", type=Path); p.add_argument("--subset", type=int); p.add_argument("--seed", type=int, default=0); p.add_argument("--ids"); p.add_argument("--out", type=Path); p.add_argument("--resume", action="store_true"); p.add_argument("--workers", type=int, default=1, help="concurrent questions (http backend only); per-question seconds include waiting when workers > 1")
    p = sub.add_parser("facts"); file_opt(p); p.add_argument("--tsv", action="store_true"); p.add_argument("--ask", nargs="+"); p.add_argument("--questions", type=Path); p.add_argument("--ids")
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("docsearch.web:create_app", factory=True, host=a.host, port=a.port)
        return
    elif a.cmd == "prefetch":
        from .models import prefetch
        prefetch(a.only)
    elif a.cmd in ("ingest", "build"):
        from .ocr_ingest import ingest
        ingest(a.paths, a.limit, a.force)
        if a.cmd == "build":
            from .index import build_index
            build_index(sorted(config.OCR_DIR.glob("*.lines.json")), config.INDEX_DIR)
    elif a.cmd == "index":
        from .index import build_index
        build_index(sorted(config.OCR_DIR.glob("*.lines.json")), config.INDEX_DIR)
    elif a.cmd == "facts":
        from . import facts, factquery
        only = _prepare(a.file) if a.file else None
        data = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        chunks = [c for c in data["chunks"] if only is None or c["doc"] in only]
        rows = facts.extract(chunks, data["docs"])
        if a.questions:
            questions = json.loads(a.questions.read_text(encoding="utf-8"))["題目"]
            wanted = set(a.ids.split(",")) if a.ids else None
            for q in questions:
                if wanted is None or q.get("編號") in wanted:
                    text = q["問題"]
                    fx = factquery.context(text, rows)
                    print(f"{q.get('編號', '')}\t{text}\t{factquery.parse(text).intent}\t{len(fx['rows']) if fx else 0}")
                    for line in fx["computed"] if fx else []:
                        print(line)
        elif a.ask:
            for question in a.ask:
                fx = factquery.context(question, rows)
                print(fx["text"] if fx else "No matching facts.")
        elif a.tsv:
            import csv
            fields = ("id", "doc", "source", "section", "entity", "attr", "label", "value", "unit", "text")
            writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
            writer.writerow(fields)
            writer.writerows([row.get(k, "") for k in fields] for row in rows)
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif a.cmd == "keyword":
        eng = _engine(_prepare(a.file))
        _print_hits(eng, eng.keyword(a.term), a.term)
    elif a.cmd == "ask":
        eng = _engine(_prepare(a.file))
        res = []
        for q in a.question:
            r = eng.answer(q, a.top)
            res.append(r)
            _print_answer(eng, r)
        if a.json:
            a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    elif a.cmd == "chat":
        eng = _engine(_prepare(a.file))
        print("ask a question, '/kw TERM' for keyword search, '/quit' to leave")
        for line in sys.stdin:
            line = line.strip()
            if line in ("/quit", "/q"):
                break
            if line.startswith("/kw "):
                _print_hits(eng, eng.keyword(line[4:].strip()), line[4:].strip())
            elif line:
                _print_answer(eng, eng.answer(line))
    elif a.cmd == "bench":
        from .bench import run
        run(a.questions, _engine(_prepare(a.file)))
    elif a.cmd == "benchjson":
        from .benchjson import run
        if a.workers < 1:
            ap.error("--workers must be at least 1")
        if a.workers > 1 and config.LLM_BACKEND != "http":
            ap.error("--workers > 1 requires DOCSEARCH_LLM_BACKEND=http")
        run(a.questions, _engine(_prepare(None)), out=a.out, subset=a.subset, seed=a.seed,
            ids=a.ids.split(",") if a.ids else None, resume=a.resume, workers=a.workers)


if __name__ == "__main__":
    main()
