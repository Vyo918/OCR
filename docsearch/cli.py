"""docsearch - OCR + context-aware keyword search / QA over scanned drawings.

  docsearch ask -f FILE "question" ...   one step: OCR FILE (cached) -> index -> grounded answer + boxed evidence crops
  docsearch keyword -f FILE TERM         one step: OCR FILE (cached) -> keyword hits with sheet position + crops
  docsearch chat [-f FILE]               interactive; keeps models loaded ("/kw term" = keyword search)
  docsearch build  [PDF|DIR ...]         OCR -> index   (= ingest + index)
  docsearch ingest [PDF|DIR ...]         OCR only       (--limit N tiles for a quick test, --force to redo)
  docsearch index                        (re)build the index from existing OCR output
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
    p = sub.add_parser("chat"); file_opt(p)
    p = sub.add_parser("bench"); p.add_argument("questions", nargs="?", type=Path, default=Path(__file__).with_name("questions.json")); file_opt(p)
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


if __name__ == "__main__":
    main()
