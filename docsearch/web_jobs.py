"""Isolated subprocess entry point for local web jobs."""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

MAX_PIXELS = 100_000_000


def preflight_pdf(path: Path) -> None:
    """Reject expensive or unsafe PDFs before importing OCR/model code."""
    try:
        info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('pdf') from exc
    encrypted = re.search(r'^Encrypted:\s*yes\b', info, re.M | re.I)
    pages = re.search(r'^Pages:\s*(\d+)\s*$', info, re.M)
    if encrypted or not pages or not 1 <= int(pages.group(1)) <= 20:
        raise ValueError('pdf')
    page_count = int(pages.group(1))
    try:
        info = subprocess.run(
            ['pdfinfo', '-f', '1', '-l', str(page_count), str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('pdf') from exc
    page_sizes = re.findall(r'^Page\s+\d+\s+size:\s*([\d.]+)\s+x\s+([\d.]+)\s+pts', info, re.M)
    if (len(page_sizes) != page_count or
            any(float(w) * 300 / 72 * float(h) * 300 / 72 > MAX_PIXELS for w, h in page_sizes)):
        raise ValueError('pdf')
    try:
        listing = subprocess.run(['pdfimages', '-list', str(path)], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('pdf') from exc
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 13 and fields[0].isdigit() and fields[1].isdigit():
            try:
                width, height = int(fields[3]), int(fields[4])
            except ValueError:
                continue
            if width * height > MAX_PIXELS:
                raise ValueError('pdf')


def _validate_source(path: Path) -> None:
    if path.suffix.lower() == '.pdf':
        preflight_pdf(path)
        return
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError('image')
        image.load()


def _crop_evidence(result: dict, paths: list[Path]) -> list[dict]:
    """Pair each crop with its evidence: locate() names the n-th cited hit on a sheet '<doc>.hit<n>.png', citing as it does."""
    cited = [e for e in result['evidence'] if re.search(rf"#{re.escape(e['id'].rsplit('#', 1)[-1])}(?!\d)", result['answer'])] or result['evidence']
    hits = cited[:4]
    ids = {}
    for doc in {e['doc'] for e in hits}:
        for n, e in enumerate((e for e in hits if e['doc'] == doc), 1):
            ids[f"{doc[:30]}.hit{n}.png"] = e['id']
    return [{'path': str(p), 'evidence_id': ids.get(p.name)} for p in paths]


class RetrievalLock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = open(self.path, 'a')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX)
        except BaseException:
            self.file.close()
            raise
        return self

    def __exit__(self, *args):
        try:
            fcntl.flock(self.file, fcntl.LOCK_UN)
        finally:
            self.file.close()


def run_job(kind: str, root: Path, payload: dict) -> dict:
    """Run the unchanged pipeline using only this collection's process environment."""
    os.environ['DOCSEARCH_WORK'] = str(root)
    from . import progress
    if kind == 'collection':
        files = [Path(p) for p in payload['files']]
        for path in files:
            progress.report('prepare')
            _validate_source(path)
        progress.report('ocr')
        from . import config
        from .index import build_index
        from .ocr_ingest import ingest
        outs = ingest(files)
        if not outs:
            raise ValueError('pipeline')
        progress.report('index')
        build_index(sorted(config.OCR_DIR.glob('*.lines.json')), config.INDEX_DIR)
        progress.report('index', 1, 1, 'chunks')
        return {'ready': True}
    if kind == 'question':
        from .search import Engine
        from . import config
        if not (config.INDEX_DIR / 'chunks.json').exists():
            raise ValueError('pipeline')
        from . import progress
        progress.report('retrieve')
        eng = Engine()
        lock_path = os.environ.get('DOCSEARCH_RETRIEVAL_LOCK')
        result = eng.answer(payload['question'], lock=RetrievalLock(lock_path) if lock_path else None)
        output = root / 'questions' / payload['job'] / 'images'
        output.mkdir(parents=True, exist_ok=True)
        progress.report('answer')
        return ({'answer': result['answer'], 'queries': result['queries'], 'evidence': result['evidence'],
                 'images': _crop_evidence(result, eng.locate(result, output))} |
                {key: result[key] for key in ('facts', 'computed') if key in result})
    raise ValueError('pipeline')


def child_main():
    kind, root, payload = sys.argv[1:4]
    try:
        result = run_job(kind, Path(root), json.loads(payload))
        print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))
    except Exception as exc:
        category = str(exc) if str(exc) in {'pdf', 'image', 'pipeline'} else 'pipeline'
        print(json.dumps({'ok': False, 'category': category, 'diagnostic': traceback.format_exc()}), flush=True)
        raise SystemExit(1)
    except SystemExit:
        print(json.dumps({'ok': False, 'category': 'pipeline', 'diagnostic': traceback.format_exc()}), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    child_main()
