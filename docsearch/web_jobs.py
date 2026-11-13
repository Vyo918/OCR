"""Isolated subprocess entry point for local web jobs."""
from __future__ import annotations

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
    if encrypted or not pages or int(pages.group(1)) > 20:
        raise ValueError('pdf')
    page_sizes = re.findall(r'^Page\s+\d+\s+size:\s*([\d.]+)\s+x\s+([\d.]+)\s+pts', info, re.M)
    if len(page_sizes) != int(pages.group(1)) or any(float(w) * 300 / 72 * float(h) * 300 / 72 > MAX_PIXELS for w, h in page_sizes):
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


def run_job(kind: str, root: Path, payload: dict) -> dict:
    """Run the unchanged pipeline using only this collection's process environment."""
    os.environ['DOCSEARCH_WORK'] = str(root)
    if kind == 'collection':
        files = [Path(p) for p in payload['files']]
        for path in files:
            _validate_source(path)
        from . import config
        from .index import build_index
        from .ocr_ingest import ingest
        outs = ingest(files)
        if not outs:
            raise ValueError('pipeline')
        build_index(sorted(config.OCR_DIR.glob('*.lines.json')), config.INDEX_DIR)
        return {'ready': True}
    if kind == 'question':
        from .search import Engine
        from . import config
        if not (config.INDEX_DIR / 'chunks.json').exists():
            raise ValueError('pipeline')
        eng = Engine()
        result = eng.answer(payload['question'])
        output = root / 'questions' / payload['job'] / 'images'
        output.mkdir(parents=True, exist_ok=True)
        paths = eng.locate(result, output)
        return {'answer': result['answer'], 'queries': result['queries'], 'evidence': result['evidence'],
                'images': [str(p) for p in paths]}
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
