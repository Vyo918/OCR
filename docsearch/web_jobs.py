"""Private subprocess worker for web uploads and questions."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def run_job(kind: str, root: Path, payload: dict) -> dict:
    """Run the normal pipeline using only this collection's workspace."""
    os.environ['DOCSEARCH_WORK'] = str(root)
    if kind == 'collection':
        from . import config
        from .index import build_index
        from .ocr_ingest import ingest
        files = [Path(p) for p in payload['files']]
        for path in files:
            if path.suffix.lower() == '.pdf':
                info = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True, timeout=30, check=True).stdout
                pages = re.search(r'^Pages:\s+(\d+)', info, re.M)
                if not pages or int(pages.group(1)) > 20 or 'Encrypted: yes' in info:
                    raise RuntimeError('PDF is encrypted or exceeds 20 pages')
        outs = ingest(files)
        if not outs:
            raise RuntimeError('No readable pages found')
        build_index(sorted(config.OCR_DIR.glob('*.lines.json')), config.INDEX_DIR)
        return {'ready': True}
    if kind == 'question':
        from .search import Engine
        from . import config
        if not (config.INDEX_DIR / 'chunks.json').exists():
            raise RuntimeError('Collection index is unavailable')
        eng = Engine()
        result = eng.answer(payload['question'])
        output = root / 'web' / payload['session'] / payload['collection'] / 'questions' / payload['job']
        output.mkdir(parents=True, exist_ok=True)
        paths = eng.locate(result, output)
        return {'answer': result['answer'], 'queries': result['queries'], 'evidence': result['evidence'],
                'images': [str(p) for p in paths]}
    raise ValueError('Unknown job type')


def child_main():
    kind, root, payload = sys.argv[1:4]
    try:
        result = run_job(kind, Path(root), json.loads(payload))
        print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)[:300]}))
        raise SystemExit(1)


if __name__ == '__main__':
    child_main()
