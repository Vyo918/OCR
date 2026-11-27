"""Isolated subprocess entry point for local web jobs."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
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


# Modules that write pages/, ocr/ and index/: the import closure of ocr_ingest and index (tests/test_index_cache.py checks it).
PIPELINE_MODULES = ('calc', 'cascade', 'config', 'device', 'engines', 'facts', 'facts_calc', 'facts_geometry',
                    'facts_labels', 'facts_tables', 'index', 'ocr_ingest', 'progress', 'reread', 'table_reread', 'timing')
# Per-job locations: they change where files go, not what the pipeline writes, so they stay out of the key.
UNKEYED_SETTINGS = frozenset({'DOCSEARCH_WORK', 'DOCSEARCH_PROGRESS_FILE'})
CACHE_DIR = '_cache'
ARTEFACTS = ('pages', 'ocr', 'index')
COMPLETE = 'complete'


def pipeline_fingerprint() -> str:
    """sha256 of the PIPELINE_MODULES sources and of the DOCSEARCH_* settings they name, at their current values."""
    package = Path(__file__).resolve().parent
    digest, names = hashlib.sha256(), set()
    for module in PIPELINE_MODULES:
        source = (package / f'{module}.py').read_bytes()
        digest.update(module.encode() + b'\0' + source + b'\0')
        names.update(name.decode() for name in re.findall(rb'DOCSEARCH_[A-Z0-9_]+', source))
    for name in sorted(names - UNKEYED_SETTINGS):
        digest.update(f'{name}={os.environ.get(name)}\n'.encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_entry(root: Path, files: list[Path]) -> Path | None:
    """Cache directory for these uploads under this pipeline; None when DOCSEARCH_INDEX_CACHE=0 or they cannot be hashed.
    Doc names come from file names (uploads/00.pdf -> doc '00'), so names, extensions, order and content are all keyed."""
    if os.environ.get('DOCSEARCH_INDEX_CACHE') == '0':
        return None
    try:
        digest = hashlib.sha256(f'index-cache-v1\n{pipeline_fingerprint()}\n'.encode())
        for path in files:
            digest.update(f'{path.name}\0{_file_sha256(path)}\n'.encode())
    except OSError as exc:
        print(f'[index cache] disabled for this job: {exc}', file=sys.stderr, flush=True)
        return None
    return root.parent.parent / CACHE_DIR / digest.hexdigest()


def _link_or_copy(src, dst):
    """Artefacts are written once and only ever replaced, never rewritten in place, so sharing them by hardlink is safe."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rebase(path: str, old_root: str, new_root: str) -> str:
    """A path recorded under the building root is re-pointed at the same file under new_root; other paths are kept."""
    absolute = os.path.abspath(path)
    if os.path.commonpath([absolute, old_root]) != old_root:
        return path
    return os.path.join(new_root, os.path.relpath(absolute, old_root))


def _rebase_json(path: Path, edit) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    edit(data)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)   # a new inode: the hardlinked original in the cache is left alone


def _reuse_index(entry: Path, root: Path) -> bool:
    """Copy a complete cache entry into root with its image paths pointing at root. False when there is no usable entry;
    root is then left without artefacts."""
    marker = entry / COMPLETE
    if not marker.is_file():
        return False
    try:
        old_root = json.loads(marker.read_text(encoding='utf-8'))
        new_root = os.path.abspath(root)
        for name in ARTEFACTS:
            if (entry / name).exists():
                shutil.copytree(entry / name, root / name, copy_function=_link_or_copy)

        def rebase_ocr(data):
            data['image'] = _rebase(data['image'], old_root, new_root)

        def rebase_chunks(data):
            for doc in data['docs'].values():
                doc['image'] = _rebase(doc['image'], old_root, new_root)

        for lines in (root / 'ocr').glob('*.lines.json'):
            _rebase_json(lines, rebase_ocr)
        _rebase_json(root / 'index' / 'chunks.json', rebase_chunks)
        return True
    except Exception as exc:
        print(f'[index cache] not reused, rebuilding: {exc}', file=sys.stderr, flush=True)
        for name in ARTEFACTS:
            shutil.rmtree(root / name, ignore_errors=True)
        return False


def _store_index(root: Path, entry: Path) -> None:
    """Publish root's artefacts as entry: copied under a temporary name, then renamed, so readers see a complete entry or none."""
    if entry.exists():
        return
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.parent / f'.tmp-{secrets.token_hex(8)}'
    try:
        tmp.mkdir()
        for name in ARTEFACTS:
            if (root / name).exists():
                shutil.copytree(root / name, tmp / name, copy_function=_link_or_copy)
        (tmp / COMPLETE).write_text(json.dumps(os.path.abspath(root)), encoding='utf-8')
        try:
            os.replace(tmp, entry)
        except OSError:
            if not (entry / COMPLETE).is_file():   # not lost to a concurrent job's complete entry: a real failure
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



def run_job(kind: str, root: Path, payload: dict) -> dict:
    """Run the unchanged pipeline using only this collection's process environment."""
    os.environ['DOCSEARCH_WORK'] = str(root)
    from . import progress
    if kind == 'collection':
        files = [Path(p) for p in payload['files']]
        for path in files:
            progress.report('prepare')
            _validate_source(path)
        from . import config
        entry = _cache_entry(root, files)
        if entry and _reuse_index(entry, root):
            progress.report('index', 1, 1, 'chunks')
            return {'ready': True, 'reused': True}
        progress.report('ocr')
        from .index import build_index
        from .ocr_ingest import ingest
        outs = ingest(files)
        if not outs:
            raise ValueError('pipeline')
        progress.report('index')
        build_index(sorted(config.OCR_DIR.glob('*.lines.json')), config.INDEX_DIR)
        progress.report('index', 1, 1, 'chunks')
        if entry:
            try:
                _store_index(root, entry)
            except Exception as exc:
                print(f'[index cache] not stored: {exc}', file=sys.stderr, flush=True)
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
