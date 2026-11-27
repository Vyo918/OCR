import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from docsearch import config
from docsearch import web_jobs
from docsearch.web_jobs import PIPELINE_MODULES, pipeline_fingerprint, run_job

PACKAGE = Path(web_jobs.__file__).resolve().parent


def fake_ingest(files):
    outs = []
    for source in files:
        doc = Path(source).stem
        config.PAGES.mkdir(parents=True, exist_ok=True)
        config.OCR_DIR.mkdir(parents=True, exist_ok=True)
        image = config.PAGES / f'{doc}.png'
        image.write_bytes(Path(source).read_bytes())
        out = config.OCR_DIR / f'{doc}.lines.json'
        out.write_text(json.dumps({'doc': doc, 'image': str(image), 'size': [1, 1],
                                   'lines': [{'doc': doc, 'text': doc, 'bbox': [0, 0, 1, 1]}]}), encoding='utf-8')
        outs.append(out)
    return outs


def fake_build_index(ocr_files, index_dir):
    docs = {}
    for lines in ocr_files:
        data = json.loads(Path(lines).read_text(encoding='utf-8'))
        docs[data['doc']] = {'image': data['image'], 'size': data['size']}
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / 'chunks.json').write_text(json.dumps({'chunks': [], 'outlines': [], 'docs': docs}), encoding='utf-8')
    (index_dir / 'emb.npy').write_bytes(b'emb')


def import_closure(start):
    found, todo = set(), list(start)
    while todo:
        name = todo.pop()
        if name in found or not (PACKAGE / f'{name}.py').exists():
            continue
        found.add(name)
        for node in ast.walk(ast.parse((PACKAGE / f'{name}.py').read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                todo.extend([node.module.split('.')[0]] if node.module else [alias.name for alias in node.names])
    return found


class IndexCacheTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('DOCSEARCH_INDEX_CACHE', None)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.web = Path(temp.name) / 'web'
        self.ingest = Mock(side_effect=fake_ingest)
        self.build = Mock(side_effect=fake_build_index)

    def make_upload(self, root, name, colour):
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        Image.new('RGB', (4, 4), colour).save(path)
        return path

    def run_collection(self, job, uploads):
        with patch.multiple(config, OCR_DIR=job / 'ocr', INDEX_DIR=job / 'index', PAGES=job / 'pages'), \
                patch('docsearch.ocr_ingest.ingest', self.ingest), patch('docsearch.index.build_index', self.build):
            return run_job('collection', job, {'files': [str(p) for p in uploads]})

    def job(self, name):
        return self.web / 'session' / name

    def test_identical_upload_reuses_index_on_new_root(self):
        first = self.job('first')
        self.run_collection(first, [self.make_upload(first / 'uploads', '00.png', 'white')])
        second = self.job('second')
        result = self.run_collection(second, [self.make_upload(second / 'uploads', '00.png', 'white')])

        self.assertEqual(result, {'ready': True, 'reused': True})
        self.assertEqual(self.ingest.call_count, 1)
        self.assertEqual(self.build.call_count, 1)
        chunks = json.loads((second / 'index' / 'chunks.json').read_text(encoding='utf-8'))
        self.assertEqual(chunks['docs']['00']['image'], str(second / 'pages' / '00.png'))
        lines = json.loads((second / 'ocr' / '00.lines.json').read_text(encoding='utf-8'))
        self.assertEqual(lines['image'], str(second / 'pages' / '00.png'))
        self.assertTrue((second / 'index' / 'emb.npy').is_file())
        original = json.loads((first / 'index' / 'chunks.json').read_text(encoding='utf-8'))
        self.assertEqual(original['docs']['00']['image'], str(first / 'pages' / '00.png'))

    def test_changed_content_rebuilds(self):
        first = self.job('first')
        self.run_collection(first, [self.make_upload(first / 'uploads', '00.png', 'white')])
        second = self.job('second')
        result = self.run_collection(second, [self.make_upload(second / 'uploads', '00.png', 'black')])

        self.assertEqual(result, {'ready': True})
        self.assertEqual(self.ingest.call_count, 2)

    def test_changed_name_rebuilds(self):
        first = self.job('first')
        self.run_collection(first, [self.make_upload(first / 'uploads', '00.png', 'white')])
        second = self.job('second')
        self.run_collection(second, [self.make_upload(second / 'uploads', '01.png', 'white')])

        self.assertEqual(self.ingest.call_count, 2)

    def test_incomplete_entry_is_ignored(self):
        first = self.job('first')
        self.run_collection(first, [self.make_upload(first / 'uploads', '00.png', 'white')])
        for entry in (self.web / web_jobs.CACHE_DIR).iterdir():
            (entry / web_jobs.COMPLETE).unlink()
        second = self.job('second')
        self.run_collection(second, [self.make_upload(second / 'uploads', '00.png', 'white')])

        self.assertEqual(self.ingest.call_count, 2)

    def test_disabled_by_environment(self):
        os.environ['DOCSEARCH_INDEX_CACHE'] = '0'
        for name in ('first', 'second'):
            job = self.job(name)
            self.run_collection(job, [self.make_upload(job / 'uploads', '00.png', 'white')])

        self.assertEqual(self.ingest.call_count, 2)
        self.assertFalse((self.web / web_jobs.CACHE_DIR).exists())

    def test_fingerprint_follows_pipeline_settings_not_work_dir(self):
        before = pipeline_fingerprint()
        with patch.dict(os.environ, {'DOCSEARCH_WORK': '/elsewhere'}):
            self.assertEqual(pipeline_fingerprint(), before)
        with patch.dict(os.environ, {'DOCSEARCH_TOP': '99'}):
            self.assertNotEqual(pipeline_fingerprint(), before)

    def test_pipeline_modules_are_the_import_closure(self):
        self.assertEqual(set(PIPELINE_MODULES), import_closure(['ocr_ingest', 'index']))


if __name__ == '__main__':
    unittest.main()
