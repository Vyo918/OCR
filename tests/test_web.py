from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from docsearch.web import State, create_app


class BodyCapTests(unittest.TestCase):
    def test_streamed_request_body_is_capped_without_content_length(self):
        with tempfile.TemporaryDirectory() as temp, patch('docsearch.web.BODY_LIMIT', 128):
            app = create_app(Path(temp))
            with TestClient(app, base_url='http://127.0.0.1') as client:
                csrf = client.get('/api/session').json()['csrf_token']
                def chunks():
                    yield b'x' * 64
                    yield b'x' * 65
                response = client.post(
                    '/api/collections', content=chunks(),
                    headers={'X-CSRF-Token': csrf, 'Origin': 'http://127.0.0.1',
                             'Content-Type': 'multipart/form-data; boundary=small'},
                )
        self.assertEqual(response.status_code, 413)

def tiny_png():
    import io
    from PIL import Image
    out = io.BytesIO()
    Image.new('RGB', (2, 2), 'white').save(out, format='PNG')
    return out.getvalue()

def test_child_command(job):
    script = """import base64,json,pathlib,sys
kind,root,payload=sys.argv[1],pathlib.Path(sys.argv[2]),json.loads(sys.argv[3])
if kind == 'collection':
 result={'ready':True}
else:
 out=root/'questions'/payload['job']/'images'/'crop.png'; out.parent.mkdir(parents=True,exist_ok=True)
 out.write_bytes(base64.b64decode(sys.argv[4]))
 result={'answer':'grounded\u2028answer','queries':['q'],'evidence':[{'id':'00#1','doc':'00','bbox':[0,0,1,1],'text':'text'}],'images':[str(out)]}
print(json.dumps({'ok':True,'result':result}, ensure_ascii=False))"""
    return [sys.executable, '-c', script, job['kind'], str(job['root']), json.dumps(job['payload']), base64.b64encode(tiny_png()).decode()]


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.temp.name), process_args=test_child_command), base_url='http://127.0.0.1')
        self.client.__enter__()
        self.csrf = self.client.get('/api/session').json()['csrf_token']
        self.headers = {'X-CSRF-Token': self.csrf, 'Origin': 'http://127.0.0.1'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_bootstrap_security_headers_and_loopback_authority(self):
        response = self.client.get('/api/session')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
        self.assertIn('default-src', response.headers['content-security-policy'])
        for host in ('evil.example', '127.0.0.1:abc', '127.0.0.1:0'):
            self.assertEqual(self.client.get('/api/session', headers={'Host': host}).status_code, 400)
        for origin in ('http://evil.example', 'not-an-origin', 'http://127.0.0.1:8001'):
            self.assertEqual(self.client.post('/api/collections', headers={**self.headers, 'Origin': origin}).status_code, 403)

    def test_valid_session_uploads_separate_batches_and_rejects_cross_session(self):
        first = self.client.post('/api/collections', files=[('files', ('drawing.png', tiny_png(), 'image/png'))], headers=self.headers)
        second = self.client.post('/api/collections', files=[('files', ('drawing.png', tiny_png(), 'image/png'))], headers=self.headers)
        self.assertEqual((first.status_code, second.status_code), (202, 202))
        state = self.client.app.state.web
        sid = self.client.cookies['docsearch_session']
        with state.condition:
            self.assertTrue(state.condition.wait_for(lambda: all(state.sessions[sid]['collections'][item.json()['id']]['record']['status'] in {'ready', 'failed'} for item in (first, second)), timeout=10))
        self.assertNotEqual(first.json()['id'], second.json()['id'])
        self.assertEqual(first.json()['files'][0]['name'], 'drawing.png')
        self.assertEqual(self.client.post('/api/collections', files=[('files', ('bad.txt', b'x', 'text/plain'))], headers=self.headers).status_code, 422)
        other = TestClient(self.client.app, base_url='http://127.0.0.1')
        with other:
            self.assertEqual(other.get('/api/session').status_code, 200)
            self.assertEqual(other.get('/api/collections/' + first.json()['id']).status_code, 404)

    def test_unready_collection_question_is_conflict_with_valid_csrf(self):
        state = self.client.app.state.web
        sid, info = next(iter(state.sessions.items()))
        cid = 'existing'
        info['collections'][cid] = {'record': {'status': 'queued'}, 'root': Path(self.temp.name), 'questions': {}, 'display': {}}
        response = self.client.post(f'/api/collections/{cid}/questions', json={'question': 'why'}, headers=self.headers)
        self.assertEqual(response.status_code, 409)

    def test_multipart_validation_cleans_partial_storage_and_requires_csrf(self):
        denied = self.client.post('/api/collections', files=[('files', ('x.png', tiny_png(), 'image/png'))])
        self.assertEqual(denied.status_code, 403)
        rejected = self.client.post('/api/collections', files=[('files', ('first.png', tiny_png(), 'image/png')), ('files', ('bad.pdf', b'not pdf', 'application/pdf'))], headers=self.headers)
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(list((Path(self.temp.name) / 'web').rglob('uploads')), [])

    def test_same_origin_question_image_lifecycle_and_job_containment(self):
        state = self.client.app.state.web
        sid, info = next(iter(state.sessions.items()))
        cid = 'ready-col'
        root = Path(self.temp.name) / 'web' / sid / cid
        root.mkdir(parents=True)
        record = {'id': cid, 'files': [{'name': 'original.pdf', 'size': 1}], 'status': 'ready', 'stage': 'ready', 'error': None}
        info['collections'][cid] = {'record': record, 'root': root, 'questions': {}, 'display': {'00': 'original.pdf'}}
        created = self.client.post(f'/api/collections/{cid}/questions', json={'question': 'why?'}, headers=self.headers)
        self.assertEqual(created.status_code, 202)
        jid = created.json()['id']
        with state.condition:
            self.assertTrue(state.condition.wait_for(lambda: info['collections'][cid]['questions'][jid]['status'] in {'ready', 'failed'}, timeout=10))
        job = self.client.get(f'/api/collections/{cid}/questions/{jid}').json()
        self.assertEqual(job['status'], 'ready')
        self.assertEqual(job['result']['evidence'][0]['doc'], 'original.pdf')
        self.assertNotIn('image_map', job)
        image_url = job['result']['images'][0]['url']
        self.assertEqual(job['result']['answer'], 'grounded\u2028answer')
        self.assertEqual(self.client.get(image_url).content, tiny_png())
        outside = Path(self.temp.name) / 'outside.png'
        outside.write_bytes(tiny_png())
        escaped = root / 'questions' / jid / 'images' / 'escape.png'
        escaped.symlink_to(outside)
        image_id = image_url.rsplit('/', 1)[-1]
        other = TestClient(self.client.app, base_url='http://127.0.0.1')
        with other:
            other.get('/api/session')
            self.assertEqual(other.get(image_url).status_code, 404)
        state.sessions[sid]['collections'][cid]['questions'][jid]['image_map'][image_id] = escaped
        self.assertEqual(self.client.get(image_url).status_code, 404)

    def test_terminal_error_uses_fixed_public_message(self):
        def failing_child(job):
            script = "import sys; print(sys.argv[1]); raise SystemExit(7)"
            return [sys.executable, '-c', script, str(Path(self.temp.name) / 'private-secret')]
        app = create_app(Path(self.temp.name), process_args=failing_child)
        with TestClient(app, base_url='http://127.0.0.1') as client:
            csrf = client.get('/api/session').json()['csrf_token']
            headers = {'X-CSRF-Token': csrf, 'Origin': 'http://127.0.0.1'}
            response = client.post('/api/collections', files=[('files', ('drawing.png', tiny_png(), 'image/png'))], headers=headers)
            self.assertEqual(response.status_code, 202)
            sid = client.cookies['docsearch_session']
            state = app.state.web
            cid = response.json()['id']
            with state.condition:
                self.assertTrue(state.condition.wait_for(
                    lambda: state.sessions[sid]['collections'][cid]['record']['status'] == 'failed', timeout=10))
            record = client.get(f'/api/collections/{cid}').json()
            self.assertEqual(record['error'], 'Processing failed. Check the local server log.')
            self.assertNotIn(str(self.temp.name), json.dumps(record))
            self.assertIn(str(self.temp.name), (Path(self.temp.name) / 'web' / sid / cid / 'job.log').read_text())
    def test_shutdown_marks_queued_jobs_terminal(self):
        state = State(Path(self.temp.name) / 'other')
        record = {'status': 'queued'}
        job = {'kind': 'collection', 'root': state.root, 'payload': {},
               'start': lambda: record.update(status='running'), 'finish': lambda _: None,
               'fail': lambda category, diagnostic: record.update(status='failed')}
        state.enqueue(job)
        state.close()
        self.assertEqual(record['status'], 'failed')


class AllowedHostTests(unittest.TestCase):
    tailnet = '100.104.213.53'

    def client(self, temp, **kwargs):
        return TestClient(create_app(Path(temp), process_args=test_child_command, **kwargs), base_url='http://' + self.tailnet + ':8000')

    def test_allowed_host_serves_session_csrf_and_upload_flow(self):
        with tempfile.TemporaryDirectory() as temp, self.client(temp, allowed_hosts=[self.tailnet]) as client:
            session = client.get('/api/session', headers={'Host': self.tailnet + ':8000'})
            self.assertEqual(session.status_code, 200)
            headers = {'X-CSRF-Token': session.json()['csrf_token'], 'Origin': 'http://' + self.tailnet + ':8000', 'Host': self.tailnet + ':8000'}
            upload = client.post('/api/collections', files=[('files', ('drawing.png', tiny_png(), 'image/png'))], headers=headers)
            self.assertEqual(upload.status_code, 202)

    def test_tailnet_host_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp, self.client(temp) as client:
            self.assertEqual(client.get('/api/session', headers={'Host': self.tailnet + ':8000'}).status_code, 400)

    def test_allowlist_does_not_admit_other_hosts_or_cross_origins(self):
        with tempfile.TemporaryDirectory() as temp, self.client(temp, allowed_hosts=[self.tailnet]) as client:
            self.assertEqual(client.get('/api/session', headers={'Host': 'evil.example'}).status_code, 400)
            self.assertEqual(client.get('/api/session', headers={'Host': self.tailnet + ':8000'}).status_code, 200)
            cross = client.post('/api/collections', headers={'Origin': 'http://127.0.0.1:8000', 'Host': self.tailnet + ':8000'})
            self.assertEqual(cross.status_code, 403)

    def test_invalid_allowed_host_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            for value in ('*', '100.104.213.53:8000', 'http://x', ''):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    create_app(Path(temp), allowed_hosts=[value])

    def test_env_var_is_parsed_and_keyword_wins(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'DOCSEARCH_ALLOWED_HOSTS': ' 100.104.213.53 ,, Tailnet.Example '}):
            with self.client(temp) as client:
                self.assertEqual(client.get('/api/session', headers={'Host': 'tailnet.example:8000'}).status_code, 200)
            with TestClient(create_app(Path(temp), allowed_hosts=[]), base_url='http://' + self.tailnet + ':8000') as client:
                self.assertEqual(client.get('/api/session').status_code, 400)


if __name__ == '__main__':
    unittest.main()
