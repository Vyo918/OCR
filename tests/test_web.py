import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from docsearch.web import create_app


class WebTests(unittest.TestCase):
    def test_private_session_and_upload_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(create_app(workspace=Path(tmp))) as client:
                bootstrap = client.get('/api/session')
                self.assertEqual(bootstrap.status_code, 200)
                csrf = bootstrap.json()['csrf_token']
                headers = {'X-CSRF-Token': csrf, 'Origin': 'http://testserver'}
                bad = client.post('/api/collections', files=[('files', ('empty.pdf', b'', 'application/pdf'))], headers=headers)
                self.assertEqual(bad.status_code, 422)
                good = client.post('/api/collections', files=[('files', ('same.pdf', b'%PDF-1.4 fake', 'application/pdf'))], headers=headers)
                self.assertEqual(good.status_code, 202)
                self.assertEqual(good.json()['status'], 'queued')
                other = TestClient(create_app(workspace=Path(tmp)))
                with other:
                    self.assertEqual(other.get('/api/collections/' + good.json()['id']).status_code, 404)

    def test_hostile_origin_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(workspace=Path(tmp))) as client:
            client.get('/api/session')
            response = client.post('/api/collections', headers={'Origin': 'http://evil.example', 'X-CSRF-Token': 'x'})
            self.assertEqual(response.status_code, 403)

    def test_unready_question_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(workspace=Path(tmp))) as client:
            csrf = client.get('/api/session').json()['csrf_token']
            response = client.post('/api/collections/missing/questions', json={'question':'q'}, headers={'X-CSRF-Token':csrf})
            self.assertEqual(response.status_code, 404)
