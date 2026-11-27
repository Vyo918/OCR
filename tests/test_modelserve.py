from __future__ import annotations

import os
import tempfile
import threading
import unittest
from multiprocessing.connection import Listener
from pathlib import Path
from unittest.mock import patch

import numpy as np

from docsearch import modelserve
from docsearch.modelserve import RemoteModel, Server, shared
from docsearch.search import Engine


class StubEmbedder:
    def encode(self, sentences, normalize_embeddings=False):
        return np.array([[float(len(s)), float(ord(s[0]))] for s in sentences], dtype=np.float32)


class StubReranker:
    def predict(self, pairs, batch_size=8):
        return np.array([float(len(q) + len(t)) for q, t in pairs], dtype=np.float32)


class ModelServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.address = str(Path(cls.temp.name) / 'models.sock')
        cls.authkey = b'test-key'
        cls.listener = Listener(cls.address, family='AF_UNIX', authkey=cls.authkey)
        server = Server(cls.authkey, loaders={'embedder': StubEmbedder, 'reranker': StubReranker})
        threading.Thread(target=server.serve_forever, args=(cls.listener,), daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.listener.close()  # unlinks the socket file
        cls.temp.cleanup()

    def remote(self, kind):
        return RemoteModel(kind, self.address, self.authkey)

    def test_encode_returns_stub_array(self):
        out = self.remote('embedder').encode(['ab', 'c'], normalize_embeddings=True)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, np.array([[2.0, 97.0], [1.0, 99.0]], dtype=np.float32))

    def test_predict_returns_stub_scores(self):
        out = self.remote('reranker').predict([('q', 'text'), ('question', 'x')], batch_size=8)
        np.testing.assert_array_equal(out, np.array([5.0, 9.0], dtype=np.float32))

    def test_concurrent_clients_get_their_own_results(self):
        results, errors = {}, []

        def call(text):
            try:
                results[text] = self.remote('embedder').encode([text] * 3)
            except Exception as exc:  # reported below
                errors.append(exc)

        threads = [threading.Thread(target=call, args=(t,)) for t in ('alpha', 'beta')]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        for text in ('alpha', 'beta'):
            np.testing.assert_array_equal(results[text], np.array([[len(text), ord(text[0])]] * 3, dtype=np.float32))

    def test_shared_returns_proxy_when_env_points_at_server(self):
        env = {modelserve.SOCKET_ENV: self.address, modelserve.AUTHKEY_ENV: self.authkey.decode()}
        with patch.dict(os.environ, env):
            proxy = shared('reranker')
        self.assertIsInstance(proxy, RemoteModel)
        self.assertEqual(proxy.predict([('q', 't')]).tolist(), [2.0])


class LocalFallbackTests(unittest.TestCase):
    def engine(self):
        engine = Engine.__new__(Engine)
        engine._embedder = engine._reranker = None
        return engine

    def test_unreachable_socket_loads_models_locally(self):
        stub = object()
        with tempfile.TemporaryDirectory() as temp:
            env = {modelserve.SOCKET_ENV: str(Path(temp) / 'missing.sock'), modelserve.AUTHKEY_ENV: 'k'}
            with patch.dict(os.environ, env), patch.object(modelserve, 'CONNECT_WAIT', 0.1), \
                    patch.object(modelserve, 'load_embedder', return_value=stub) as load_embedder, \
                    patch.object(modelserve, 'load_reranker', return_value=stub) as load_reranker:
                engine = self.engine()
                self.assertIs(engine.embedder(), stub)
                self.assertIs(engine.reranker(), stub)
        load_embedder.assert_called_once_with()
        load_reranker.assert_called_once_with()

    def test_no_socket_env_loads_models_locally(self):
        stub = object()
        with patch.dict(os.environ), patch.object(modelserve, 'load_embedder', return_value=stub) as load:
            os.environ.pop(modelserve.SOCKET_ENV, None)
            self.assertIs(self.engine().embedder(), stub)
        load.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
