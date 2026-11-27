"""Shared retrieval models for the web service.

One long-lived process loads the embedder (BGE-M3) and the reranker once and answers question jobs over a local
Unix socket, so each question subprocess no longer reloads them from disk.

  python -m docsearch.modelserve <socket>

The authkey comes in DOCSEARCH_MODEL_AUTHKEY. The parent keeps the server's stdin open: EOF means the parent closed
it or died, and the server exits. The server holds models only; chunks and indexes stay in each question subprocess.
A question subprocess finds the server through DOCSEARCH_MODEL_SOCKET and DOCSEARCH_MODEL_AUTHKEY (see shared()); without
them, or when the socket is unreachable, it loads the models itself.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener

import numpy as np

from . import config
from .device import device, retrieval_placement

SOCKET_ENV = "DOCSEARCH_MODEL_SOCKET"
AUTHKEY_ENV = "DOCSEARCH_MODEL_AUTHKEY"
# ponytail: a client retries a missing or refusing socket for this long (the server may still be starting), no readiness handshake
CONNECT_WAIT = 5.0
OPS = frozenset({"encode", "predict"})
log = logging.getLogger(__name__)
_warned = False


def load_embedder():
    """Same construction as the local path: placement and dtype from retrieval_placement()."""
    from sentence_transformers import SentenceTransformer
    build, half = retrieval_placement(device())
    model = SentenceTransformer(config.EMBED_ID, device=build)
    if half:
        model.half()
    return model


def load_reranker():
    import torch
    from sentence_transformers import CrossEncoder
    build, half = retrieval_placement(device())
    return CrossEncoder(config.RERANK_ID, device=build, model_kwargs={"dtype": torch.float16 if half else torch.float32})


LOADERS = {"embedder": load_embedder, "reranker": load_reranker}


class RemoteModel:
    """Stands in for a SentenceTransformer or CrossEncoder with the methods search.py calls, run by the server.
    `model` and `to()` make Engine.on_gpu a no-op: the server owns placement."""

    def __init__(self, kind: str, address: str, authkey: bytes):
        self.kind, self.address, self.authkey = kind, address, authkey

    @property
    def model(self):
        return self

    def to(self, *_):
        return self

    def encode(self, sentences, **kwargs):
        return self._call("encode", sentences, **kwargs)

    def predict(self, sentences, **kwargs):
        return self._call("predict", sentences, **kwargs)

    def _call(self, op: str, *args, **kwargs):
        with _connect(self.address, self.authkey) as conn:
            conn.send((self.kind, op, args, kwargs))
            status, value = conn.recv()
        if status != "ok":
            raise RuntimeError(f"model server {op} on {self.kind} failed:\n{value}")
        return value


def _connect(address: str, authkey: bytes):
    deadline = time.monotonic() + CONNECT_WAIT
    while True:
        try:
            return Client(address, family="AF_UNIX", authkey=authkey)
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def shared(kind: str) -> RemoteModel | None:
    """A RemoteModel for `kind` ('embedder' or 'reranker') when DOCSEARCH_MODEL_SOCKET names a reachable server, else None."""
    global _warned
    address = os.environ.get(SOCKET_ENV)
    if not address:
        return None
    remote = RemoteModel(kind, address, os.environ[AUTHKEY_ENV].encode())
    try:
        remote._call("ping")
    except (OSError, EOFError, AuthenticationError) as exc:
        if not _warned:
            _warned = True
            log.warning("model server %s unreachable (%r); loading retrieval models locally", address, exc)
        return None
    return remote


class Server:
    def __init__(self, authkey: bytes, loaders: dict | None = None):
        self.authkey = authkey
        self.loaders = LOADERS if loaders is None else loaders
        self.models: dict = {}
        self.gpu = threading.Lock()  # one GPU: one model call at a time; the first call of each model loads it under this lock

    def serve_forever(self, listener: Listener) -> None:
        """Serve until the listener is closed (accept fails)."""
        while True:
            try:
                conn = listener.accept()
            except AuthenticationError:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn) -> None:
        with conn:
            try:
                kind, op, args, kwargs = conn.recv()
                reply = ("ok", self.handle(kind, op, args, kwargs))
            except Exception:
                reply = ("error", traceback.format_exc())
            try:
                conn.send(reply)
            except OSError:
                pass  # the client went away

    def handle(self, kind: str, op: str, args: tuple, kwargs: dict):
        if op == "ping":
            return None
        if op not in OPS or kind not in self.loaders:
            raise ValueError(f"unknown request {kind!r}.{op!r}")
        with self.gpu:
            if kind not in self.models:
                self.models[kind] = self.loaders[kind]()
            return np.asarray(getattr(self.models[kind], op)(*args, **kwargs))


def _exit_with_parent() -> None:
    sys.stdin.read()  # EOF when the web server closes the pipe or dies
    os._exit(0)


def main(argv: list[str]) -> None:
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    authkey = os.environ[AUTHKEY_ENV].encode()
    with Listener(argv[1], family="AF_UNIX", authkey=authkey) as listener:
        Server(authkey).serve_forever(listener)


if __name__ == "__main__":
    main(sys.argv)
