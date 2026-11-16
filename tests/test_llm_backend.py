import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from docsearch import config
from docsearch.search import Engine


def sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def delta(finish=None, **fields):
    return {"choices": [{"delta": fields, "finish_reason": finish}]}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        text = self.server.reply(body)
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class _Server:
    """Real HTTP server on a thread. `reply(body)` returns the SSE text for one request."""

    def __init__(self, reply):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.requests = []
        self.httpd.reply = reply
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def requests(self):
        return self.httpd.requests


def _engine():
    e = Engine.__new__(Engine)
    e.seed = 0
    return e


class LlmBackendTests(unittest.TestCase):
    def openai(self, url):
        return mock.patch.multiple(config, LLM_BACKEND="openai", LLM_URL=url, LLM_API_KEY="", LLM_ID="test-model")

    def test_reasoning_content_field(self):
        reply = lambda body: sse(delta(reasoning_content="Let me "), delta(reasoning_content="think."), delta(content="Answer: "), delta(content="3"), delta("stop"))
        with _Server(reply) as srv, self.openai(srv.url):
            answer, reasoning = _engine().chat("sys", "usr")
            req = srv.requests[0]
        self.assertEqual((answer, reasoning), ("Answer: 3", "Let me think."))
        self.assertTrue(req["stream"])
        self.assertEqual((req["model"], req["seed"], req["max_tokens"]), ("test-model", 0, 8500))
        self.assertEqual((req["temperature"], req["top_p"], req["top_k"]), (0.6, 0.95, 20))
        self.assertNotIn("chat_template_kwargs", req)
        self.assertEqual([m["role"] for m in req["messages"]], ["system", "user"])

    def test_inline_think_in_content(self):
        reply = lambda body: sse(delta(content="<think>Pipe is 4 inch."), delta(content="</think>\n\nThe answer is 4."), delta("stop"))
        with _Server(reply) as srv, self.openai(srv.url):
            answer, reasoning = _engine().chat("sys", "usr")
        self.assertEqual((answer, reasoning), ("The answer is 4.", "Pipe is 4 inch."))

    def test_length_with_only_reasoning_runs_fallback(self):
        reasoning = "A" * 5000 + "TAILMARK"

        def reply(body):
            if body.get("chat_template_kwargs"):
                return sse(delta(content="Final: 7"), delta("stop"))
            return sse(delta(reasoning_content=reasoning), delta("length"))

        with _Server(reply) as srv, self.openai(srv.url):
            answer, got = _engine().chat("sys", "usr")
            first, fallback = srv.requests
        self.assertEqual((answer, got), ("Final: 7", reasoning))
        self.assertEqual(fallback["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(fallback["messages"][0]["content"], "sys")
        user = fallback["messages"][1]["content"]
        self.assertTrue(user.startswith("usr\n\nYour earlier reasoning ran out of budget"))
        self.assertTrue(user.endswith(reasoning[-4000:]))
        self.assertNotIn("A" * 4001, user)
        self.assertEqual(first["messages"][1]["content"], "usr")

    def test_truncated_reasoning_stream_raises(self):
        reply = lambda body: f"data: {json.dumps(delta(reasoning_content='thinking'))}\n\n"
        with _Server(reply) as srv, self.openai(srv.url):
            with self.assertRaises(RuntimeError) as exc:
                _engine().chat("sys", "usr")
            self.assertIn(srv.url + "/chat/completions", str(exc.exception))
            self.assertIn("without finish_reason or [DONE]", str(exc.exception))

    def test_empty_finished_answer_raises(self):
        reply = lambda body: sse(delta(content="", reasoning_content=""), delta("stop"))
        with _Server(reply) as srv, self.openai(srv.url):
            with self.assertRaises(RuntimeError) as exc:
                _engine().chat("sys", "usr")
            self.assertIn(srv.url + "/chat/completions", str(exc.exception))
            self.assertIn("no answer text", str(exc.exception))

    def test_empty_length_fallback_raises(self):
        def reply(body):
            if body.get("chat_template_kwargs"):
                return sse(delta(content="  "), delta("stop"))
            return sse(delta(reasoning_content="thinking"), delta("length"))

        with _Server(reply) as srv, self.openai(srv.url):
            with self.assertRaises(RuntimeError) as exc:
                _engine().chat("sys", "usr")
            self.assertIn(srv.url + "/chat/completions", str(exc.exception))
            self.assertIn("no answer text", str(exc.exception))

    def test_finish_reason_or_done_independently_complete_stream(self):
        replies = [
            f"data: {json.dumps(delta(content='answer'))}\n\ndata: {json.dumps(delta('stop'))}\n\n",
            sse(delta(content="answer")),
        ]
        for response in replies:
            with self.subTest(response=response), _Server(lambda body: response) as srv, self.openai(srv.url):
                self.assertEqual(_engine().chat("sys", "usr"), ("answer", ""))

    def test_connection_refused_names_url(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{s.getsockname()[1]}/v1"
        with self.openai(url), self.assertRaises(RuntimeError) as exc:
            _engine().chat("sys", "usr")
        self.assertIn(url + "/chat/completions", str(exc.exception))

    def test_sheet_context_is_stable_and_skips_neighbourhood(self):
        e = _engine()
        e.docs = {"B": {}, "A": {}}
        e.chunks = [
            {"id": "A#0", "doc": "A", "kind": "block", "text": "a0"},
            {"id": "A#1", "doc": "A", "kind": "neighbourhood", "text": "an"},
            {"id": "A#2", "doc": "A", "kind": "section", "text": "a2"},
            {"id": "B#3", "doc": "B", "kind": "block", "text": "b3"},
            {"id": "X#4", "doc": "X", "kind": "block", "text": "x4"},
        ]
        first = e.evidence_text([e.chunks[1]], "sheet")
        self.assertEqual(first, e.evidence_text([e.chunks[0]], "sheet"))
        self.assertEqual(first, '<chunk id="B#3" kind="block">\nb3\n</chunk>\n\n<chunk id="A#0" kind="block">\na0\n</chunk>\n\n<chunk id="A#2" kind="section">\na2\n</chunk>')

    def test_chunks_context_format(self):
        e = _engine()
        ev = [{"id": "A#0", "kind": "block", "text": "a0"}, {"id": "A#2", "kind": "section", "text": "a2"}]
        self.assertEqual(e.evidence_text(ev, "chunks"), '<chunk id="A#0" kind="block">\na0\n</chunk>\n\n<chunk id="A#2" kind="section">\na2\n</chunk>')


if __name__ == "__main__":
    unittest.main()
