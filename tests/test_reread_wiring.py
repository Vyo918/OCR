import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from docsearch import benchjson, config
from docsearch.search import Engine


BLOCK = 'Re-read of drawing crops (synthetic):\n[R1] D#1 | rows F001 | OCR: 12.5 | re-read: 12.8'


def chunk(i, text, kind='block', section='S', doc='D'):
    return {'id': f'D#{i}', 'doc': doc, 'kind': kind, 'bbox': [0, 0, 20, 10], 'body': text, 'text': text, 'section': section}


class StubReread:
    def __init__(self, out):
        self.out = out
        self.calls = []

    def run(self, ev, fx):
        self.calls.append(([c['id'] for c in ev], fx))
        return self.out


def engine(replies, chunks=None, rr=None, rerank=None):
    e = Engine.__new__(Engine)
    e.chunks = chunks or [chunk(0, 'evidence'), chunk(1, 'source')]
    e.outlines = {'D': ['outline']}
    e.facts = []
    e.hybrid = lambda q: list(range(len(e.chunks)))
    e.rerank = rerank or (lambda q, c, top: list(range(len(e.chunks)))[:top])
    e.prompts = []
    e.replies = list(replies)
    e.stub = StubReread(rr) if rr is not None else None
    e.rereader = lambda: e.stub

    def chat(system, user):
        e.prompts.append((system, user))
        return e.replies.pop(0), 'reasoning'
    e.chat = chat
    return e


class ReadWiringTests(unittest.TestCase):
    def setUp(self):
        for patcher in (mock.patch.object(config, 'CONTEXT_MODE', 'chunks'),
                        mock.patch.object(config, 'FACTS', False, create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_off_has_no_reread_key_and_identical_prompt(self):
        e = engine(['answer'], rr=BLOCK)
        with mock.patch.object(config, 'REREAD', '0'), mock.patch.object(config, 'VERIFY', False):
            result = e.answer('容積是多少？', top=1)
        self.assertNotIn('reread', result)
        self.assertEqual(e.stub.calls, [])
        self.assertEqual(e.prompts[-1][1], 'Document outline:\n## D\noutline\n\nEvidence chunks:\n<chunk id="D#0" kind="block">\nevidence\n</chunk>\n\nQuestion: 容積是多少？')
        self.assertNotIn('Re-read of drawing crops', e.prompts[-1][0])

    def test_on_puts_block_before_question_and_rule_in_system(self):
        e = engine(['answer'], rr={'text': BLOCK, 'records': [{'id': 'R1'}], 'evidence': ['12.8']})
        with mock.patch.object(config, 'REREAD', 'openai'), mock.patch.object(config, 'VERIFY', False):
            result = e.answer('容積是多少？', top=1)
        system, user = e.prompts[-1]
        self.assertEqual(user, 'Document outline:\n## D\noutline\n\nEvidence chunks:\n<chunk id="D#0" kind="block">\nevidence\n</chunk>\n\n' + BLOCK + '\n\nQuestion: 容積是多少？')
        self.assertLess(system.index('- `Re-read of drawing crops`'), system.index("Answer in the question's language"))
        self.assertEqual(result['reread'], [{'id': 'R1'}])
        self.assertEqual(e.stub.calls, [(['D#0'], None)])

    def test_on_without_text_keeps_prompt_and_records_empty(self):
        e = engine(['answer'], rr={'text': None, 'records': [], 'evidence': []})
        with mock.patch.object(config, 'REREAD', 'openai'), mock.patch.object(config, 'VERIFY', False):
            result = e.answer('容積是多少？', top=1)
        self.assertEqual(result['reread'], [])
        self.assertNotIn('Re-read of drawing crops', e.prompts[-1][0])
        self.assertNotIn(BLOCK, e.prompts[-1][1])

    def test_verify_accepts_number_only_in_reread_evidence(self):
        rr = {'text': BLOCK, 'records': [], 'evidence': ['re-read digits 19.05 mm']}
        e = engine(['19.05 mm [D#0]'], rr=rr)
        with mock.patch.object(config, 'REREAD', 'openai'), mock.patch.object(config, 'VERIFY', True):
            result = e.answer('容積是多少？', top=1)
        self.assertEqual(result['verify']['verdict'], 'ok')
        self.assertFalse(result['verify']['retried'])
        self.assertEqual(len(e.prompts), 1)

    def test_refusal_retry_keeps_block(self):
        chunks = [chunk(0, 'other 3 m3'), chunk(1, '調整池(2) 有效容積 12.5 m3')]
        rr = {'text': BLOCK, 'records': [], 'evidence': []}
        e = engine(['NOT FOUND', '12.5 m3 [D#1]'], chunks=chunks, rr=rr, rerank=lambda q, c, top: [0, 1][:top])
        # the question names a unit, so prepare() takes the reranker branch: stub it, anchor keeps the first pool entry
        e.reranker = lambda: SimpleNamespace(model=None, predict=lambda pairs, batch_size: [0.0] * len(pairs))
        e.on_gpu = lambda model: contextlib.nullcontext()
        e.novel = lambda pool, top: pool
        with mock.patch.object(config, 'REREAD', 'openai'), mock.patch.object(config, 'VERIFY', True), \
             mock.patch('docsearch.entities.anchor', lambda q, novel, pool, chunks, top: pool[:top]):
            result = e.answer('調整池(2)的有效容積是多少？', top=1)
        self.assertTrue(result['verify']['retried'])
        self.assertEqual(len(e.prompts), 2)
        self.assertIn('\n\n' + BLOCK + '\n\nQuestion:', e.prompts[1][1])
        self.assertIn('<chunk id="D#1"', e.prompts[1][1])


class BenchjsonReadTests(unittest.TestCase):
    def run_bench(self, reread_flag, rr_value):
        questions = [{"編號": "Q0", "問題": "q0", "答案": "1", "答案狀態": "有依據", "類別": "A", "難度": "簡單"}]

        class Stub:
            def answer(self, question):
                return {"answer": "1", "evidence": [{"id": "p1"}], "facts": [], "computed": [], "reread": rr_value}
        with tempfile.TemporaryDirectory() as td, mock.patch.object(config, 'REREAD', reread_flag):
            source, out = Path(td) / "input.json", Path(td) / "out"
            source.write_text(json.dumps({"題目": questions}), encoding="utf-8")
            benchjson.run(source, Stub(), out=out)
            record = json.loads((out / "results.jsonl").read_text().splitlines()[0])
            summary = (out / "summary.md").read_text()
        return record, summary

    def test_record_carries_reread_only_when_on(self):
        record, summary = self.run_bench('openai', [{'id': 'R1'}])
        self.assertEqual(record['reread'], [{'id': 'R1'}])
        self.assertIn('reread=openai', summary)

        record, summary = self.run_bench('0', None)
        self.assertNotIn('reread', record)
        self.assertNotIn('reread=', summary)


if __name__ == '__main__':
    unittest.main()
