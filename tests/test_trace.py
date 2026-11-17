import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from docsearch import config, factquery, trace
from docsearch.search import Engine


class AnswerEngine(Engine):
    """Engine.answer with retrieval stubbed; it records the (system, user) messages sent to chat."""

    def __init__(self):
        self.chunks = [{'id': 'D#0', 'doc': 'D', 'kind': 'block', 'bbox': [0, 0, 20, 10], 'body': 'evidence', 'text': 'evidence'},
                       {'id': 'D#1', 'doc': 'D', 'kind': 'block', 'bbox': [0, 10, 20, 20], 'body': 'source', 'text': 'source'}]
        self.outlines = {'D': ['outline']}
        self.facts = [{'id': 'F001', 'doc': 'D', 'source': 'calc', 'entity': '調整池', 'family': '調整池', 'equipment': None, 'index': None,
                       'ekey': '調整池', 'attr': 'effective_volume', 'label': '有效容積', 'value': 237.37, 'value_si': 237.37, 'unit': 'm3',
                       'unit_si': 'm3', 'text': '有效容積 237.37m3', 'section': '', 'src': [{'chunk': 'D#1', 'bbox': [0, 10, 20, 20]}], 'extra': {}}]
        self.hybrid = lambda q: [0]
        self.rerank = lambda q, c, top: [0]
        self.prompts = []
        self.chat = lambda system, user: (self.prompts.append((system, user)) or ('answer', 'reasoning'))


class BuildPromptIdentityTests(unittest.TestCase):
    QUESTION = '調整池實際有效容積是多少？'

    def test_build_prompt_matches_answer_flag_off_and_on(self):
        e = AnswerEngine()
        seen = {}
        for enabled in (False, True):
            with self.subTest(facts=enabled), mock.patch.object(config, 'FACTS', enabled), \
                 mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
                e.answer(self.QUESTION)
                fx = factquery.context(self.QUESTION, e.facts) if enabled else None
                self.assertEqual(e.prompts[-1], e.build_prompt(self.QUESTION, [e.chunks[0]], fx))
                seen[enabled] = e.prompts[-1]
        self.assertNotEqual(seen[False], seen[True])


class FakeTraceEngine:
    """Retrieval stubs with the Engine method shapes. chat raises, so any LLM call fails the test."""

    def __init__(self, image: str):
        self.chunks = [{'id': 'D#0', 'doc': 'D', 'kind': 'block', 'bbox': [0, 0, 40, 10], 'body': '調整池(1) 有效容積', 'text': '調整池(1) 有效容積', 'section': 'S'},
                       {'id': 'D#1', 'doc': 'D', 'kind': 'block', 'bbox': [0, 10, 40, 20], 'body': '其他', 'text': '其他', 'section': ''}]
        self.docs = {'D': {'size': [100, 100], 'image': image}}
        self.facts = []
        self.outlines = {'D': ['outline']}

    def expand(self, q):
        return [q]

    def lexical(self, q, k):
        return [0, 1]

    def dense(self, queries, k):
        return [[1, 0] for _ in queries]

    def reranker(self):
        return self

    def predict(self, pairs, batch_size):
        return [0.9 - 0.8 * j for j in range(len(pairs))]

    def novel(self, order, top):
        return order[:top]

    def build_prompt(self, question, ev, fx):
        return ('system', f'user {question} {len(ev)} {fx is not None}')

    def chat(self, system, user):
        raise AssertionError('trace must not call the LLM')


class TraceTests(unittest.TestCase):
    def test_fuse_sums_reciprocal_ranks(self):
        out = trace.fuse([[1, 2], [2, 3]])
        self.assertEqual([i for i, _ in out], [2, 1, 3])
        self.assertAlmostEqual(dict(out)[1], 1 / 61)
        self.assertAlmostEqual(dict(out)[2], 1 / 62 + 1 / 61)

    def test_diff_marks_changed_line_only(self):
        self.assertEqual(trace._diff('a\nb\nc', 'a\nB\nc'), [['eq', 'a'], ['del', 'b'], ['ins', 'B'], ['eq', 'c']])

    def test_crop_is_png_of_region_with_source_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'page.png')
            Image.frombytes('RGB', (100, 80), os.urandom(100 * 80 * 3)).save(path)
            url = trace._crop_url(path, [100, 80], [10, 10, 50, 40], [[12, 12, 30, 30]], 10 ** 6)
        self.assertTrue(url.startswith('data:image/png;base64,'))
        png = base64.b64decode(url.split(',', 1)[1])
        self.assertEqual(Image.open(io.BytesIO(png)).size, (40, 30))

    def test_region_is_padded_and_clamped(self):
        self.assertEqual(trace._region([100, 100], [[5, 5, 95, 95]]), [0, 0, 100, 100])

    def test_verdicts(self):
        self.assertEqual(trace._answer('A', {'grade': 'pass', 'answer': 'x'})['verdict'], 'correct')
        self.assertEqual(trace._answer('A', {'grade': 'fail', 'answer': 'x'})['verdict'], 'incorrect')
        empty = trace._answer('A', None)
        self.assertIsNone(empty['verdict'])
        self.assertIsNone(empty['answer'])

    def test_question_item_accepts_list_wrapper_or_bare_item(self):
        self.assertEqual(trace.question_item({'題目': [{'編號': 'x'}]}), {'編號': 'x'})
        self.assertEqual(trace.question_item({'編號': 'x'}), {'編號': 'x'})

    def test_render_escapes_script_text_and_fills_placeholder(self):
        data = {'title': '</script><b>x', 'stages': [], 'compare': {}, 'answers': {}, 'meta': {}, 'reference': ''}
        with tempfile.TemporaryDirectory() as tmp:
            out = trace.render(data, Path(tmp, 'sub', 'trace.html'))
            html = out.read_text(encoding='utf-8')
        self.assertNotIn('</script><b>x', html)
        self.assertIn('\\u003c/script\\u003e', html)
        self.assertNotIn(trace.PLACEHOLDER, html)

    def test_render_rejects_template_without_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp, 'template.html')
            template.write_text('no slot here', encoding='utf-8')
            with mock.patch.object(trace, 'TEMPLATE', template):
                with self.assertRaises(ValueError):
                    trace.render({}, Path(tmp, 'out.html'))

    def test_collect_builds_every_stage_without_llm(self):
        question = '調整池(1)的有效容積是多少？'
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'D.lines.json').write_text(json.dumps({'doc': 'D', 'lines': [{'text': '調整池(1)', 'bbox': [0, 0, 20, 10]}]}), encoding='utf-8')
            engine = FakeTraceEngine(str(Path(tmp, 'missing.png')))
            data = trace.collect(engine, question, question_id='q1', ocr_dir=Path(tmp), pages_dir=Path(tmp, 'pages'))
            json.dumps(data, ensure_ascii=False)
        self.assertEqual([s['id'] for s in data['stages']],
                         ['question', 'page', 'ocr', 'chunks', 'expand', 'retrieve', 'rrf', 'rerank', 'anchor', 'facts', 'prompt', 'answers'])
        self.assertIsNone(data['answers']['a']['answer'])
        self.assertGreater(data['compare']['a']['chars'], 0)
        self.assertEqual(data['meta']['question_id'], 'q1')


if __name__ == '__main__':
    unittest.main()
