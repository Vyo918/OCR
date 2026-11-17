import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from docsearch import config, factquery, timing, trace
from docsearch.ocr_ingest import tiles
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

    def evidence_text(self, ev, mode):
        return '\n\n'.join(f'<chunk id="{c["id"]}">\n{c["text"]}\n</chunk>' for c in ev)

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

    def test_page_url_is_webp_within_sheet_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'page.png')
            page = Image.new('RGB', (2600, 1800), 'white')
            ImageDraw.Draw(page).rectangle([100, 100, 900, 400], outline='black', width=12)
            page.save(path)
            url = trace._page_url(path, trace.SHEET_BUDGET)
        self.assertTrue(url.startswith('data:image/webp;base64,'))
        webp = base64.b64decode(url.split(',', 1)[1])
        self.assertLessEqual(len(webp), trace.SHEET_BUDGET)
        self.assertEqual(Image.open(io.BytesIO(webp)).size, (trace.SHEET_WIDTH, round(1800 * trace.SHEET_WIDTH / 2600)))

    def test_source_area_is_padded_and_clamped(self):
        self.assertEqual(trace._source_area([100, 100], [[5, 5, 95, 95]]), [0, 0, 100, 100])
        self.assertEqual(trace._source_area([1000, 1000], [[400, 400, 500, 450]]), [340, 340, 560, 510])

    def test_overlay_tiles_are_the_ingest_grid_and_mark_blank(self):
        engine = FakeTraceEngine('')
        engine.docs = {'D': {'size': [2600, 1400], 'image': ''}}
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp, 'page.png')
            img = Image.new('RGB', (2600, 1400), 'white')
            ImageDraw.Draw(img).rectangle([0, 0, 999, 999], fill='black')
            img.save(page)
            Path(tmp, 'D.lines.json').write_text(json.dumps({'doc': 'D', 'lines': [{'text': 'a', 'bbox': [10, 10, 60, 30]}]}), encoding='utf-8')
            overlay = trace._overlay(engine, 'D', [], page, Path(tmp), set())
        grid = list(tiles(2600, 1400, trace.PASS_TILE, trace.PASS_OVERLAP))
        self.assertEqual([t[1:5] for t in overlay['tiles']], [list(t) for t in grid])
        self.assertEqual(overlay['tiles'][0][1:7], [0, 0, 1280, 1280, 0, 1])
        self.assertEqual(sum(t[5] for t in overlay['tiles']), len(grid) - 1)

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
        data = {'title': '</script><b>x', 'stages': [], 'compare': {}, 'answers': {}, 'meta': {}, 'reference': '',
                'overview': {'index': {'label': '', 'nodes': []}, 'answer': {'label': '', 'nodes': []}}}
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
            html = trace.render(data, Path(tmp, 'trace.html')).read_text(encoding='utf-8')
        self.assertEqual([s['id'] for s in data['stages']],
                         ['render', 'tiling', 'ocr_pass', 'reread', 'merge', 'chunking', 'index_build', 'fact_parse',
                          'question', 'page', 'ocr', 'chunks', 'expand', 'retrieve', 'rrf', 'rerank', 'anchor', 'facts',
                          'prompt', 'llm', 'answers'])
        self.assertIsNone(data['answers']['a']['answer'])
        self.assertGreater(data['compare']['a']['chars'], 0)
        self.assertEqual(data['meta']['question_id'], 'q1')
        stages = {s['id']: s for s in data['stages']}
        overlay = data['overlays']['D']
        self.assertIsNone(overlay['src'])
        self.assertEqual(overlay['lines'], [[0, 0, 20, 10, '調整池(1)']])
        self.assertEqual({c['id'] for c in overlay['chunks']}, {'D#0', 'D#1'})
        self.assertEqual(stages['page']['output'][0]['t'], 'overlay')
        for stage, layers in (('page', ['src']), ('ocr', ['src', 'lines']), ('chunks', ['lines', 'chunks', 'ev', 'src']), ('tiling', ['tiles'])):
            self.assertEqual(stages[stage]['output'][0]['layers'], layers)
        slope = next(b for b in stages['rrf']['output'] if b['t'] == 'slope')
        self.assertEqual((slope['upto'], slope['depth'], len(slope['cols'])), (2, 60, 5))
        self.assertEqual({r['id'] for r in slope['rows']}, {'D#0', 'D#1'})
        strip = next(b for b in stages['rerank']['output'] if b['t'] == 'strip')
        self.assertEqual([item[0] for item in strip['items']], ['D#0', 'D#1'])
        for row in stages['prompt']['output'][0]['rows']:
            self.assertEqual(sum(seg[1] for seg in row['segs']), row['total'])
        for marker in ('"t": "overlay"', '"t": "slope"', '"t": "strip"', '"t": "stack"'):
            self.assertIn(marker, html)

    def test_timing_summary_sums_question_documents_and_derives_llm(self):
        indexing = {'D': {'prepare': 1.0, 'pp_ocr': 2.5, 'chunking': 0.5}, 'E': {'pp_ocr': 4.0},
                    timing.INDEX: {'embedding': 3.0}, timing.META: {'device': 'cpu', 'engine': 'hybrid', 'recorded': 'now'}}
        s = trace.timing_summary(indexing, ['D'], {'retrieve': 12.3, 'rerank': 100.0, 'prompt': 5.0}, 60.0)
        self.assertTrue(s['index_recorded'])
        self.assertEqual([st['name'] for st in s['index']], ['prepare', 'pp_ocr', 'chunking', 'embedding'])
        self.assertEqual(s['index'][1], {'name': 'pp_ocr', 'seconds': 2.5, 'per_doc': {'D': 2.5}})
        self.assertEqual(s['index'][3]['per_doc'], {'whole file set': 3.0})
        self.assertEqual(s['index_total'], 7.0)
        self.assertEqual(s['missing_docs'], [])
        self.assertEqual(s['device'], 'cpu')
        self.assertEqual([q['name'] for q in s['question']], ['retrieve', 'rerank', 'prompt'])
        self.assertEqual(s['llm'], 59.883)
        self.assertEqual(s['question_total'], 60.0)

    def test_timing_summary_without_timing_file_keeps_question_side(self):
        s = trace.timing_summary(None, ['D'], {'retrieve': 10.0}, 5.0)
        self.assertFalse(s['index_recorded'])
        self.assertEqual(s['index'], [])
        self.assertEqual(s['missing_docs'], ['D'])
        self.assertEqual(s['llm'], 4.99)

    def test_llm_stage_is_never_negative_and_unknown_without_recorded_seconds(self):
        self.assertEqual(trace.timing_summary(None, [], {'retrieve': 5000.0}, 1.0)['llm'], 0.0)
        self.assertIsNone(trace.timing_summary(None, [], {'retrieve': 5000.0}, None)['llm'])

    def test_collect_reads_indexing_seconds_of_every_document_in_the_file_set(self):
        question = '調整池(1)的有效容積是多少？'
        indexing = {'D': {'pp_ocr': 2.0}, timing.META: {'device': 'cpu', 'engine': 'hybrid', 'recorded': 'now'}}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'D.lines.json').write_text(json.dumps({'doc': 'D', 'lines': [{'text': '調整池(1)', 'bbox': [0, 0, 20, 10]}]}), encoding='utf-8')
            engine = FakeTraceEngine(str(Path(tmp, 'missing.png')))
            data = trace.collect(engine, question, ocr_dir=Path(tmp), pages_dir=Path(tmp, 'pages'), indexing=indexing,
                                 recorded={'facts': {'seconds': 90.0}})
        self.assertEqual(data['timing']['docs'], ['D'])
        self.assertEqual(data['timing']['index'][0], {'name': 'pp_ocr', 'seconds': 2.0, 'per_doc': {'D': 2.0}})
        self.assertEqual(data['timing']['result_seconds'], 90.0)
        self.assertGreater(data['timing']['llm'], 89.9)

    def test_page_says_when_indexing_time_is_not_recorded(self):
        self.assertIn('indexing time not recorded for this index', trace.TEMPLATE.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
