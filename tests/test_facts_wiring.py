import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch import config, facts
from docsearch.search import Engine


SYSTEM = ("You answer questions about OCR'd construction drawings (Taiwanese MEP / wastewater) using ONLY the chunks provided.\n"
          "Reading conventions:\n"
          "- OCR is noisy (look-alike characters, 1/l, 0/O, dropped glyphs). Chunks overlap and repeat the same labels: count distinct items, not mentions. Reconcile garbled text with neighbouring chunks and say when you did.\n"
          "- Calculation sheets (計算書) and equipment tables carry design values; plan labels show where things are. Prefer the former for counts, sizes and ratings. When two sources disagree or cover different items, report each one with its source.\n"
          "- Drawing dimensions are cm unless written with m. `L790×W352.5×H350/D320` = length x width x total height / effective depth, so volume = L x W x D; lines `[tank size read: ...]` are that decoding done by the system, with the volume already computed - use them. `4\"PVC連通管x2支` = two 4-inch PVC connecting pipes; `B.O.P+320` = pipe bottom level.\n"
          "- A plan title with a volume under it (`<plan name>平面圖` followed by a sum of L x W x H terms `= ... m3`) gives the total of EVERYTHING drawn on that plan level, not of one named unit: never use it as the size of one named unit. A unit's own size comes from its calculation sheet or its own label. A `(1)(2)(3)` list on the calculation sheet is the list of that unit's tanks.\n"
          "- Never do arithmetic in your head and never write a computed result yourself. Write only the expression, as {{7.9*3.525*3.2}}; the system replaces it by `expression = exact value`.\n"
          "- Separate what the drawing states from what you infer. If the information is not in the chunks, say it was NOT FOUND - never guess, never substitute a different item (e.g. do not answer a fire-pump question with another pump's rating).\n"
          "Answer in the question's language, concisely, then list the chunk ids used.")


class WiringTests(unittest.TestCase):
    def engine(self):
        e = Engine.__new__(Engine)
        e.chunks = [{'id': 'D#0', 'doc': 'D', 'kind': 'block', 'bbox': [0, 0, 20, 10], 'body': 'evidence', 'text': 'evidence'},
                    {'id': 'D#1', 'doc': 'D', 'kind': 'block', 'bbox': [0, 10, 20, 20], 'body': 'source', 'text': 'source'}]
        e.outlines = {'D': ['outline']}
        e.facts = [{'id': 'F001', 'doc': 'D', 'source': 'calc', 'entity': '調整池', 'family': '調整池', 'equipment': None, 'index': None, 'ekey': '調整池', 'attr': 'effective_volume', 'label': '有效容積', 'value': 237.37, 'value_si': 237.37, 'unit': 'm3', 'unit_si': 'm3', 'text': '有效容積 237.37m3', 'section': '', 'src': [{'chunk': 'D#1', 'bbox': [0, 10, 20, 20]}], 'extra': {}}]
        e.hybrid = lambda q: [0]
        e.rerank = lambda q, c, top: [0]
        e.chat = lambda system, user: (e.prompts.append((system, user)) or ('answer', 'reasoning'))
        e.prompts = []
        return e

    def test_off_and_unmatched_preserve_prompt(self):
        e = self.engine()
        for enabled, question in [(False, '消防泵浦幾HP?'), (True, '消防泵浦幾HP?')]:
            with mock.patch.object(config, 'FACTS', enabled, create=True), mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
                result = e.answer(question)
            self.assertEqual(e.prompts[-1], (SYSTEM, 'Document outline:\n## D\noutline\n\nEvidence chunks:\n<chunk id="D#0" kind="block">\nevidence\n</chunk>\n\nQuestion: ' + question))
            self.assertNotIn('facts', result)

    def test_matching_rows_and_evidence(self):
        e = self.engine()
        with mock.patch.object(config, 'FACTS', True, create=True):
            result = e.answer('調整池實際有效容積是多少？')
        self.assertIn('"Fact rows" were parsed by code', e.prompts[-1][0])
        self.assertIn('answer from that row', e.prompts[-1][0])
        self.assertIn('\n\nQuestion: 調整池實際有效容積是多少？', e.prompts[-1][1])
        self.assertIn('[F001]', e.prompts[-1][1].split('\n\nQuestion:')[0])
        self.assertEqual([x['id'] for x in result['evidence']], ['D#0', 'D#1'])
        self.assertEqual(result['facts'][0]['id'], 'F001')
        self.assertIn('computed', result)

    def test_member_indices_are_parser_local(self):
        sheet = facts.Sheet('D', (100, 100), [], 10)
        with mock.patch.object(facts, 'sheets', return_value=[sheet]), \
             mock.patch('docsearch.facts_calc.rows', return_value=[{'extra': {}}]), \
             mock.patch('docsearch.facts_tables.rows', return_value=[{'extra': {}}]), \
             mock.patch('docsearch.facts_labels.rows', return_value=[{'extra': {}}, {'source': 'plan_volume', 'extra': {'member_idx': [0]}}]):
            rows = facts.extract([], {})
        self.assertEqual([r['id'] for r in rows], ['F001', 'F002', 'F003', 'F004'])
        self.assertEqual(rows[-1]['extra']['members'], ['F003'])
        self.assertNotIn('member_idx', rows[-1]['extra'])

    def test_cli_asks_without_engine(self):
        from docsearch.cli import main
        chunks = {'docs': {'D': {'size': [100, 100]}}, 'chunks': [{'id': 'D#0', 'doc': 'D', 'kind': 'block', 'lines': [{'text': '初沉池1-5)', 'bbox': [0, 0, 50, 10]}, {'text': 'H=1.85m', 'bbox': [0, 11, 50, 21]}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'chunks.json').write_text(json.dumps(chunks), encoding='utf-8')
            with mock.patch.object(config, 'INDEX_DIR', Path(tmp)), contextlib.redirect_stdout(io.StringIO()) as out:
                main(['facts', '--ask', '初沉池(1-5)的有效水深是多少？'])
        self.assertIn('[F002] 初沉池(1-5)', out.getvalue())


if __name__ == '__main__':
    unittest.main()
