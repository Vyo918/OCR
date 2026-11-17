import unittest
from unittest import mock

from docsearch import config, verify
from docsearch.search import Engine


def chunk(i, text, kind='block', section='S', doc='D'):
    return {'id': f'D#{i}', 'doc': doc, 'kind': kind, 'bbox': [0, 0, 20, 10], 'body': text, 'text': text, 'section': section}


def fake_engine(chunks, replies):
    e = Engine.__new__(Engine)
    e.chunks = chunks
    e.outlines = {'D': ['outline']}
    e.facts = []
    e.hybrid = lambda q: list(range(len(chunks)))
    e.rerank = lambda q, c, top: list(range(len(chunks)))[:top]
    e.prompts = []
    e.replies = list(replies)

    def chat(system, user):
        e.prompts.append((system, user))
        return e.replies.pop(0), 'reasoning'
    e.chat = chat
    return e


class CheckTests(unittest.TestCase):
    def test_supported_and_unsupported_numbers(self):
        self.assertEqual(verify.check('答案 19.05 mm [D#0]', ['頂板 19.05 mm'])['verdict'], 'ok')
        self.assertEqual(verify.check('答案 237 m3 [D#0]', ['容積 12.5 m3']),
                         {'verdict': 'unsupported_number', 'unsupported': ['237 m3']})

    def test_fullwidth_digits(self):
        self.assertEqual(verify.check('答案 １９.０５ mm', ['頂板 19.05 mm'])['verdict'], 'ok')

    def test_length_units_compare_in_metres(self):
        self.assertEqual(verify.check('1.905 m', ['190.5 cm'])['verdict'], 'ok')
        self.assertEqual(verify.check('190.5 cm', ['1905 mm'])['verdict'], 'ok')
        self.assertEqual(verify.check('19.05 m', ['1905 mm'])['verdict'], 'unsupported_number')

    def test_tolerance_from_written_decimals(self):
        self.assertEqual(verify.check('19.05 mm', ['19.049 mm'])['verdict'], 'ok')
        self.assertEqual(verify.check('237 m3', ['237.37 m3'])['verdict'], 'unsupported_number')

    def test_calculator_result_supported_when_inputs_are(self):
        self.assertEqual(verify.check('{{7.9*3.525*3.2}}', ['7.9, 3.525, 3.2'])['verdict'], 'ok')
        self.assertIn('7.9*4.1*3.2', verify.check('{{7.9*4.1*3.2}}', ['7.9, 3.525, 3.2'])['unsupported'])

    def test_plain_chain_inputs_are_checked_not_result(self):
        self.assertEqual(verify.check('7.9 × 3.525 × 3.2 = 89.1', ['7.9, 3.525, 3.2'])['verdict'], 'ok')

    def test_ids_enumerators_and_think_blocks_ignored(self):
        self.assertEqual(verify.check('1. 容積 12.5 m3 [D#1]\nChunks used: D#1, F001', ['容積 12.5 m3'])['verdict'], 'ok')
        self.assertEqual(verify.check('<think>237 m3 draft</think>容積 12.5 m3', ['容積 12.5 m3'])['verdict'], 'ok')

    def test_refusal(self):
        self.assertEqual(verify.check('NOT FOUND', [])['verdict'], 'refusal')
        self.assertEqual(verify.check('未找到相關資料', [])['verdict'], 'refusal')

    def test_fact_rows_and_computed_lines_are_evidence(self):
        rows = [{'value': 237.37, 'unit': 'm3', 'value_si': 237.37, 'unit_si': 'm3'}]
        self.assertEqual(verify.check('容積 237.37 m3', [], rows)['verdict'], 'ok')
        self.assertEqual(verify.check('12.8 m3', [], computed_lines=['F001: 3.2 * 4 = 12.8 m3'])['verdict'], 'ok')

    def test_malformed_numeral_is_skipped(self):
        self.assertEqual(verify.numbers_in('二十一十 m3'), [])

    def test_missing_names_in_corpus_but_not_evidence(self):
        self.assertEqual(verify.missing('調整池(2)的有效容積是多少？', ['調整池(2) 有效容積 12.5 m3'], []), ['調整池 2'])
        self.assertEqual(verify.missing('調整池(2)的有效容積是多少？', ['調整池(2) 有效容積 12.5 m3'], ['調整池(2) 12.5 m3']), [])
        self.assertEqual(verify.missing('容積 12.5 m3 是否正確？', ['容積 12.5 m3'], ['其他 3 m3']), ['12.5 m3'])
        self.assertEqual(verify.missing('容積 99 m3 是否正確？', ['容積 12.5 m3'], ['其他 3 m3']), [])


class EngineTests(unittest.TestCase):
    def test_flag_off_leaves_result_and_prompt_unchanged(self):
        e = fake_engine([chunk(0, '容積 12.5 m3')], ['答案 12.5 m3 [D#0]'])
        with mock.patch.object(config, 'VERIFY', False), mock.patch.object(config, 'FACTS', False), \
             mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
            result = e.answer('容積多少？')
        self.assertNotIn('verify', result)
        self.assertEqual(len(e.prompts), 1)
        self.assertEqual(result['answer'], '答案 12.5 m3 [D#0]')

    def test_supported_answer_is_not_retried(self):
        e = fake_engine([chunk(0, '容積 12.5 m3')], ['答案 12.5 m3 [D#0]'])
        with mock.patch.object(config, 'VERIFY', True), mock.patch.object(config, 'FACTS', False), \
             mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
            result = e.answer('容積多少？')
        self.assertEqual(result['verify']['verdict'], 'ok')
        self.assertFalse(result['verify']['retried'])
        self.assertEqual(len(e.prompts), 1)

    def test_unsupported_number_retries_once_with_note(self):
        e = fake_engine([chunk(0, '容積 12.5 m3')], ['答案 237 m3 [D#0]', '答案 12.5 m3 [D#0]'])
        with mock.patch.object(config, 'VERIFY', True), mock.patch.object(config, 'FACTS', False), \
             mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
            result = e.answer('容積多少？')
        self.assertEqual(result['answer'], '答案 12.5 m3 [D#0]')
        self.assertEqual(result['verify'], {'verdict': 'unsupported_number', 'retried': True,
                                            'first_answer': '答案 237 m3 [D#0]', 'unsupported': ['237 m3']})
        self.assertEqual(len(e.prompts), 2)
        self.assertIn('237 m3', e.prompts[1][1])

    def test_refusal_with_corpus_name_missing_from_evidence_widens_once(self):
        chunks = [chunk(0, '其他設備 3 m3', section='S'), chunk(1, '容積 12.5 m3', section='S2'),
                  chunk(2, '容積 12.5 m3 節', kind='section', section='S')]
        e = fake_engine(chunks, ['NOT FOUND', '容積 12.5 m3 [D#2]'])
        with mock.patch.object(config, 'VERIFY', True), mock.patch.object(config, 'FACTS', False), \
             mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
            result = e.answer('容積 12.5 m3 是否正確？', top=1)
        self.assertEqual(result['answer'], '容積 12.5 m3 [D#2]')
        self.assertEqual(result['verify']['verdict'], 'refusal')
        self.assertTrue(result['verify']['retried'])
        self.assertEqual(result['verify']['first_answer'], 'NOT FOUND')
        self.assertEqual([c['id'] for c in result['evidence']], ['D#2', 'D#0', 'D#1'])
        self.assertEqual(len(e.prompts), 2)

    def test_genuine_refusal_is_not_retried(self):
        chunks = [chunk(0, '其他設備 3 m3'), chunk(1, '容積 12.5 m3', section='S2')]
        e = fake_engine(chunks, ['NOT FOUND'])
        with mock.patch.object(config, 'VERIFY', True), mock.patch.object(config, 'FACTS', False), \
             mock.patch.object(config, 'CONTEXT_MODE', 'chunks'):
            result = e.answer('容積 99 m3 是否正確？', top=1)
        self.assertEqual(result['answer'], 'NOT FOUND')
        self.assertFalse(result['verify']['retried'])
        self.assertEqual(len(e.prompts), 1)


if __name__ == '__main__':
    unittest.main()
