import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from docsearch import timing


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        patcher = mock.patch.object(timing.device, 'device', return_value='cpu')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def read(self):
        return json.loads((self.work / timing.FILE).read_text(encoding='utf-8'))

    def test_record_writes_stage_and_meta(self):
        timing.record(self.work, 'P1', 'pp_ocr', 1.23456)
        data = self.read()
        self.assertEqual(data['P1'], {'pp_ocr': 1.235})
        self.assertEqual(data[timing.META]['device'], 'cpu')
        self.assertIn('engine', data[timing.META])
        self.assertIn('T', data[timing.META]['recorded'])

    def test_later_records_merge_and_replace_only_their_stage(self):
        timing.record(self.work, 'P1', 'prepare', 2.0)
        timing.record(self.work, 'P1', 'pp_ocr', 5.0)
        timing.record(self.work, 'P1', 'pp_ocr', 6.5)
        self.assertEqual(self.read()['P1'], {'prepare': 2.0, 'pp_ocr': 6.5})

    def test_nested_docs_keep_their_own_stages(self):
        timing.record(self.work, 'P1', 'chunking', 0.5)
        timing.record(self.work, 'P2', 'chunking', 0.25)
        timing.record(self.work, timing.INDEX, 'embedding', 9.0)
        data = self.read()
        self.assertEqual(data['P1'], {'chunking': 0.5})
        self.assertEqual(data['P2'], {'chunking': 0.25})
        self.assertEqual(data[timing.INDEX], {'embedding': 9.0})

    def test_write_is_atomic_and_leaves_no_temp_file(self):
        timing.record(self.work, 'P1', 'prepare', 1.0)
        self.assertEqual(sorted(p.name for p in self.work.iterdir()), [timing.FILE])

    def test_unreadable_file_is_replaced_not_fatal(self):
        (self.work / timing.FILE).write_text('{not json', encoding='utf-8')
        timing.record(self.work, 'P1', 'prepare', 1.0)
        self.assertEqual(self.read()['P1'], {'prepare': 1.0})

    def test_failed_write_is_reported_and_swallowed(self):
        blocker = self.work / 'occupied'
        blocker.write_text('a file where the work directory should be')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            timing.record(blocker, 'P1', 'prepare', 1.0)
        self.assertIn('[timing] not recorded (P1 prepare)', err.getvalue())

    def test_stage_measures_the_block(self):
        with timing.stage(self.work, 'P1', 'vlm_reread'):
            time.sleep(0.02)
        self.assertGreaterEqual(self.read()['P1']['vlm_reread'], 0.02)

    def test_stage_records_nothing_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with timing.stage(self.work, 'P1', 'pp_ocr'):
                raise RuntimeError('boom')
        self.assertFalse((self.work / timing.FILE).exists())

    def test_load_is_none_without_a_file(self):
        self.assertIsNone(timing.load(self.work / timing.FILE))
        timing.record(self.work, 'P1', 'prepare', 1.0)
        self.assertEqual(timing.load(self.work / timing.FILE)['P1'], {'prepare': 1.0})


if __name__ == '__main__':
    unittest.main()
