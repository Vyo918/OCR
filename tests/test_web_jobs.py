import subprocess
import tempfile
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from docsearch.web_jobs import preflight_pdf


class RetrievalLockTests(unittest.TestCase):
    def test_retrieval_is_exclusive_but_unlocked_work_overlaps(self):
        with tempfile.TemporaryDirectory() as temp:
            script = """import sys,time
from docsearch.web_jobs import RetrievalLock
with RetrievalLock(sys.argv[1]):
 print('locked', flush=True)
 time.sleep(.35)
print('unlocked', flush=True)
time.sleep(.5)
"""
            lock = str(Path(temp) / 'lock')
            first = subprocess.Popen([sys.executable, '-c', script, lock], stdout=subprocess.PIPE, text=True)
            second = None
            try:
                self.assertEqual(first.stdout.readline().strip(), 'locked')
                start = time.monotonic()
                second = subprocess.Popen([sys.executable, '-c', script, lock], stdout=subprocess.PIPE, text=True)
                self.assertEqual(first.stdout.readline().strip(), 'unlocked')
                self.assertEqual(second.stdout.readline().strip(), 'locked')
                self.assertIsNone(first.poll())
                self.assertEqual(second.stdout.readline().strip(), 'unlocked')
                self.assertLess(time.monotonic() - start, 1.1)
                self.assertEqual(first.wait(timeout=3), 0)
                self.assertEqual(second.wait(timeout=3), 0)
            finally:
                for process in (first, second):
                    if process:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                        process.stdout.close()


class PdfPreflightTests(unittest.TestCase):
    def test_rejects_encrypted_pdf_with_padded_pdfinfo_value(self):
        info = 'Pages:          1\nEncrypted:      yes (print:yes copy:yes change:yes addNotes:yes)\n'
        with patch('docsearch.web_jobs.subprocess.run', return_value=subprocess.CompletedProcess([], 0, info, '')):
            with self.assertRaisesRegex(ValueError, 'pdf'):
                preflight_pdf(Path('document.pdf'))

    def test_rejects_oversized_rendered_page_before_rasterizing(self):
        info = 'Pages: 1\nEncrypted: no\nPage 1 size: 20000 x 20000 pts\n'
        def run(args, **kwargs):
            if args[0] == 'pdfinfo':
                return subprocess.CompletedProcess(args, 0, info, '')
            return subprocess.CompletedProcess(args, 0, 'page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio\n', '')
        with patch('docsearch.web_jobs.subprocess.run', side_effect=run):
            with self.assertRaisesRegex(ValueError, 'pdf'):
                preflight_pdf(Path('document.pdf'))

    def test_missing_pdf_tool_has_categorized_failure(self):
        with patch('docsearch.web_jobs.subprocess.run', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, 'pdf'):
                preflight_pdf(Path('document.pdf'))


if __name__ == '__main__':
    unittest.main()
