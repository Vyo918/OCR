import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docsearch.web_jobs import preflight_pdf


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
