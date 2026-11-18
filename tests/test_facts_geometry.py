import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from docsearch import facts_geometry


class Geometry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 'page.png')
        self.members = []
        for i, (x, width) in enumerate(((40, 176), (240, 176), (440, 234), (700, 234))):
            self.members.append({'id': f'F{i+1:03}', 'doc': 'sheet', 'source': 'plan_label',
                                 'attr': 'label', 'entity': f'池({i})', 'index': str(i),
                                 'family': '初沉池', 'src': [{'chunk': str(i), 'bbox': [x+50, 240, x+90, 260]}]})
        self.formula = {'id': 'F005', 'doc': 'sheet', 'source': 'plan_volume', 'value': 239.5,
                        'src': [{'chunk': 'formula', 'bbox': [0, 0, 10, 10]}],
                        'extra': {'expr': '7.9x3.52x1.85x2+4.675x7.9x1.85x2',
                                  'members': [r['id'] for r in self.members]}}
        self.draw()

    def draw(self, heights=(395, 395, 395, 395), widths=(176, 176, 234, 234)):
        image = Image.new('RGB', (1000, 600), 'white')
        draw = ImageDraw.Draw(image)
        for x in range(0, 1000, 50):
            draw.line((x, 0, x, 599), fill=(210, 210, 210))
        for x, width, height in zip((40, 240, 440, 700), widths, heights):
            draw.rectangle((x, 50, x+width, 50+height), outline=(10, 25, 60), width=2)
        for y in range(80, 430, 30):
            draw.line((330, y, 330, y+12), fill='red', width=3)
        image.save(self.path)

    def extract(self):
        return facts_geometry.rows(self.members + [self.formula], {'sheet': {'image': self.path}})

    def test_assignment_and_sources(self):
        rows = self.extract()
        volumes = [r for r in rows if r['attr'] == 'effective_volume']
        self.assertEqual(len(rows), 8)
        self.assertEqual([round(r['value'], 2) for r in volumes], [51.44, 51.44, 68.33, 68.33])
        self.assertEqual(volumes[0]['extra']['formula_row'], 'F005')
        self.assertEqual(len(volumes[0]['src']), 2)

    def test_rejects_unreliable_plans(self):
        self.formula['extra']['members'].pop()
        self.assertEqual(self.extract(), [])
        self.formula['extra']['members'].append('F004')
        self.formula['value'] = 200
        self.assertEqual(self.extract(), [])
        self.formula['value'] = 239.5
        self.draw(heights=(395, 395, 395, 350))
        self.assertEqual(self.extract(), [])
        self.draw(widths=(176, 176, 176, 176))
        self.assertEqual(self.extract(), [])
        os.unlink(self.path)
        self.assertEqual(self.extract(), [])

    def test_switch_disables_geometry(self):
        with patch.dict(os.environ, {'DOCSEARCH_GEOMETRY': '0'}):
            self.assertEqual(self.extract(), [])
