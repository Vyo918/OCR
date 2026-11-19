"""Re-read pump specification boxes at higher resolution during ingest."""
from collections import Counter
from itertools import product
import math
import os
import re

from PIL import Image, ImageOps

from .facts_tables import pump_region, pump_title
from .ocr_ingest import inside


CELL_VARIANTS = tuple(product((0, 3), (3, 4, 5, 6), (None, 110, 130, 150, 170)))
CELL_READING = re.compile(r'\s*(\d{1,2})\s*(["″°吋]?)\s*')

PHASE_VOLTAGE = re.compile(r'(?<!\d)([13])\s+(\d{3})\s*V\b', re.I)
VOLTAGE = re.compile(r'(?<!\d)(\d{3})\s*V\b', re.I)


def _keep_phase(old, new):
    for line in new:
        voltage = VOLTAGE.search(line['text'])
        if not voltage or re.search(r'\d\s*(?:相|[φΦ])?\s*$', line['text'][:voltage.start()]):
            continue
        for original in old:
            phase = PHASE_VOLTAGE.search(original['text'])
            if (phase and phase.group(2) == voltage.group(1)
                    and (inside(original['bbox'], line['bbox']) >= .6
                         or inside(line['bbox'], original['bbox']) >= .6)):
                line['text'] = (line['text'][:voltage.start()] + phase.group(1) + ' '
                                + line['text'][voltage.start():])
                break


def _vote_cell(page, engine, lines, title, name):
    tx0, ty0, tx1, ty1 = title['bbox']
    label = next((line for line in lines
                  if line['text'].startswith('口')
                  and ty0 < line['bbox'][1] < ty0 + 9 * (ty1 - ty0)
                  and abs(line['bbox'][0] - (tx0 - 8)) <= 40), None)
    if label is None:
        return lines
    lx0, ly0, lx1, ly1 = label['bbox']
    cell = (lx1 + 2, ly0 - 2, tx1 + 12, ly1 + 2)
    if cell[0] >= cell[2] or cell[1] >= cell[3]:
        return lines
    crop = page.crop(cell)
    votes = Counter()
    for pad, scale, threshold in CELL_VARIANTS:
        variant = ImageOps.expand(crop, border=pad, fill='white') if pad else crop
        variant = variant.resize((variant.width * scale, variant.height * scale),
                                 Image.Resampling.LANCZOS)
        if threshold is not None:
            variant = ImageOps.autocontrast(variant.convert('L')).point(
                lambda v: 255 if v > threshold else 0).convert('RGB')
        for text, _, _ in engine.spot(variant):
            match = CELL_READING.fullmatch(text)
            if match:
                votes[match.group(1)] += 1
    if not votes:
        return lines
    ranked = votes.most_common()
    digit, count = ranked[0]
    total = votes.total()
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if total < 8 or count / total < .6 or count < 2 * runner_up:
        return lines
    lines = [line for line in lines if inside(line['bbox'], cell) < .6]
    lines.append({'doc': name, 'text': digit + '"', 'bbox': list(cell),
                  'score': count / total, 'src': 'cellvote'})
    lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
    return lines


def reread(page, engine, lines, name):
    if os.environ.get('DOCSEARCH_TABLE_REREAD', '1') == '0':
        return lines
    if not callable(getattr(engine, 'spot', None)):
        engine = getattr(engine, 'fast', None)
    if not callable(getattr(engine, 'spot', None)):
        return lines
    titles = [line for line in lines if pump_title(line['text'])]
    for title in titles:
        x0, y0, x1, y1 = pump_region(title['bbox'])
        box = (max(0, math.floor(x0)), max(0, math.floor(y0)),
               min(page.width, math.ceil(x1)), min(page.height, math.ceil(y1)))
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        old = [line for line in lines if inside(line['bbox'], box) >= .6]
        crop = page.crop(box)
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)
        found = engine.spot(crop)
        if len(found) * 2 >= len(old):
            new = []
            for text, (a, b, c, d), score in found:
                line = {'doc': name, 'text': text,
                        'bbox': [round(box[0] + a / 3), round(box[1] + b / 3),
                                 round(box[0] + c / 3), round(box[1] + d / 3)]}
                if score is not None:
                    line['score'] = round(score, 3)
                new.append(line)
            # Keep the page-level title and take only the rows from the crop.
            new = [line for line in new if inside(line['bbox'], title['bbox']) < .6]
            _keep_phase(old, new)
            lines = [line for line in lines if line is title or inside(line['bbox'], box) < .6] + new
            lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
        lines = _vote_cell(page, engine, lines, title, name)
    return lines
