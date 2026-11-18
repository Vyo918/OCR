"""Re-read pump specification boxes at higher resolution during ingest."""
import math
import os

from PIL import Image

from .facts_tables import pump_region, pump_title
from .ocr_ingest import inside


def reread(page, engine, lines, name):
    if os.environ.get('DOCSEARCH_TABLE_REREAD', '1') == '0' or not callable(getattr(engine, 'spot', None)):
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
        if len(found) * 2 < len(old):
            continue
        new = []
        for text, (a, b, c, d), score in found:
            line = {'doc': name, 'text': text,
                    'bbox': [round(box[0] + a / 3), round(box[1] + b / 3),
                             round(box[0] + c / 3), round(box[1] + d / 3)]}
            if score is not None:
                line['score'] = round(score, 3)
            new.append(line)
        # The title is what _pump keys on: keep the page-level read of it and take only the rows from the crop.
        new = [line for line in new if inside(line['bbox'], title['bbox']) < .6]
        lines = [line for line in lines if line is title or inside(line['bbox'], box) < .6] + new
        lines.sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))
    return lines
