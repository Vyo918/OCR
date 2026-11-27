"""Extract equipment legends and pump specification tables from OCR geometry."""
import re

from . import index
from .facts import quantity, row


def _cy(line):
    return (line.bbox[1] + line.bbox[3]) / 2


def _same_row(a, b):
    """Lines share a legend row when their vertical extents overlap by at least half the shorter line."""
    overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    return overlap >= .5 * min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])


def _glyph(line):
    """Lone digit or symbol misread from a legend symbol; never part of a name."""
    return len(line.norm) <= 2 and not re.search(r'[\u4e00-\u9fff]', line.norm)


def _legend(sheet):
    lines = sheet.lines
    headers = [l for l in lines if '图例' in l.key.replace(' ', '')]
    result = []
    for header in headers:
        descriptions = [l for l in lines if '说明' in l.key.replace(' ', '') and abs(_cy(l) - _cy(header)) <= sheet.h]
        if not descriptions:
            continue
        left = header.bbox[2]
        cells = sorted((l for l in lines if l.bbox[0] > descriptions[0].bbox[2] and l.bbox[1] > header.bbox[3]
                        and re.fullmatch(r'\s*\d+(?:\.\d+)?\s*(?:只|台|組|组|套|支|m)\s*', l.norm, re.I)), key=_cy)
        previous = header.bbox[3]
        for cell in cells:
            if cell.bbox[1] - previous > 8 * sheet.h:
                break
            same = sorted((l for l in lines if l.bbox[0] >= left and l.bbox[2] <= cell.bbox[0]
                           and _same_row(l, cell)), key=lambda l: l.bbox[0])
            if not same:
                break
            previous = cell.bbox[3]
            text = ' '.join(l.norm for l in same if not _glyph(l))
            if not text:
                continue
            entity = re.split(r'\s*[(（]|\s*[×xX](?=\d)|\s+(?=\d)', text)[0].strip()
            used = same + [cell]
            quantity_row = quantity(cell.norm)
            length = quantity_row['unit'] == 'm'
            material = re.search(r'[(（]([A-Za-z][^()（）]*)[)）]', text)
            result.append(row(sheet, 'legend', entity, 'length' if length else 'qty', '長度' if length else '數量',
                              quantity_row, used, section='圖例', extra={'material': material.group(1)} if material else {}))
            patterns = [('flow', '流量', r'\d+(?:\.\d+)?\s*m3/min'), ('head', '揚程', r'\d+(?:\.\d+)?\s*mH'),
                        ('hp', '馬力', r'\d+(?:\.\d+)?\s*HP'), ('air', '風量', r'\d+(?:\.\d+)?\s*CMM'),
                        ('pressure', '壓力', r'\d+(?:\.\d+)?\s*mmA[gq]'), ('bore', '口徑', r'\d+(?:\.\d+)?\s*["″°”]'),
                        ('other', '容量', r'\d+(?:\.\d+)?\s*L'), ('qty', '數量', r'(?<=[×xX])\s*\d+\s*(?:台|只)')]
            spec = text[len(entity):]
            for attr, label, pattern in patterns:
                for m in re.finditer(pattern, spec, re.I):
                    component = entity
                    if 'PE貯藥桶' in spec and m.start() > spec.index('PE貯藥桶'):
                        component = 'PE貯藥桶'
                    value = quantity(m.group().replace('”', '"'))
                    result.append(row(sheet, 'legend', component, attr, label, value, same, section='圖例'))
    return result


_PUMP_TITLE = re.compile(r'(.{0,6}泵)\s*([A-Z]?)\s*规格', re.I)


def pump_region(bbox):
    """Region around a pump title, including the specification rows."""
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    return (x0 - 1.5*h, y0 - .5*h, x1 + .6*(x1 - x0), y1 + 10*h)


def pump_title(text):
    return _PUMP_TITLE.search(index.fold(index.norm(text)))


def _pump(sheet):
    lines = sheet.lines
    titles = [l for l in lines if pump_title(l.text)]
    result = []
    for title in titles:
        match = pump_title(title.text)
        if not match:
            continue
        entity = re.split(r'规[格格]|規格', title.norm, flags=re.I)[0].strip()
        letter = match.group(2).upper() or None
        h = title.bbox[3] - title.bbox[1]
        x0, _, x1, _ = pump_region(title.bbox)
        nearby = [l for l in lines if l is not title and x0 <= l.bbox[0] <= x1
                  and title.bbox[3] - h*.5 <= l.bbox[1] <= title.bbox[3] + 9*h]
        nearby.sort(key=_cy)
        groups = []
        for l in nearby:
            if groups and abs(_cy(l) - _cy(groups[-1][0])) < .25*h:
                groups[-1].append(l)
            else:
                groups.append([l])
        last = title.bbox[3]
        seen = set()
        for group in groups:
            if min(l.bbox[1] for l in group) - last > 2*h or ('hp' in seen and min(l.bbox[1] for l in group) - last > .5*h):
                break
            last = max(last, max(l.bbox[3] for l in group))
            group.sort(key=lambda l: l.bbox[0])
            text = ' '.join(l.norm for l in group).replace('′′', '″')
            folded = index.fold(text)
            attrs = []
            for attr, label, pattern in [('qty','數量',r'\d+\s*台'), ('flow','流量',r'[><≥≤]?\s*\d+(?:\.\d+)?\s*L\s*/\s*min'), ('hp','馬力',r'\d+(?:\.\d+)?\s*HP')]:
                m = re.search(pattern, text, re.I)
                if m: attrs.append((attr, label, quantity(m.group())))
            power = re.search(r'(\d)\D?\s*(\d{3})\s*V\s*(\d{2})\s*HZ', text, re.I)
            if power:
                q = quantity(power.group(2))
                q.update(unit='V', unit_si='V')
                attrs.append(('power','電源',q))
            mode = re.search(r'[^\s]*?(?:運轉|連轉)', text)
            if mode: attrs.append(('mode','動作方式',None))
            for attr, label, pattern in [('brand','廠牌',r'厂牌\s*[:：]\s*(\S+)'), ('model','型號',r'型号\s*[:：]\s*(\S+)')]:
                if re.search(pattern, folded, re.I): attrs.append((attr,label,None))
            if not attrs:
                label = 'head' if re.search(r'扬程|構程|构程', folded) else 'bore' if re.search(r'口[径得]?|口徑', folded) else None
                if label is None and 'power' in seen and 'flow' not in seen:
                    label = 'bore'
                elif label is None and 'flow' in seen and 'head' not in seen:
                    label = 'head'
                if label:
                    value = re.search(r'[><≥≤]?\s*\d+(?:\.\d+)?\s*[M"″°吋]?', text, re.I)
                    if label == 'bore':
                        value = re.search(r'[><≥≤]?\s*\d+(?:\.\d+)?\s*["″°吋]', text) or value
                    if value:
                        raw = value.group().replace('M', 'm').replace('″', '"')
                        suspect = label == 'bore' and not re.search(r'["″°吋]', raw)
                        attrs.append((label, '口徑(未可靠辨識)' if suspect else '揚程' if label == 'head' else '口徑',
                                      None if suspect else quantity(raw)))
            for attr,label,q in attrs:
                extra = {'phase': int(power.group(1)), 'frequency': int(power.group(3))} if attr == 'power' else {}
                if attr == 'bore' and q is None:
                    extra['suspect'] = 'no inch mark'
                item = row(sheet, 'pump_table', entity, attr, label, q, group, section=title.norm, index=letter, extra=extra)
                if attr == 'mode':
                    item['value'] = mode.group()
                elif attr in ('brand', 'model'):
                    item['value'] = re.split(r'[:：]', text, maxsplit=1)[-1].strip()
                result.append(item)
                seen.add(attr)
    return result


def rows(sheet):
    return _legend(sheet) + _pump(sheet)
