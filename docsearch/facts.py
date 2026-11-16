"""Geometry and value helpers for facts extracted from OCR lines."""
from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from . import calc, index, ocr_ingest


FAMILIES = ('初沉池', '調整池', '接觸曝氣池', '終沉池', '消毒池', '放流池', '污泥貯留池', '抽水池')
EQUIPMENT = ('泵浦', '鼓風機', '散氣盤', '加藥機', '抽風機', '流量計')
SOFT_QUALIFIERS = ('粗氣泡', '細氣泡', '立式', '魯氏')
ATTRS = {
    'tank_dims': ('設計尺寸',), 'effective_volume': ('有效容積', '有效體積'),
    'effective_depth': ('有效水深', '水深'), 'label_depth': ('水深', '深度'),
    'label_width': ('寬度',), 'retention_time': ('停留時間',),
    'flow': ('流量', '水量'), 'design_flow': ('設計水量', '處理水量'),
    'sewage_flow': ('污水量',), 'head': ('揚程',), 'bore': ('口徑', '管徑'),
    'hp': ('馬力', 'HP'), 'qty': ('數量', '總數'), 'persons': ('人數', '使用人數'),
    'pressure': ('壓力',), 'air': ('風量', '出風量'),
    'power': ('電源', '電壓'), 'mode': ('動作方式', '運轉方式'),
    'brand': ('廠牌',), 'model': ('型號',), 'water_quality': ('水質', 'BOD', 'COD', 'SS'),
}


@dataclass(frozen=True)
class Line:
    text: str
    norm: str
    key: str
    bbox: tuple
    src: tuple


@dataclass(frozen=True)
class Sheet:
    doc: str
    size: tuple
    lines: list[Line]
    h: float


def _line(text, bbox, src):
    n = index.norm(text)
    return Line(text, n, index.fold(n), tuple(bbox), tuple(src))


def sheets(chunks, docs):
    grouped = {}
    for c in chunks:
        if c.get('kind') != 'block':
            continue
        for item in c.get('lines', []):
            bb = tuple(item['bbox'])
            grouped.setdefault(c['doc'], []).append(_line(item['text'], bb, ((c['id'], bb),)))
    return [Sheet(doc, tuple(docs[doc]['size']), lines,
                  median(l.bbox[3] - l.bbox[1] for l in lines))
            for doc, lines in grouped.items() if lines]


def dedupe(lines):
    return [a for a in lines if not any(len(a.text) < len(b.text) and
            ocr_ingest.inside(a.bbox, b.bbox) >= 0.6 for b in lines)]


def _match_text(text):
    chars, positions = [], []
    for i, c in enumerate(text):
        if not c.isspace():
            chars.append({'（': '(', '）': ')', 'O': '0', 'o': '0', 'l': '1'}.get(c, c))
            positions.append(i)
    return ''.join(chars), positions


def stitch(lines, h):
    pending = sorted(lines, key=lambda l: (l.bbox[0], l.bbox[1]))
    result = []
    while pending:
        a = pending.pop(0)
        for i, b in enumerate(pending):
            if b.bbox[0] >= a.bbox[2] or abs((a.bbox[1]+a.bbox[3]-b.bbox[1]-b.bbox[3])/2) >= 0.5*h:
                continue
            at, ai = _match_text(a.text)
            bt, bi = _match_text(b.text)
            match = None
            for k in range(min(len(at), len(bt)), 2, -1):
                for j in range(3):
                    score = fuzz.ratio(at[-k:], bt[j:j+k])
                    if j+k <= len(bt) and score >= 85:
                        match = (k, j)
                        break
                if match:
                    break
            if match:
                k, j = match
                edits = Levenshtein.editops(at[-k:], bt[j:j+k])
                if edits and edits[0].tag == 'delete' and edits[0].src_pos == 0:
                    k -= 1
                bb = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                      max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
                a = _line(a.text[:ai[-k]] + b.text[bi[j]:], bb, tuple(dict.fromkeys(a.src + b.src)))
                pending.pop(i)
                pending.insert(0, a)
                break
        else:
            result.append(a)
    return result


def cn_number(s):
    digits = {'零': 0, '一': 1, '二': 2, '兩': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    if '十' in s:
        a, b = s.split('十', 1)
        return (digits[a] if a else 1) * 10 + (digits[b] if b else 0)
    return digits.get(s)


_UNITS = r'Nm3/min|m3/人\.d|m3/m3/min|m3/min|m3/(?:day|日|天|d)|立方公尺|CMD|CMM|L/min|mmA[gq]|kg0?2|mg/[!1lL]|mH|cm|mm|HP|m3|m2|小時|分鐘|hr|h|min|吋|["″°]|只|台|組|组|套|支|座|槽|個|人|L|m'
_NUMBER = r'(?:\d[\d,]*(?:\.\d+)?|[一二兩两三四五六七八九十]+)'
_QUANTITY = re.compile(r'(?P<op>>=|<=|≧|≦|≥|≤|>|<|≈|≒|約|大於)?\s*(?P<num>' + _NUMBER + r')\s*(?P<unit>' + _UNITS + r')?', re.I)


def quantity(text):
    t = index.norm(text)
    m = _QUANTITY.search(t)
    if not m:
        return None
    n = m['num'].replace(',', '')
    value = float(n) if n[0].isdigit() else cn_number(n)
    if value is None:
        return None
    unit = m['unit'] or ''
    lower = unit.lower()
    op = {'>=':'≥', '≧':'≥', '<=':'≤', '≦':'≤', '≈':'~', '≒':'~', '約':'~', '大於':'>'}.get(m['op'], m['op'])
    if op is None:
        if '以上' in t[m.end():]: op = '≥'
        elif '以下' in t[m.end():]: op = '≤'
        elif '左右' in t[m.end():]: op = '~'
    canonical = {'cmd':'m3/d', 'm3/day':'m3/d', 'm3/日':'m3/d', 'm3/天':'m3/d', '立方公尺':'m3', 'cmm':'m3/min', 'nm3/min':'Nm3/min', 'mh':'m', '小時':'hr', 'h':'hr', '分鐘':'min', 'mmag':'mmAq', 'mmaq':'mmAq', 'kg02':'kgO2', 'kg2':'kgO2', 'hp':'HP'}.get(lower, unit)
    if lower.startswith('mg/'): canonical = 'mg/L'
    if unit in ('只','台','組','组','套','支','座','槽','個'): canonical = 'count'
    if unit == '人': canonical = 'person'
    if unit in ('"', '″', '°', '吋'): canonical = 'in'
    si, factor = (canonical, 1)
    if lower == 'l/min': canonical, si, factor = 'L/min', 'm3/min', .001
    elif lower == 'cm': si, factor = 'm', .01
    elif canonical == 'in': si, factor = 'mm', 25.4
    return {'op': op, 'value': value, 'unit': canonical, 'value_si': value * factor, 'unit_si': si, 'word': unit}


def equation(expr_text, printed):
    p = str(printed)
    try:
        pv = float(p.replace(',', ''))
        text = index.norm(expr_text)
        text = re.sub(r'(?i)m3/人\.d|m3/m3/min|m3/min|m3/day|m3/d|m3/日|m2|m3|kg/d|min/d|人/戶|戶|人|個|mH|hr|m', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = text.split('=')
        text = next((s for s in text if re.search(r'\d', s) and re.search(r'[+*/xX×÷-]', s)), '')
        text = text.strip()
        trailing = re.search(r'÷\s*(\d+(?:\.\d+)?)\s*$', text)
        if trailing:
            prefix = calc.evaluate(text[:trailing.start()])
            if abs(prefix - float(trailing.group(1))) <= .02 * abs(prefix):
                text = text[:trailing.start()]
        computed = calc.evaluate(text)
        decimals = len(p.partition('.')[2])
        return {'expr': text, 'computed': computed, 'printed': pv,
                'consistent': abs(computed-pv) <= max(.005*abs(pv), 10**-decimals)}
    except (ValueError, OverflowError):
        return None


def family_of(text):
    t = index.fold(index.norm(text)).replace(' ', '')
    for family in sorted(FAMILIES, key=len, reverse=True):
        f = index.fold(family)
        for i in range(len(t) - len(f) + 1):
            candidate = t[i:i+len(f)]
            if candidate == f or (len(f) >= 3 and Levenshtein.distance(candidate, f) <= 1):
                tail = t[i+len(f):]
                series = re.match(r'\s*\(?\s*(\d)\s*[-+]', tail)
                number = FAMILIES.index(family) + 1 if family != '抽水池' else 1
                if not series or int(series.group(1)) == number:
                    return family
    return None


def row(sheet, source, entity, attr, label, q, lines, **kw):
    entity = index.norm(entity)
    key = re.sub(r'\([^)]*\)|立式|魯氏|鲁氏|定量|\s+', '', index.fold(entity))
    result = {'doc': sheet.doc, 'source': source, 'section': '', 'entity': entity,
              'family': family_of(entity), 'equipment': next ( (e for e in EQUIPMENT if index.fold(e) in index.fold(entity)), None),
              'index': None, 'ekey': key, 'attr': attr, 'label': label,
              'op': None, 'value': None, 'unit': None, 'value_si': None, 'unit_si': None,
              'text': '\n'.join(l.text for l in lines),
              'src': [{'chunk': c, 'bbox': list(b)} for c, b in dict.fromkeys(s for l in lines for s in l.src)],
              'extra': {}}
    if q:
        result.update({k: q[k] for k in ('op', 'value', 'unit', 'value_si', 'unit_si') if k in q})
    result.update(kw)
    return result


def extract(chunks, docs):
    from . import facts_calc, facts_tables, facts_labels

    result = []
    for sheet in sheets(chunks, docs):
        sheet = Sheet(sheet.doc, sheet.size, stitch(dedupe(sheet.lines), sheet.h), sheet.h)
        for parser in (facts_calc, facts_tables, facts_labels):
            rows = parser.rows(sheet)
            for row in rows:
                row['id'] = f"F{len(result) + 1:03d}"
                result.append(row)
            for row in rows:
                extra = row.get('extra', {})
                if 'member_idx' in extra:
                    extra['members'] = [rows[i]['id'] for i in extra.pop('member_idx')]
    return result
