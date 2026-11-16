"""Extract plan labels and volumes.

Plan volume ``extra.member_idx`` contains indices into this function's result;
the caller resolves them to fact IDs after numbering all rows.
"""
import re

from . import facts, index


_LABEL = re.compile(r'^([\u3400-\u9fff]{3,7})(?:\s*\(?\s*(\d)?\s*([-+])\s*(\d)\s*\)?|\s*\(?\s*(\d)\s*\)?)?\s*\)?(?:\s*H\s*=\s*\d+(?:\.\d+)?\s*m)?$', re.I)
_SPEC = re.compile(r'^([HW])\s*[=\-]\s*(\d+(?:\.\d+)?)\s*m$', re.I)
_FORMULA = re.compile(r'([\d.]+(?:\s*[xX×*]\s*[\d.]+)+(?:\s*\+\s*[\d.]+(?:\s*[xX×*]\s*[\d.]+)+)*)\s*=\s*([\d.]+)\s*m3', re.I)
_PUMP = re.compile(r'^污水泵\s*([A-Z])$', re.I)
_HP = re.compile(r'^(\d+)\s*HP\s*[xX×]\s*(\d+)$', re.I)


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) / max(1, min(a[2]-a[0], b[2]-b[0]))


def _distance(a, b):
    return ((a[0]+a[2]-b[0]-b[2]) ** 2 + (a[1]+a[3]-b[1]-b[3]) ** 2) ** .5


def rows(sheet):
    lines = facts.stitch(facts.dedupe(sheet.lines), sheet.h)
    labels, specs, pumps, hps, titles, formulas = [], [], [], [], [], []
    for line in lines:
        text = line.norm.strip()
        if 'PVC' in text.upper():
            continue
        match = _LABEL.fullmatch(text)
        if match:
            name = match[1]
            family = next((f for f in facts.FAMILIES if index.fold(f) == index.fold(name)), None)
            if family and match[2]:
                expected = facts.FAMILIES.index(family) + 1 if family != '抽水池' else 1
                if int(match[2]) != expected:
                    family = None
            elif not family:
                family = facts.family_of(name + (match[2] + '-' if match[2] else ''))
            if family and len(name) == len(family):
                number = facts.FAMILIES.index(family) + 1 if family != '抽水池' else 1
                idx = (f'{match[2] or number}-{match[4]}' if match[3] else match[5])
                labels.append((line, family, idx))
        if _SPEC.fullmatch(text):
            specs.append(line)
        if _PUMP.fullmatch(text):
            pumps.append(line)
        if _HP.fullmatch(text):
            hps.append(line)
        if '平面圖' in text:
            titles.append(line)
        if _FORMULA.search(text):
            formulas.append(line)

    pairs = []
    for i, (label, _, _) in enumerate(labels):
        for j, spec in enumerate(specs):
            if label.bbox[1] - .3*sheet.h <= spec.bbox[1] <= label.bbox[3] + 1.5*sheet.h and _overlap(label.bbox, spec.bbox) >= .3:
                pairs.append((_distance(label.bbox, spec.bbox), i, j))
    assigned, used = {}, set()
    for _, i, j in sorted(pairs):
        if i not in assigned and j not in used:
            assigned[i] = specs[j]
            used.add(j)

    result, label_positions = [], []
    for i, (line, family, idx) in enumerate(labels):
        inline = re.search(r'H\s*=\s*(\d+(?:\.\d+)?)\s*m', line.norm, re.I)
        spec = assigned.get(i)
        # A bare family name needs a nearby spec to distinguish it from prose.
        if idx is None and not inline and spec is None:
            continue
        entity = family + (f'({idx})' if idx else '')
        label_positions.append((len(result), line))
        result.append(facts.row(sheet, 'plan_label', entity, 'label', '標籤', None, [line], family=family, index=idx))
        if inline or spec:
            m = _SPEC.fullmatch(spec.norm.strip()) if spec and not inline else None
            letter, value = (m[1].upper(), m[2]) if m else ('H', inline[1])
            result.append(facts.row(sheet, 'plan_label', entity, 'label_depth' if letter == 'H' else 'label_width', letter, facts.quantity(value + 'm'), [line] + ([spec] if m else []), family=family, index=idx))

    candidates = sorted((_distance(p.bbox, h.bbox), i, j) for i, p in enumerate(pumps) for j, h in enumerate(hps) if p.bbox[1] <= h.bbox[1] <= p.bbox[3] + 1.5*sheet.h and _overlap(p.bbox, h.bbox) >= .3)
    used_p, used_h = set(), set()
    for _, i, j in candidates:
        if i in used_p or j in used_h:
            continue
        used_p.add(i); used_h.add(j)
        pump, hp = pumps[i], hps[j]
        letter = _PUMP.fullmatch(pump.norm.strip())[1].upper()
        match = _HP.fullmatch(hp.norm.strip())
        entity = '污水泵' + letter
        for attr, label, val in [('hp', '馬力', match[1] + 'HP'), ('qty', '數量', match[2] + '台')]:
            result.append(facts.row(sheet, 'plan_label', entity, attr, label, facts.quantity(val), [pump, hp], index=letter, equipment='泵浦'))

    for title in sorted(titles, key=lambda l: (l.bbox[1], l.bbox[0])):
        possible = [f for f in formulas if title.bbox[1] <= f.bbox[1] <= title.bbox[3] + 1.5*sheet.h and _overlap(title.bbox, f.bbox) >= .3]
        if not possible:
            continue
        formula = min(possible, key=lambda f: _distance(title.bbox, f.bbox))
        m = _FORMULA.search(formula.norm)
        check = facts.equation(m[1], m[2])
        if not check:
            continue
        name = re.sub(r'[Xx×]\s*\d{1,2}', '', title.norm).strip()
        members = []
        for pos, label in label_positions:
            below = [t for t in titles if t.bbox[1] >= label.bbox[3]]
            if below and min(below, key=lambda t: (t.bbox[1], _distance(label.bbox, t.bbox))) is title:
                members.append(pos)
        result.append(facts.row(sheet, 'plan_volume', name, 'plan_volume', '容積', facts.quantity(m[2] + 'm3'), [title, formula], section=name, extra={'expr': m[1], 'member_idx': members}))
        result.append(facts.row(sheet, 'check', name, 'plan_volume', '容積', facts.quantity(m[2] + 'm3'), [title, formula], section=name, extra=check))
    return result
