"""Extract calculation facts in the drawing's column reading order."""
import re

from . import facts, index


def _attribute(label):
    key = index.fold(label)
    matches = [(len(index.fold(term)), attr) for attr, terms in facts.ATTRS.items()
               for term in terms if index.fold(term) in key]
    return max(matches, default=(0, 'other'))[1]


def _unit_heading(text):
    return bool(index.H2.match(text) and any(index.fold(f) in index.fold(text) for f in facts.FAMILIES))


def _columns(sheet):
    lines = facts.stitch(facts.dedupe(sheet.lines), sheet.h)
    heads = [l for l in lines if index.H1.match(l.norm) or _unit_heading(l.norm)]
    if not any('計算書' in l.norm for l in lines) and sum(_unit_heading(l.norm) for l in heads) < 3:
        return []
    anchors = []
    for x in sorted(l.bbox[0] for l in heads):
        if not anchors or x - anchors[-1][-1] > 10 * sheet.h:
            anchors.append([x])
        else:
            anchors[-1].append(x)
    starts = [min(a) for a in anchors]
    result = []
    for i, x in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else float('inf')
        col = sorted((l for l in lines if x-sheet.h <= l.bbox[0] <= x+8*sheet.h and l.bbox[2] < end), key=lambda l: (l.bbox[1], l.bbox[0]))
        # Keep runs containing headings; short excerpts can have intentional gaps.
        runs = []
        for line in col:
            if not runs or line.bbox[1] - max(l.bbox[3] for l in runs[-1]) > 3*sheet.h:
                runs.append([])
            runs[-1].append(line)
        selected = [n for n, run in enumerate(runs) if any(l in heads or index.H3.match(l.norm) or index.H4.match(l.norm) for l in run)]
        if selected:
            col = [l for run in runs[min(selected):max(selected)+1] for l in run]
        else:
            col = []
        result.extend(col)
    return result


def rows(sheet):
    ordered = _columns(sheet)
    if not ordered:
        return []
    result = []
    path = ['', '', '', '']
    entity = ''
    use = ''
    volumes = []
    previous = None

    def add(attr, label, q, lines, name=None, source='calc', **kw):
        result.append(facts.row(sheet, source, name if name is not None else entity, attr, label, q, lines,
                                section=' > '.join(p for p in path if p), **kw))

    def check(text, printed, lines, name=None):
        e = facts.equation(text, printed)
        if e:
            add('other', '算式檢核', facts.quantity(str(printed)), lines, name, 'check', extra=e)

    for line in ordered:
        text = line.norm.strip()
        if text.startswith('>'):
            continue
        if text.startswith('=') and previous:
            old, oldline = previous
            text = old + text
            lines = [oldline, line]
        else:
            lines = [line]
        previous = (text, line)
        level = index.heading_level(text)
        if level == 2 and not _unit_heading(text):
            level = None
        if level:
            path[level-1] = text.rstrip('：:')
            path[level:] = [''] * (4-level)
            if level == 2:
                entity = next((f for f in facts.FAMILIES if index.fold(f) in index.fold(text)), '')
                volumes = []
            if level == 1:
                entity = ''
            # Heading lines may also contain a quantity.
        m = re.search(r'\((\d)\)(.{2,6}?)部分', text)
        if m:
            use = m.group(2)
        dims = index.parse_dims(text)
        for lab, L, W, H, D in dims:
            if not entity:
                continue
            vol = L*W*D
            volumes.append(vol)
            extra = {'L': L, 'W': W, 'H': H, 'D': D, 'volume': vol}
            add('tank_dims', '設計尺寸', None, lines, index=lab, extra=extra)
            add('effective_depth', '有效水深', facts.quantity(f'{D}m'), lines, index=lab)
        if dims:
            continue
        pump = re.search(r'採用\s*([\d.]+\s*m3/min)\s*[×x]\s*([\d.]+\s*mH)\s*[×x]\s*([\d.]+\s*HP)\s*[×x]\s*([\d.一二兩三四五六七八九十]+\s*台)', text, re.I)
        if pump:
            for attr, value in zip(('flow', 'head', 'hp', 'qty'), pump.groups()):
                add(attr, attr, facts.quantity(value), lines, entity+'泵浦')
            continue
        blower = re.search(r'設置鼓風出風量\s*[:：]\s*([\d.]+\s*Nm3/min).*?p\s*=\s*([\d.]+\s*mmA[gq]).*?([\d.]+\s*MM).*?([\d.]+\s*HP)', text, re.I)
        if blower:
            for attr, value in zip(('air', 'pressure', 'bore', 'hp'), blower.groups()):
                add(attr, attr, facts.quantity(value), lines, entity+'鼓風機')
            continue
        diffuser = re.search(r'採用\s*(\d+只)(粗|細)氣泡散氣盤', text)
        if diffuser:
            add('qty', '數量', facts.quantity(diffuser[1]), lines, entity+diffuser[2]+'氣泡散氣盤')
            note = re.search(r'\((消毒池)(\d+只)\)', text)
            if note:
                add('qty', '數量', facts.quantity(note[2]), lines, note[1]+'散氣盤', extra={'note': note[0]})
            continue
        count = re.search(r'(.+?加藥機)\s*[×x]\s*(\d+\s*台)', text)
        if count:
            add('qty', '數量', facts.quantity(count[2]), lines, entity+'定量加藥機')
        per = re.search(r'(\d+\s*人)\s*[×x]\s*([\d.]+\s*m3/人[.\-]d)\s*[=≈≒]\s*([\d.]+\s*m3/(?:day|d))', text, re.I)
        if per and use:
            for attr, val in zip(('persons', 'other', 'sewage_flow'), per.groups()):
                add(attr, {'persons':'人數', 'other':'每人污水量', 'sewage_flow':'污水量'}[attr], facts.quantity(val.replace('人-d','人.d')), lines, use)
            check(text, re.search(r'[\d.]+', per[3])[0], lines, use)
            continue
        quality = any('水質' in p for p in path[:3])
        if quality:
            qm = re.search(r'(BOD|COD|SS|大腸菌類)\s*([:：<>])\s*(\d+)\s*(mg/[!1lL])?', text, re.I)
            if qm:
                add('water_quality', qm[1], facts.quantity((qm[2] if qm[2] in '<>' else '') + qm[3] + (qm[4] or '')), lines, '放流水質' if '放流' in path[2] else '原污水水質')
            continue
        if level and not re.search(r'[:：=].*\d|[×x]\s*\d', text):
            continue
        sep = re.search(r'[:：=]', text)
        if not sep:
            continue
        label = text[:sep.start()].strip()
        if label in ('V', 'DT', 'L'):
            label = path[3] or path[2] or label
        valtext = re.split(r'\.{2,}|-{2,}|0K|OK', text[sep.end():], maxsplit=1, flags=re.I)[0]
        parts = re.split(r'[<>]', valtext, maxsplit=1)
        target = parts[0]
        nums = list(re.finditer(r'\d[\d,]*(?:\.\d+)?\s*(?:Nm3/min|m3/人\.d|m3/min|m3/(?:day|d|日)|CMD|kg/d|mg/[!1lL]|m2|m3|mH|HP|hr|小時|分鐘|人|台|只|m)?', target, re.I))
        if not nums:
            continue
        chosen = nums[0] if len(parts) > 1 and not re.search(r'[=÷×+]', target) else nums[-1]
        q = facts.quantity(chosen.group())
        if not q:
            continue
        attr = _attribute(label)
        if '有效容量' in label or '實際容量' in label or ('有效容量' in path[3] and text.startswith('V=')) or ('實際容量' in path[3] and text.startswith('V=')):
            attr = 'effective_volume'
        if '污水量' in label:
            attr = 'design_flow' if '設計' in label else 'sewage_flow'
        if '使用人數' in label:
            attr = 'persons'
        add(attr, label, q, lines)
        if attr == 'effective_volume' and volumes:
            total = sum(volumes)
            add('effective_volume', '池體容積總和檢核', q, lines, source='check', extra={'expr': ' + '.join(map(str, volumes)), 'computed': total, 'printed': q['value'], 'consistent': abs(total-q['value']) <= max(.005*abs(q['value']), .01)})
        if re.search(r'[=÷×+]', valtext):
            check(text, chosen.group().strip().split()[0].rstrip('m'), lines)
    return result
