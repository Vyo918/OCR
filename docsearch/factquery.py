"""Select and compute grounded facts for a question, without a model."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from . import calc, facts, index


def _fold(s):
    return index.fold(index.norm(s or '')).replace(' ', '').lower()

PIPE_TERMS = ('管', '連通', '聯通', '連接')


@dataclass(frozen=True)
class Query:
    text: str
    family: str | None
    equipment: str | None
    indexes: tuple
    attrs: frozenset
    intent: str
    pump: str | None
    hard: tuple
    soft: tuple
    pipe: bool


def parse(question):
    q = _fold(question)
    glossary = json.loads(Path(__file__).with_name('glossary.json').read_text(encoding='utf-8'))
    candidates = []
    for group in glossary:
        family = next((f for f in facts.FAMILIES if f in group), None)
        if family:
            candidates.extend((len(_fold(term)), family) for term in group if _fold(term) in q and len(term) >= 2)
    family = max(candidates, default=(0, None))[1]
    if '泵' in q and family and _fold(family) not in q:
        family = None
    equip = [(len(_fold(e)), e) for e in facts.EQUIPMENT if _fold(e) in q]
    if '泵' in q:
        equip.append((1, '泵浦'))
    equipment = max(equip, default=(0, None))[1]
    indexes = re.findall(r'(\d)\s*[-+－]\s*(\d)', q)
    indexes = [a + '-' + b for a, b in indexes]
    if not indexes and family:
        m = re.search(re.escape(_fold(family)) + r'\((\d)\)', q)
        if m:
            indexes.append('(' + m.group(1) + ')')
    indexes += [x.upper() for x in re.findall(r'泵(?:浦)?([a-z])', q)]
    terms = [(len(_fold(term)), attr) for attr, names in facts.ATTRS.items() for term in names if _fold(term) in q]
    attrs = {a for n, a in terms if n == max((x[0] for x in terms), default=0)}
    if '水深' in q:
        attrs.update(('effective_depth', 'label_depth'))
    if '容积' in q or '体积' in q:
        attrs.update(('effective_volume', 'plan_volume'))
    if re.search(r'几[槽台只支组座个套]|多少[槽台只支组]|数量|共有|总共|一共|共几|总数', q):
        intent = 'count'
    elif re.search(r'合计|总计|加总|相加|总和|两区', q):
        intent = 'sum'
    elif re.search(r'哪.*[较最][大小多少高低]', q):
        intent = 'compare'
    elif re.search(r'是否一致|一致|相符|逐项相加', q):
        intent = 'check'
    else:
        intent = 'lookup'
    if '逐项相加' in q or '一致' in q or '相符' in q:
        intent = 'check'
    pump = None
    m = re.search(r'([\u4e00-\u9fff]+泵)', q)
    if m:
        pump = m.group(1)
        for prefix in ('的', '与', '和'):
            pump = pump.split(prefix)[-1]
    hard = tuple(x for x in ('消防', '雨水', '滞洪', '污水') if x in q)
    return Query(q, family, equipment, tuple(indexes), frozenset(attrs), intent, pump, hard,
                 tuple(_fold(x) for x in facts.SOFT_QUALIFIERS if _fold(x) in q), any(_fold(x) in q for x in PIPE_TERMS))


def _common(a, b):
    return max((len(a[i:j]) for i in range(len(a)) for j in range(i + 1, len(a) + 1) if a[i:j] in b), default=0)


def _overlap(entity, text):
    """Characters of the entity that the question shares in runs, e.g. entity 甲乙丙 in question 甲乙的丙丁 scores 2."""
    return sum(b.size for b in SequenceMatcher(None, _fold(entity), text, autojunk=False).get_matching_blocks())


def select(rows, query):
    by_id = {r.get('id'): r for r in rows}
    def member_family(r):
        members = r.get('extra', {}).get('members', [])
        return any(_fold(query.family) in _fold(by_id.get(x, {}).get('entity', '')) for x in members)
    def family_matches(r):
        if not query.family:
            return True
        entity = _fold(r.get('entity', ''))
        # Prefer explicit entity names over a parser's fuzzy family classification.
        named = [f for f in facts.FAMILIES if _fold(f) in entity]
        volume = r if r.get('source') == 'plan_volume' else next((v for v in rows if r.get('source') == 'check' and v.get('source') == 'plan_volume' and v.get('doc') == r.get('doc') and v.get('entity') == r.get('entity')), None)
        return (query.family in named if named else r.get('family') == query.family) or bool(volume and member_family(volume))
    if not query.family and not query.equipment and not query.pump and not (query.attrs & {'persons', 'sewage_flow', 'design_flow', 'water_quality'}):
        return []
    selected = [r for r in rows if family_matches(r)]
    if query.family and query.equipment:
        selected = [r for r in selected if _fold(query.family) in _fold(r.get('entity', '') + r.get('section', ''))]
    if query.equipment:
        selected = [r for r in selected if _fold(query.equipment) in _fold(r.get('equipment') or r.get('entity')) or (query.equipment == '泵浦' and '泵' in _fold(r.get('entity'))) or (query.family and query.intent == 'check' and r.get('source') == 'calc' and r.get('attr') in query.attrs)]
    if query.hard:
        selected = [r for r in selected if all(h in _fold(r.get('entity', '') + r.get('section', '')) or (h == '滞洪' and query.pump) for h in query.hard)]
    if query.pump and not query.family:
        scores = {}
        for r in selected:
            m = re.search(r'[\u4e00-\u9fff]+泵', _fold(r.get('entity', '')))
            if m:
                name = m.group()
                score = _common(query.pump, name)
                if score >= 2 and score >= .6 * min(len(query.pump), len(name)):
                    scores[id(r)] = score
        best = max(scores.values(), default=0)
        selected = [r for r in selected if scores.get(id(r), 0) == best and best]
    if query.indexes:
        selected = [r for r in selected if str(r.get('index') or '').upper() in query.indexes]
    if query.attrs:
        selected = [r for r in selected if r.get('attr') in query.attrs]
    elif query.intent == 'count' and query.equipment:
        selected = [r for r in selected if r.get('attr') == 'qty']
    if query.intent == 'count' and query.family and not query.equipment:
        selected = [r for r in selected if r.get('attr') in ('tank_dims', 'label_depth')]
        # Series includes other named tanks, but not unrelated series.
        selected += [r for r in rows if r not in selected and r.get('source') == 'plan_label' and r.get('attr') == 'label_depth' and str(r.get('index') or '').startswith('1-') and query.family == '初沉池']
    selected = [r for r in selected if r.get('source') != 'check' or not r.get('extra', {}).get('consistent', True) or query.intent == 'check']
    order = {s: i for i, s in enumerate(('calc', 'legend', 'pump_table', 'plan_label', 'plan_volume', 'check'))}
    selected.sort(key=lambda r: (-sum(x in _fold(r.get('entity', '')) for x in query.soft), -_overlap(r.get('entity', ''), query.text), order.get(r.get('source'), 99)))
    return selected[:24]


def _v(r):
    value = r.get('value')
    return (r.get('op') or '') + (calc.fmt(value) if isinstance(value, (int, float)) else str(value or '')) + (' ' + r['unit'] if r.get('unit') else '')


def compute(rows, query):
    lines = []
    if query.intent == 'count' and query.family and not query.equipment and not query.pipe:
        tanks = sorted({str(r['index']) for r in rows if r.get('source') == 'calc' and r.get('attr') == 'tank_dims' and r.get('index')})
        named = sorted({str(r['index']) for r in rows if r.get('source') == 'plan_label' and r.get('family') == query.family and r.get('index')})
        series = sorted({str(r['index']) for r in rows if r.get('source') == 'plan_label' and r.get('index')})
        for name, items in [('calc tanks', tanks), ('labels by name', named), ('label numbering series', series)]:
            if items:
                lines.append(f'{name}: {len(items)} [{", ".join(items)}]')
    if query.intent in ('sum', 'count', 'check'):
        groups = {}
        for r in rows:
            if r.get('source') != 'check' and r.get('attr') in ('qty', 'persons') and isinstance(r.get('value'), (int, float)):
                groups.setdefault((r['source'], r['attr'], r.get('unit')), []).append(r)
        for (source, attr, unit), group in groups.items():
            parts = [r for r in group if r.get('entity')] if attr == 'persons' else group
            total = next((r for r in group if not r.get('entity') and r.get('unit') == 'person'), None) if attr == 'persons' else None
            if len(parts) > 1:
                expr = '+'.join(calc.fmt(r['value']) for r in parts)
                line = f'{source} {attr}: {expr} = {calc.fmt(calc.evaluate(expr))} ({", ".join(r["id"] for r in parts)})'
                if total:
                    line += f'; printed {calc.fmt(total["value"])} ({total["id"]})'
                lines.append(line)
    if query.intent == 'compare':
        for source in dict.fromkeys(r.get('source') for r in rows):
            group = [r for r in rows if r.get('source') == source and isinstance(r.get('value_si'), (int, float))]
            if len(group) > 1:
                group.sort(key=lambda r: r['value_si'], reverse=True)
                lines.append(source + ': ' + ' > '.join(f'{r["entity"]} {_v(r)} ({r["id"]})' for r in group))
    for r in rows:
        extra = r.get('extra', {})
        if r.get('source') == 'check' and query.intent == 'check':
            lines.append(f'{r["id"]}: {extra.get("expr")} = {calc.fmt(extra["computed"])}; printed {calc.fmt(extra["printed"])} ({"consistent" if extra.get("consistent") else "differs"})')
    groups = {}
    for r in rows:
        if r.get('source') != 'check' and isinstance(r.get('value_si'), (int, float)):
            groups.setdefault((r.get('ekey'), r.get('attr'), r.get('unit_si')), []).append(r)
    for (_, attr, _), group in groups.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a.get('source') == b.get('source'):
                    continue
                x, y = a['value_si'], b['value_si']
                if abs(x-y) > (.1 if attr == 'bore' else .005) * max(abs(x), abs(y)):
                    lines.append(f'{a["entity"]} {a.get("label")}: {_v(a)} ({a["source"]}, {a["id"]}) vs {_v(b)} ({b["source"]}, {b["id"]}): the drawings give two values')
    return lines[:8]


def context(question, rows):
    query = parse(question)
    selected = select(rows, query)
    if not selected:
        return None
    computed = compute(selected, query)
    lines = ['Fact rows parsed by code from the OCR text (values as printed; [Fnnn] row id, then source chunk id):']
    for r in selected:
        chunks = ', '.join(s['chunk'] for s in r.get('src', []))
        lines.append(f'[{r["id"]}] {r.get("entity", "")} | {r.get("label", "")} = {_v(r)} | {r.get("source", "")}: {r.get("section", "")} | {chunks} | printed: {r.get("text", "")}')
    if computed:
        lines.append('Computed by code from these rows (exact):')
        lines.extend('- ' + line for line in computed)
    return {'rows': selected, 'computed': computed, 'text': '\n'.join(lines)}
