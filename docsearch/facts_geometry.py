"""Derive individual plan volumes from measured tank outlines."""
import os
import re

import numpy as np
from PIL import Image

from . import facts, reread


def _formula(plan, members, rows):
    terms = []
    for term in plan.get('extra', {}).get('expr', '').split('+'):
        parts = re.split(r'\s*[xX×*]\s*', term.strip())
        if len(parts) != 4 or not all(re.fullmatch(r'\d+(?:\.\d+)?', p) for p in parts):
            return None
        a, b, c, count = map(float, parts)
        if min(a, b, c) <= 0 or count <= 0 or not count.is_integer():
            return None
        terms.append(([a, b, c], int(count)))
    if sum(count for _, count in terms) != len(members):
        return None
    common = set(terms[0][0]).intersection(*(set(t) for t, _ in terms[1:]))
    depths = {r['value'] for r in rows if r.get('source') == 'plan_label' and
              r.get('attr') == 'label_depth' and r.get('doc') == plan['doc'] and
              any(r.get('entity') == m.get('entity') for m in members) and r.get('value') is not None}
    if len(depths) > 1 or (depths and next(iter(depths)) not in common):
        return None
    if not depths and len(common) != 2:
        return None
    depth = next(iter(depths)) if depths else min(common)
    lengths = common - {depth}
    if len(lengths) != 1:
        return None
    length = lengths.pop()
    widths = []
    for factors, count in terms:
        factors = factors.copy()
        factors.remove(depth)
        factors.remove(length)
        widths.extend(factors * count)
    return length, depth, sorted(widths)


def _nearest(indices, centre, before):
    candidates = indices[indices < centre] if before else indices[indices > centre]
    if not len(candidates):
        return None
    return int(candidates[-1] if before else candidates[0])


def _measure(image, member):
    bbox = member['src'][0]['bbox']
    cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
    x0, x1 = max(0, cx-500), min(image.width, cx+500)
    y0, y1 = max(0, cy-500), min(image.height, cy+500)
    a = np.asarray(image.crop((x0, y0, x1, y1)).convert('RGB'))
    r, g, b = (a[:, :, i].astype(np.int16) for i in range(3))
    blue = (b > r + 12) & (b > g + 5) & (r < 180)
    black = (r < 100) & (g < 100) & (b < 100)
    red = (r > g + 40) & (r > b + 30) & (g < 160)
    wall = blue | black
    ink = wall | red
    x, y = cx-x0, cy-y0
    left, right = max(0, x-120), min(a.shape[1], x+120)
    h = np.flatnonzero((ink[:, left:right].sum(axis=1) >= (right-left)*.60) & (wall[:, left:right].sum(axis=1) >= 15))
    top, bottom = _nearest(h, y, True), _nearest(h, y, False)
    if top is None or bottom is None or bottom-top < 30:
        return None
    segment = ink[top:bottom+1]
    walls = wall[top:bottom+1]
    v = np.flatnonzero((segment.sum(axis=0) >= (bottom-top)*.60) & (walls.sum(axis=0) >= (bottom-top)*.12))
    l, r = _nearest(v, x, True), _nearest(v, x, False)
    if l is None or r is None:
        return None
    return r-l, bottom-top


def rows(existing, docs):
    if os.environ.get('DOCSEARCH_GEOMETRY', '1') == '0':
        return []
    by_id = {r['id']: r for r in existing}
    result = []
    for plan in existing:
        if plan.get('source') != 'plan_volume':
            continue
        members = [by_id.get(i) for i in plan.get('extra', {}).get('members', [])]
        if not plan.get('doc') or not members or any(m is None or m.get('doc') != plan['doc'] for m in members):
            continue
        formula = _formula(plan, members, existing)
        if formula is None:
            continue
        length, depth, widths = formula
        volumes = [length * width * depth for width in widths]
        if not plan.get('value') or abs(sum(volumes) / plan['value'] - 1) > .005:
            continue
        try:
            with Image.open(reread.resolve_image(docs[plan['doc']]['image'])) as image:
                measured = [_measure(image, member) for member in members]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if any(m is None for m in measured):
            continue
        heights = [m[1] for m in measured]
        if max(heights) / min(heights) > 1.05:
            continue
        ordered = sorted(range(len(members)), key=lambda i: (measured[i][0], i))
        if any(abs(measured[i][0] / (measured[i][1] / length) / width - 1) > .06
               for i, width in zip(ordered, widths)):
            continue
        assignments = dict(zip(ordered, widths))
        for i, member in enumerate(members):
            width = assignments[i]
            volume = width * length * depth
            extra = {'length_m': length, 'width_m': width, 'depth_m': depth,
                     'volume_m3': volume, 'basis': 'plan_geometry', 'formula_row': plan['id']}
            for attr, label, q in (('tank_dims', '平面實測尺寸', None),
                                   ('effective_volume', '平面實測有效容積', facts.quantity(f'{volume}m3'))):
                row = facts.row(facts.Sheet(plan['doc'], None, [], 0), 'plan_dims', member['entity'], attr, label, q, [],
                                family=member.get('family'), index=member.get('index'), extra=extra.copy())
                row['text'] = f'平面實測 {width:g}×{length:g}×{depth:g} m'
                row['src'] = member['src'] + plan['src']
                result.append(row)
    return result
