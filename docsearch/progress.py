"""Best-effort progress reporting for isolated jobs."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

_COLLECTION = {'prepare': (0, 5), 'ocr': (5, 65), 'index': (65, 100)}
_QUESTION = {'retrieve': (0, 12), 'reason': (12, 95), 'answer': (95, 99)}
_last = None
_last_at = 0.0
def read_progress(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('stage') not in (*_COLLECTION, *_QUESTION):
            return None
        done, total, unit = (data.get(k) for k in ('done', 'total', 'unit'))
        if done is not None and (type(done) is not int or done < 0):
            return None
        if total is not None and (type(total) is not int or total < 0):
            return None
        if unit is not None and unit not in {'pages', 'tiles', 'chunks', 'tokens'}:
            return None
        if (done is None) != (total is None) and data['stage'] != 'reason':
            return None
        return {'stage': data['stage'], 'done': done, 'total': total, 'unit': unit}
    except (OSError, ValueError, TypeError):
        return None


def report(stage, done=None, total=None, unit=None):
    global _last, _last_at
    filename = os.environ.get('DOCSEARCH_PROGRESS_FILE')
    if not filename:
        return
    try:
        now = time.monotonic()
        current = (stage, done, total, unit)
        if _last is not None and current[0] == _last[0] and now - _last_at < .25 and done != total and done is not None:
            return
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                json.dump({'stage': stage, 'done': done, 'total': total, 'unit': unit}, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        _last, _last_at = current, now
    except OSError:
        pass


def percent_for(kind, stage, done, total):
    stages = _COLLECTION if kind == 'collection' else _QUESTION if kind == 'question' else {}
    if stage == 'ready':
        return 100
    if stage not in stages:
        return None
    low, high = stages[stage]
    if stage == 'reason':
        fraction = min(max((done or 0) / 4000, 0), 1)
    elif done is None or total is None or total <= 0:
        fraction = 0
    else:
        fraction = min(max(done / total, 0), 1)
    return min(high, max(low, int(low + (high - low) * fraction)))
