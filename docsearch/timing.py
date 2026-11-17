"""Wall-clock seconds of each indexing stage, merged into <work>/timing.json.

The file holds {"<doc>": {"<stage>": seconds}, "_meta": {"device", "engine", "recorded"}}. Stages are per document (one
page image); the index-wide embedding is stored under INDEX. A stage recorded again replaces its earlier value, so a
rebuilt document shows its latest measurement. Measuring never breaks the pipeline: a failed write is reported on stderr.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config, device

FILE = "timing.json"
META = "_meta"
INDEX = "_index"


def load(path: Path) -> dict | None:
    """The recorded timings, or None when there is no readable file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _meta() -> dict:
    try:
        dev = device.device()
    except Exception:
        dev = "unknown"
    return {"device": dev, "engine": config.OCR_ENGINE, "recorded": datetime.now().isoformat(timespec="seconds")}


def record(work_dir: Path, doc: str, name: str, seconds: float) -> None:
    try:
        path = Path(work_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = load(path) or {}
        data.setdefault(doc, {})[name] = round(seconds, 3)
        data[META] = _meta()
        tmp = path.with_name(f"{FILE}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.rename(tmp, path)
    except Exception as e:
        print(f"[timing] not recorded ({doc} {name}): {e}", file=sys.stderr, flush=True)


@contextmanager
def stage(work_dir: Path, doc: str, name: str):
    """Times the block and records it on normal completion; a block that raises records nothing."""
    t0 = time.perf_counter()
    yield
    record(work_dir, doc, name, time.perf_counter() - t0)
