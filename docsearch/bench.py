"""Benchmark: run the reference questions through the system and compare with the human reference answer.

questions.json = [{id, question, reference, facts, forbidden}]
  facts      list; every entry must be present in the answer. An entry is a string, a list of alternative strings
             (any one counts) or {"re": pattern}. Matching is done on a squashed copy of the answer (NFKC, no spaces,
             lower case, Traditional folded to Simplified, curly quotes -> '"', "x"/"×" -> "x"), so write patterns
             against that form.
  forbidden  list of regex; none may match (e.g. a fire-pump rating for a question whose true answer is "not in the drawing").
The automatic check is a coarse guard; the report puts the human reference and the system answer side by side for reading.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config
from .index import fold, norm

_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2033": '"', "\u2032": "'", "\u2019": "'", "\u2018": "'", "\uff02": '"'})


def squash(s: str) -> str:
    return fold(norm(s.translate(_QUOTES))).lower().replace("''", '"').replace(" ", "").replace(",", "")


def has_fact(answer: str, fact) -> bool:
    a = squash(answer)
    if isinstance(fact, dict):
        return re.search(fold(fact["re"]).lower(), a) is not None
    return any(squash(f) in a for f in ([fact] if isinstance(fact, str) else fact))


def label(fact) -> str:
    if isinstance(fact, dict):
        return "re:" + fact["re"]
    return fact if isinstance(fact, str) else " | ".join(fact)


def run(questions_file: Path, engine) -> Path:
    qs = json.loads(questions_file.read_text())
    config.BENCH_DIR.mkdir(parents=True, exist_ok=True)
    results, md = [], ["# Benchmark: human reference answer vs system answer\n"]
    for q in qs:
        t0 = time.time()
        r = engine.answer(q["question"])
        checks = {label(f): has_fact(r["answer"], f) for f in q.get("facts", [])}
        bad = [p for p in q.get("forbidden", []) if re.search(fold(p).lower(), squash(r["answer"]))]
        r.update(id=q["id"], reference=q["reference"], fact_checks=checks, forbidden_hits=bad, seconds=round(time.time() - t0))
        print(f"[{q['id']}] {r['seconds']}s  key facts {sum(checks.values())}/{len(checks)}  forbidden {len(bad)}", flush=True)
        results.append(r)
        ok = sum(checks.values())
        md += [f"## {q['id']}. {q['question']}\n",
               f"**Reference (human, read from the drawings)**\n\n{q['reference']}\n",
               f"**System answer** ({r['seconds']} s)\n\n{r['answer']}\n",
               f"**Key facts in system answer: {ok}/{len(checks)}** - " + ", ".join(f"{'OK' if v else 'MISSING'} `{f}`" for f, v in checks.items()),
               *([f"\n**Forbidden claim present:** {bad}"] if bad else []),
               "\n**Top evidence**\n" + "\n".join(f"- `{e['id']}` @ {e['bbox']}: {e['text'].splitlines()[0][:90]}" for e in r["evidence"][:6]) + "\n"]
    out = config.BENCH_DIR / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (config.BENCH_DIR / "report.md").write_text("\n".join(md))
    tot = sum(sum(r["fact_checks"].values()) for r in results)
    n = sum(len(r["fact_checks"]) for r in results)
    print(f"\nbenchmark: {tot}/{n} key facts found, {sum(bool(r['forbidden_hits']) for r in results)} forbidden claims -> {config.BENCH_DIR / 'report.md'}")
    return out
