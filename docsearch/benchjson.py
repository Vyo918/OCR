"""Benchmark the Chinese-keyed drawing question set."""
from __future__ import annotations

import json
import random
import re
import statistics
import time
import unicodedata
import os
from collections import Counter, defaultdict
from pathlib import Path

from . import config

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_REFUSAL = re.compile(r"未標示|未顯示|無法從圖面|沒有|not\s+shown", re.I)
_UNITS = ("立方公尺", "m³", "m3", "hp", "HP", "馬力", "小時", "hr", "hrs", "hour", "hours", "h", "cmd", "m3/日", "m3/天", "m3/day", "m3/d", "台", "組", "槽", "只")


def _normal(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).replace("／", "/").replace(",", "").replace("約", "").lower().replace("m³", "m3").replace("立方公尺", "m3").replace("小時", "hr").replace("hours", "hr").replace("hour", "hr").replace("hrs", "hr")


def _tokens(s: str):
    text = _normal(s)
    units = "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))
    return [(float(m.group().replace(",", "")), text[m.end():].lstrip()[:12]) for m in _NUMBER.finditer(text)]


def _contains(ref_token, answer: str, tolerance: bool):
    value, tail = ref_token
    text = _normal(answer)
    units = "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))
    unit = re.match(rf"(?:{units})?", tail).group()
    for match in _NUMBER.finditer(text):
        candidate = float(match.group().replace(",", ""))
        answer_unit = re.match(rf"(?:{units})?", text[match.end():].lstrip()).group()
        ref_folded = "hr" if unit in ("hour", "hours", "hrs", "h") else "m3/day" if unit in ("cmd", "m3/日", "m3/天", "m3/day", "m3/d") else "m3" if unit in ("m3", "立方公尺", "m³") else unit
        answer_folded = "hr" if answer_unit in ("hour", "hours", "hrs", "h") else "m3/day" if answer_unit in ("cmd", "m3/日", "m3/天", "m3/day", "m3/d") else "m3" if answer_unit in ("m3", "立方公尺", "m³") else answer_unit
        if unit and answer_unit and ref_folded != answer_folded:
            continue
        if candidate == value or tolerance and value and abs(candidate - value) <= abs(value) * .01:
            return True
    return False


def grade(question: dict, answer: str, evidence) -> str:
    refs = _tokens(question.get("答案", ""))
    actual = _tokens(answer)
    state = question.get("答案狀態", "")
    if state == "圖面未標示":
        if refs:
            return "pass" if _REFUSAL.search(answer) and not actual else "review"
        return "review" if actual else "pass" if _REFUSAL.search(answer) else "review"
    if not refs:
        return "review"
    tolerance = question.get("驗證方式") in ("尺寸計算", "跨區計算")
    found = all(_contains(ref, answer, tolerance) for ref in refs)
    if state == "圖面存在差異":
        return "review" if found else "fail"
    if not found:
        return "fail"
    return "pass" if evidence else "review"


def select_questions(questions: list[dict], subset: int | None, seed: int = 0) -> list[dict]:
    if subset is None or subset >= len(questions):
        return list(questions)
    mandatory = [q for q in questions if q.get("答案狀態") != "有依據" or q.get("難度") in ("計算", "整合", "辨識")]
    chosen = list(mandatory)
    remaining = [q for q in questions if q not in mandatory]
    slots = max(0, subset - len(chosen))
    groups = defaultdict(list)
    for q in remaining:
        groups[q.get("類別", "")].append(q)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Largest-remainder allocation gives deterministic proportional category coverage.
    total = len(remaining)
    allocations = {k: min(len(v), slots * len(v) // total) for k, v in groups.items()} if total else {}
    left = slots - sum(allocations.values())
    order = sorted(groups, key=lambda k: (-(slots * len(groups[k]) % total), str(k))) if total else []
    for key in order:
        if left and allocations[key] < len(groups[key]):
            allocations[key] += 1
            left -= 1
    chosen.extend(q for key, group in groups.items() for q in group[:allocations[key]])
    return sorted(chosen, key=lambda q: str(q.get("編號", "")))


def _summary(results, path):
    counts = Counter(r["grade"] for r in results)
    elapsed = [r["seconds"] for r in results]
    lines = ["# Benchmark summary", "", f"Pass/fail/review: {counts['pass']}/{counts['fail']}/{counts['review']}", f"Rate: {counts['pass'] / len(results):.1%}" if results else "Rate: n/a", f"Seconds median/mean: {statistics.median(elapsed):.1f}/{statistics.mean(elapsed):.1f}" if elapsed else "Seconds median/mean: n/a", "", "## By category and difficulty", ""]
    for field in ("類別", "難度"):
        lines.append(f"### {field}")
        groups = defaultdict(list)
        for r in results: groups[r.get(field, "")].append(r)
        for key, items in sorted(groups.items()):
            c = Counter(x["grade"] for x in items)
            lines.append(f"- {key}: {c['pass']}/{c['fail']}/{c['review']} pass/fail/review ({c['pass']/len(items):.1%})")
        lines.append("")
    lines += ["## Fail and review items", ""]
    for r in results:
        if r["grade"] in ("fail", "review", "error"):
            lines += [f"### {r['id']} — {r['question']} ({r['grade']})", f"Reference: {r['reference']}", f"Answer: {r['answer']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(questions_file: Path, engine, out: Path | None = None, subset: int | None = None, seed: int = 0, ids: list[str] | None = None, resume: bool = False):
    data = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    questions = data["題目"]
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q.get("編號") in wanted]
    questions = select_questions(questions, subset, seed)
    out = Path(out) if out else config.BENCH_DIR / ("json-" + config.LLM_ID.replace("/", "-"))
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    results = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line] if resume and result_path.exists() else []
    done = {r["id"] for r in results}
    with result_path.open("a", encoding="utf-8") as stream:
        for q in questions:
            qid = q["編號"]
            if qid in done:
                continue
            start = time.monotonic()
            answer, evidence, state = "", [], "review"
            try:
                response = engine.answer(q["問題"])
                answer, evidence = response.get("answer", ""), response.get("evidence", [])
                state = grade(q, answer, evidence)
            except Exception as exc:
                answer, state = f"ERROR: {exc}", "error"
            record = {"id": qid, "question": q["問題"], "reference": q.get("答案", ""), "answer": answer, "grade": state, "seconds": round(time.monotonic() - start, 3), "evidence_ids": [e.get("id") for e in evidence], "model_id": config.LLM_ID, "device": __import__("os").environ.get("DOCSEARCH_DEVICE", "auto"), "類別": q.get("類別", ""), "難度": q.get("難度", "")}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(record)
            if len(results) % 10 == 0:
                _summary(results, out / "summary.md")
    _summary(results, out / "summary.md")
    return results
