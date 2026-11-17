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

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_REFUSAL = re.compile(r"未標示|未顯示|未列出|未註明|未提供|未找到|未在[^，。,.]{0,20}找到|無法從圖面|無法確認|無任何|沒有|not\s+(?:shown|found)", re.I)
_UNITS = ("立方公尺", "m³", "m3", "hp", "HP", "馬力", "小時", "hr", "hrs", "hour", "hours", "h", "cmd", "m3/日", "m3/天", "m3/day", "m3/d", "台", "組", "槽", "只")
_CITE = re.compile(r"(?i)\s*(chunk\s*ids?(\s+used)?\s*[:：]|#\d+\s*$|C\.D棟)")
_CONT = re.compile(r"(?:[。.！!？?；;]\s*|\s+)(?=The\s|Chunk|According|After reviewing|However|however|但|在提供的|根據\s*chunk)")
_PAREN = re.compile(r"[（(][^（()）]*[）)]")
_ASIDE = re.compile(r"(?i)(?:e\.g\.?|例如|refers to|指)[^。；;\n]*")
_FORMULA_TAIL = re.compile(r"\s*[（(][^（()）]*[×x*＊＋+÷/=\-][^（()）]*[）)]\s*$")
_RELAXED = re.compile(r"(?<![\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_NEG = re.compile(r"^(?:不一致|不能|不可以|不是|不符|不相符|否|不|no\b|not\b|inconsistent\b|cannot\b)", re.I)
_POS = re.compile(r"^(?:一致|可以|能|是|符合|相符|yes\b|consistent\b|can\b)", re.I)
_WINNER = re.compile(r"污水水?泵\s*[A-Za-z]")


def _normal(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).replace("／", "/").replace(",", "").replace("約", "").replace("每台", "台").lower().replace("m³", "m3").replace("立方公尺", "m3").replace("小時", "hr").replace("hours", "hr").replace("hour", "hr").replace("hrs", "hr")


def _body(s: str) -> str:
    s = str(s)
    m = _CITE.search(s)
    if m:
        s = s[:m.start()]
    return re.sub(r"(?:\s*#\d+)+\s*$", "", s)


def _compact(s: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(s)).lower())


def _fold(unit: str) -> str:
    if unit in ("hour", "hours", "hrs", "h"):
        return "hr"
    if unit in ("cmd", "m3/日", "m3/天", "m3/day", "m3/d"):
        return "m3/day"
    if unit in ("m3", "立方公尺", "m³"):
        return "m3"
    if unit == "馬力":
        return "hp"
    return unit



def _tokens(s: str):
    text = _normal(s)
    return [(float(m.group().replace(",", "")), text[m.end():].lstrip()[:12]) for m in _NUMBER.finditer(text)]


def _contains(ref_token, answer: str, tolerance: bool):
    value, tail = ref_token
    text = _normal(answer)
    units = "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))
    unit = _fold(re.match(rf"(?:{units})?", tail).group())
    for match in _NUMBER.finditer(text):
        candidate = float(match.group().replace(",", ""))
        answer_unit = _fold(re.match(rf"(?:{units})?", text[match.end():].lstrip()).group())
        if unit and answer_unit and unit != answer_unit:
            continue
        if candidate == value or tolerance and value and abs(candidate - value) <= abs(value) * .01:
            return True
    return False


def _claim(body: str) -> str:
    text = re.split(r"\n\s*\n|\s{2,}", body.strip(), maxsplit=1)[0]
    m = _CONT.search(text)
    if m:
        text = text[:m.start()]
    return _ASIDE.sub(" ", _PAREN.sub(" ", text))


def _first_sentence(s: str) -> str:
    return re.split(r"[。！!？?；;\n]|\.(?=\s|$)", s.strip(), maxsplit=1)[0]


def _polarity(s: str) -> str | None:
    t = unicodedata.normalize("NFKC", str(s)).strip()
    if _NEG.match(t):
        return "neg"
    if _POS.match(t):
        return "pos"
    return None


def _drop_formula(s: str) -> str:
    return _FORMULA_TAIL.sub("", s)


def _in_order(ref_vals, text: str) -> bool:
    values = iter([float(m.group().replace(",", "")) for m in _RELAXED.finditer(_normal(text))])
    return all(any(v == x for x in values) for v in ref_vals)


def _is_dimension(ref_raw: str, refs) -> bool:
    marks = re.findall(r"\d+(?:\.\d+)?\s*(?:cm|公分)", _normal(ref_raw))
    return len(marks) >= 2 and len(marks) == len(refs)


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower().replace("污水水泵", "污水泵")


def _winner_claim(question: dict, ref_raw: str, claim: str, ref_vals) -> bool:
    text = question.get("問題", "")
    if "哪一種" not in text or "較大" not in text:
        return False
    m = _WINNER.search(ref_raw)
    if not m or _norm_name(m.group()) not in _norm_name(claim):
        return False
    return all(value in ref_vals for value, _ in _tokens(claim))


def grade(question: dict, answer: str, evidence) -> str:
    ref_raw = str(question.get("答案", ""))
    refs = _tokens(_drop_formula(ref_raw))
    body = _body(answer)
    actual = _tokens(body)
    state = question.get("答案狀態", "")
    if state == "圖面未標示":
        qnums = {value for value, _ in _tokens(question.get("問題", ""))}
        fresh = [t for t in actual if t[0] not in qnums]
        if refs:
            return "pass" if _REFUSAL.search(body) and not fresh else "review"
        return "review" if fresh else "pass" if _REFUSAL.search(body) else "review"
    claim = _claim(body)
    ref_vals = [value for value, _ in refs]
    if refs:
        if _is_dimension(ref_raw, refs):
            found = _in_order(ref_vals, claim)
        else:
            tolerance = question.get("驗證方式") in ("尺寸計算", "跨區計算")
            found = all(_contains(ref, claim, tolerance) for ref in refs)
    else:
        ref_text = _compact(ref_raw)
        if not ref_text:
            return "review"
        found = ref_text in _compact(body)
    ref_pol, ans_pol = _polarity(ref_raw), _polarity(_first_sentence(body))
    if ref_pol and ans_pol:
        if ans_pol != ref_pol:
            return "fail"
        return "review" if found or not _tokens(claim) else "fail"
    if _winner_claim(question, ref_raw, claim, ref_vals):
        return "review"
    if not found:
        return "review" if not refs else "fail"
    if state == "圖面存在差異":
        return "review"
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
    reread = f" reread={config.REREAD}" if config.REREAD != "0" else ""
    lines = ["# Benchmark summary", "", f"Config: llm={config.LLM_ID} backend={config.LLM_BACKEND} context={config.CONTEXT_MODE} top={config.TOP_K} facts={int(config.FACTS)}{reread}", "", f"Pass/fail/review: {counts['pass']}/{counts['fail']}/{counts['review']}", f"Rate: {counts['pass'] / len(results):.1%}" if results else "Rate: n/a", f"Seconds median/mean: {statistics.median(elapsed):.1f}/{statistics.mean(elapsed):.1f}" if elapsed else "Seconds median/mean: n/a", "", "## By category and difficulty", ""]
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
            llm, context = config.LLM_ID, config.CONTEXT_MODE
            fact_rows, computed = [], []
            verify = None
            reread = None
            try:
                response = engine.answer(q["問題"])
                answer, evidence = response.get("answer", ""), response.get("evidence", [])
                llm, context = response.get("llm", llm), response.get("context", context)
                fact_rows, computed = response.get("facts", []), response.get("computed", [])
                verify = response.get("verify")
                reread = response.get("reread")
                state = grade(q, answer, evidence)
            except Exception as exc:
                answer, state = f"ERROR: {exc}", "error"
            record = {"id": qid, "question": q["問題"], "reference": q.get("答案", ""), "answer": answer, "grade": state, "seconds": round(time.monotonic() - start, 3), "evidence_ids": [e.get("id") for e in evidence], "facts": [r["id"] for r in fact_rows], "computed": computed, "facts_mode": config.FACTS, "model_id": config.LLM_ID, "llm": llm, "context": context, "device": __import__("os").environ.get("DOCSEARCH_DEVICE", "auto"), "類別": q.get("類別", ""), "難度": q.get("難度", "")}
            if config.VERIFY:
                record["verify"] = verify
            if config.REREAD != "0":
                record["reread"] = reread
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(record)
            if len(results) % 10 == 0:
                _summary(results, out / "summary.md")
    _summary(results, out / "summary.md")
    return results
