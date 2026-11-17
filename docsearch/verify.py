"""Answer check for the LLM answer, with at most one retry (Engine.answer; flag DOCSEARCH_VERIFY=1, config.VERIFY, default off).

The check is pure code and runs on the raw model text, before calc.fill:
- unsupported_number: a number in the answer appears in none of the evidence chunks, fact rows or computed lines. A number the
  calculator produced ({{expr}}, or a plain `a x b = c` chain) is supported when all of its inputs are.
- refusal: the answer says NOT FOUND (or a Chinese equivalent).
- ok: anything else.

The retry (Engine._verify) is at most one extra LLM call:
- unsupported_number: same evidence, plus a note naming the numbers; the model answers again or says NOT FOUND.
- refusal: only when the question names a unit id or a number with a unit that the corpus contains and the retrieved evidence does
  not (missing()). The evidence is then widened by the section chunk of the first evidence chunk and the next 12 reranked chunks.
  Otherwise the refusal stands, so genuinely unanswerable questions stay refused.

Assumptions and ceilings:
- Numbers are compared as values after facts.quantity canonicalises the unit: cm, mm and in compare in metres, L/min in m3/min,
  CMD in m3/d. Two numbers with units match only when their SI units agree; a unit-less number matches on its written value alone.
- A number written with d decimals matches evidence within half a unit of its last digit; an integer must match exactly.
  Fractions (3/4) are read as decimals. 噸 is not converted to kg, so 噸 only matches 噸.
- Chunk ids (D#3), fact row ids (F001), citation lines and list enumerators ("1.") are not numbers.
- Chunk headers (sheet position, title) count as evidence, so a header number can support an answer number.
- Constants inside a calculator expression (0.785 in 0.785*D*D*H) are inputs and must appear in the evidence.
- Plain chains are recognised only for calc._PLAIN (no unit between the factors).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import calc, facts, index
from .entities import chunk_entities, question_entities

EPS = 1e-9
_UNITS = facts._UNITS + "|公噸|噸"   # facts has no tonne; kept as its own unit word (no kg conversion)
_IDS = re.compile(r"[^\s,，、;；()（）\[\]]+#\d+|(?<![0-9A-Za-z])F\d{3,}(?!\d)")
_ENUM = re.compile(r"^\s*\d{1,2}[.)、]\s+")
_LABEL = re.compile(r"^\W*(?:chunks?|sources?|citations?|references?|evidence|引用|來源|依據|參考|證據)\b", re.I)
_FRACTION = re.compile(r"(?<![0-9A-Za-z./])(\d+)/(\d+)(?![0-9./])")
_NUMBER = re.compile(
    rf"(?<![0-9.,])(?:(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*(?P<unit>{_UNITS}))?"
    rf"|(?P<cn>[一二兩两三四五六七八九十]+)\s*(?P<cunit>{_UNITS}))", re.I)
_REFUSAL = re.compile(r"not\s*found|未找到|找不到|查無|查不到|無法找到|沒有找到", re.I)


@dataclass(frozen=True)
class Num:
    text: str                        # as written
    value: float                     # in the written unit
    unit: str                        # canonical unit from facts.quantity, '' when none
    si: float                        # value in the SI unit `unit_si` (lengths in metres)
    unit_si: str
    tol: float = 0.0                 # half a unit of the last written digit (written unit); 0 = exact
    derived: tuple | None = None     # calculator result: the input numbers it was computed from


def _body(text: str) -> str:
    """What the reader sees: the text after the last </think>, before any unclosed <think>."""
    return text.rsplit("</think>", 1)[-1].split("<think>", 1)[0]


def _final(text: str) -> str:
    """The answer without citation lines (`Chunks used: D#1, D#2`), chunk and row ids, and list enumerators."""
    lines = []
    for line in _body(text).splitlines():
        line = index.norm(line)
        if _LABEL.match(line) and not re.search(r"\d", _IDS.sub(" ", line)):
            continue
        lines.append(_ENUM.sub("", line))
    return "\n".join(lines)


def _decimal(m: re.Match) -> str:
    n, d = int(m[1]), int(m[2])
    return m[0] if d == 0 else f"{n / d:.6f}".rstrip("0").rstrip(".")


def _si(value_si: float, unit_si: str) -> tuple[float, str]:
    """facts.quantity keeps mm as its SI unit; lengths compare in metres so that cm, mm and in agree."""
    return (value_si / 1000, "m") if unit_si == "mm" else (value_si, unit_si)


def _quantities(text: str) -> list[Num]:
    """Every number in `text`, with its unit canonicalised by facts.quantity (the same normaliser the fact rows use)."""
    s = _FRACTION.sub(_decimal, _IDS.sub(" ", index.norm(text)))
    out = []
    for m in _NUMBER.finditer(s):
        if m["num"]:
            value, unit = m["num"].replace(",", ""), m["unit"] or ""
            dec = len(value.partition(".")[2])
        else:
            try:
                number = facts.cn_number(m["cn"])
            except KeyError:                       # malformed numeral run such as 二十一十
                continue
            if number is None:
                continue
            value, unit, dec = str(number), m["cunit"], 0
        q = facts.quantity(f"{value} {unit}")
        si, unit_si = _si(q["value_si"], q["unit_si"])
        out.append(Num(m.group(0).strip(), float(value), q["unit"], si, unit_si, 0.5 * 10 ** -dec if dec else 0.0))
    return out


def numbers_in(text: str) -> list[Num]:
    """Numbers the reader sees in an answer. A calculator expression contributes its input numbers plus its result, which is
    derived from them; a plain `a x b = c` chain the same way (calc.fill would recompute a wrong c)."""
    body = _final(text)
    out: list[Num] = []
    for m in calc._CALC.finditer(body):
        inputs = _quantities(m.group(1))
        out += inputs
        try:
            value = calc.evaluate(m.group(1))
        except ValueError:
            continue
        out.append(Num(m.group(1).strip(), value, "", value, "", 0.0, tuple(inputs)))
    body = calc._CALC.sub(" ", body)
    for m in calc._PLAIN.finditer(body):
        inputs = _quantities(m.group(1))
        value = float(m.group(2))
        out += inputs + [Num(m.group(2), value, "", value, "", 0.0, tuple(inputs))]
    body = calc._PLAIN.sub(" ", body)
    return out + _quantities(body)


def _scale(n: Num) -> float:
    return n.si / n.value if n.value else 1.0


def _same(a: Num, e: Num) -> bool:
    """Does evidence number `e` state the answer number `a`? Units must agree when both have one; the tolerance is the answer's own
    rounding, so an answer written as 19.05 matches 19.049 and an answer written as 237 does not match 237.37."""
    if a.unit_si and e.unit_si:
        return a.unit_si == e.unit_si and abs(a.si - e.si) <= a.tol * _scale(a) + EPS
    return abs(a.value - e.value) <= a.tol + EPS


def _row(r: dict) -> Num:
    si, unit_si = _si(float(r.get("value_si", r["value"])), r.get("unit_si") or "")
    return Num(f"{r['value']} {r.get('unit') or ''}".strip(), float(r["value"]), r.get("unit") or "", si, unit_si)


def supported(answer_numbers: list[Num], evidence_texts: list[str], fact_rows=(), computed_lines=()) -> list[str]:
    """The texts of the answer numbers that no evidence supports (empty list: all supported)."""
    evidence = [n for t in evidence_texts for n in _quantities(t)]
    evidence += [_row(r) for r in fact_rows if isinstance(r.get("value"), (int, float))]
    evidence += [n for line in computed_lines for n in _quantities(line)]

    def ok(n: Num) -> bool:
        if n.derived is not None:
            return all(ok(i) for i in n.derived)
        return any(_same(n, e) for e in evidence)
    return list(dict.fromkeys(n.text for n in answer_numbers if not ok(n)))


def refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(_body(answer)))


def check(answer: str, evidence_texts: list[str], fact_rows=(), computed_lines=()) -> dict:
    """{verdict: ok | unsupported_number | refusal, unsupported: [texts]} for the raw model answer."""
    if refusal(answer):
        return {"verdict": "refusal", "unsupported": []}
    missed = supported(numbers_in(answer), evidence_texts, fact_rows, computed_lines)
    return {"verdict": "unsupported_number" if missed else "ok", "unsupported": missed}


def missing(question: str, corpus: list[str], evidence: list[str]) -> list[str]:
    """The names the question asks about that the corpus contains and the evidence does not: unit ids ('調整池 A') and numbers
    with a unit ('237.37 m3'). Unit-less numbers are not names (too common to mean anything)."""
    ents = [e for e in question_entities(question) if e[1] is not None]
    qty = [n for n in _quantities(question) if n.unit_si]

    def names(text: str) -> set[int]:
        found = set(chunk_entities(text)) & set(ents)
        nums = _quantities(text)
        return {i for i, e in enumerate(ents) if e in found} | {len(ents) + j for j, n in enumerate(qty)
                                                               if any(_same(n, c) for c in nums)}
    in_corpus: set[int] = set()
    for t in corpus:
        in_corpus |= names(t)
    in_evidence: set[int] = set()
    for t in evidence:
        in_evidence |= names(t)
    labels = [f"{fam} {ident}" for fam, ident in ents] + [n.text for n in qty]
    return [labels[i] for i in sorted(in_corpus - in_evidence)]


def note(unsupported: list[str]) -> str:
    """The suffix that asks the model to re-check the numbers it could not support."""
    return ("\n\nNote: the previous answer stated these numbers, which do not appear in the evidence chunks or fact rows: "
            + ", ".join(unsupported) + ". Re-check them against the evidence only, then answer again, or say NOT FOUND if the "
            "evidence does not support them.")
