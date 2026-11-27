"""Equipment named in a question that no OCR text in the index carries (the NOT FOUND guard).

The 4B LLM answers such questions with another item's value (消防泵浦 was answered with a 1HP symbol). `check` finds the
equipment words (pumps, blowers, fans, dosing machines, meters, mixers) in a question and returns the first one that no
chunk of the index contains, with the OCR lines nearest to it. Pipes and tanks are not equipment here: OCR garbles their
names, so a missing pipe or tank name must not trigger the guard.
"""
from __future__ import annotations

import re

from .index import fold, norm

SUFFIXES = ("泵浦", "泵", "鼓風機", "抽風機", "風機", "加藥機", "流量計", "攪拌機")
_FORMS = sorted({*SUFFIXES, *(fold(s) for s in SUFFIXES)}, key=len, reverse=True)   # both scripts, longest first
# The CJK run before an equipment word, lazily: the head stops at the first word, so 調整池泵浦和終沉池泵浦 gives two mentions.
_MENTION = re.compile(rf"(?P<head>[\u4e00-\u9fff]*?)(?P<suffix>{'|'.join(_FORMS)})")


def _key(s: str) -> str:
    """Comparison form: simplified, no spaces, lower case, 泵浦 folded to 泵 (OCR drops 浦 and the drawings use both)."""
    return fold(norm(s)).replace(" ", "").replace("泵浦", "泵").lower()


def _mentions(question: str) -> list[tuple[str, list[str]]]:
    """Per equipment word: (head, candidates). Candidates are the word with the last 2..n head characters before it, shortest
    first, so the leading question words (請問, 初沉至) are dropped: 請問調整池泵浦 -> 調整池泵浦. A bare suffix is never a
    candidate: it needs at least two characters before it."""
    out = []
    for m in _MENTION.finditer(norm(question)):
        head = m["head"]
        cands = [head[i:] + m["suffix"] for i in range(len(head) - 2, -1, -1)]
        if cands:
            out.append((head, cands))
    return out


def _near1(pattern: str, text: str) -> bool:
    """True when some substring of `text` is within edit distance 1 of `pattern` (Sellers' approximate substring match)."""
    prev = list(range(len(pattern) + 1))
    for ch in text:
        cur = [0]
        for i in range(1, len(pattern) + 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (pattern[i - 1] != ch)))
        if cur[-1] <= 1:
            return True
        prev = cur
    return False


def _present(name: str, texts: list[str]) -> bool:
    """The name is in the OCR text exactly, or within one edit of it when the name has at least 4 characters (a misread
    character). Lenient on purpose: a false NOT FOUND is worse than a missed one, which the LLM path then answers as before."""
    key = _key(name)
    fuzzy = len(key) >= 4
    halves = (key[:len(key) // 2], key[len(key) // 2:])   # one edit cannot break both halves, so one half is always intact
    for t in texts:
        if key in t:
            return True
        if fuzzy and any(h in t for h in halves) and _near1(key, t):
            return True
    return False


def _common(a: str, b: str) -> int:
    """Length of the longest common substring."""
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


def _nearest(head: str, chunks: list[dict]) -> list[tuple[str, dict]]:
    """Up to 3 distinct OCR lines sharing the most characters (>= 2) with the equipment name's head, with their chunks."""
    hk = _key(head)
    scored = []
    for c in chunks:
        for ln in c.get("lines", []):
            score = _common(hk, _key(ln["text"]))
            if score >= 2:
                scored.append((score, norm(ln["text"]), c))
    scored.sort(key=lambda t: -t[0])   # stable: ties keep index order
    out: dict[str, dict] = {}
    for _, text, c in scored:
        out.setdefault(text, c)
        if len(out) == 3:
            break
    return list(out.items())


def check(question: str, chunks: list[dict]) -> dict | None:
    """None when the question names no equipment or every equipment name is in the OCR text. Otherwise
    {'term': the absent name as written in the question, 'nearest': [(OCR line, chunk), ...]}."""
    # ponytail: folds every chunk on each call; cache the folded texts on the Engine if this ever shows up in a profile.
    texts = [_key(c["text"]) for c in chunks]
    for head, cands in _mentions(question):
        if any(_present(name, texts) for name in cands):
            continue
        return {"term": cands[0], "nearest": _nearest(head[-2:], chunks)}
    return None
