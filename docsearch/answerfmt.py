"""Conservative formatting of answers against retrieved evidence."""
import re


_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = re.compile(rf"(?<![\w.])({_NUMBER})[ \t]*(v|hz|m3|m³)(?![\w])", re.IGNORECASE)
_RATING = re.compile(rf"(?<![\w.])({_NUMBER})[ \t]*V[ \t]+({_NUMBER})[ \t]*Hz(?![\w])", re.IGNORECASE)
_ROW = re.compile(rf"電源[ \t]+([13])[ \t]+({_NUMBER})[ \t]*V[ \t]+({_NUMBER})[ \t]*Hz(?![\w])", re.IGNORECASE)
_PHASE = re.compile(r"單相|单相|三相|single[ -]?phase|three[ -]?phase|\b[13][ -]?phase\b", re.IGNORECASE)
_UNIT_TEXT = r"[A-Za-zμ%℃°³²][A-Za-zμ%℃°³²/]*"
_VOLT = re.compile(rf"(?<![\w.])({_NUMBER})[ \t]*V(?![\w])")
_SIZE = re.compile(rf"(?<![\w.])(?:\(\d+\))?L({_NUMBER})[ \t]*[×xX][ \t]*W({_NUMBER})[ \t]*[×xX][ \t]*H({_NUMBER})[ \t]*/[ \t]*D({_NUMBER})(?!\w|\.\d)")
_QTY = re.compile(rf"(?<![\w.])({_NUMBER})[ \t]*({_UNIT_TEXT})(?!\w)")
_BOUND = re.compile(rf"([<>≦≤≧≥])[ \t]*({_NUMBER})(?!\d|\.\d)(?:[ \t]*({_UNIT_TEXT})(?!\w))?")
_COMPARATOR = re.compile(r"小於|大於|超過|低於|高於|以下|以上|[<>≦≧≤≥]")
_LIMIT_WORD = {"<": "小於", "≦": "不大於", "≤": "不大於", ">": "大於", "≧": "不小於", "≥": "不小於"}


def normalize(answer: str, question: str, contexts: list[str]) -> str:
    """Format units, add evidence-backed phase, decode tank sizes and limit comparators."""
    units = {"v": "V", "hz": "Hz", "m3": "m3", "m³": "m³"}
    formatted = _UNIT.sub(lambda m: m.group(1) + (" " if m.group(2).lower().startswith("m") else "") + units[m.group(2).lower()], answer)
    if not _PHASE.search(answer):
        if any(word in question for word in ("電源", "電壓", "相")):
            formatted = _rating_phase(formatted, contexts)
        if any(word in question for word in ("電源", "電壓")):
            formatted = _voltage_phase(formatted, contexts)
    if "尺寸" in question:
        formatted = _size(formatted)
    if "上限" in question or "下限" in question:
        formatted = _limit(formatted, contexts)
    return formatted


def _rating_phase(formatted: str, contexts: list[str]) -> str:
    ratings = list(_RATING.finditer(formatted))
    if len(ratings) != 1:
        return formatted
    rating = ratings[0]
    phases = {row.group(1) for text in contexts for row in _ROW.finditer(text)
              if row.group(2) == rating.group(1) and row.group(3) == rating.group(2)}
    if len(phases) != 1:
        return formatted
    phase = "單相" if phases.pop() == "1" else "三相"
    return formatted[:rating.start()] + phase + " " + formatted[rating.start():]


def _voltage_phase(formatted: str, contexts: list[str]) -> str:
    volts = list(_VOLT.finditer(formatted))
    if len(volts) != 1 or "hz" in formatted.lower():
        return formatted
    volt = volts[0]
    rows = {(row.group(1), row.group(3)) for text in contexts for row in _ROW.finditer(text)
            if row.group(2) == volt.group(1)}
    if len(rows) != 1:
        return formatted
    phase, hz = rows.pop()
    name = "單相" if phase == "1" else "三相"
    return formatted[:volt.start()] + f"{name} {volt.group(1)}V {hz}Hz" + formatted[volt.end():]


def _size(formatted: str) -> str:
    if all(word in formatted for word in ("長", "寬", "有效水深")):
        return formatted
    return _SIZE.sub(lambda m: f"{m.group(0)}（長 {m.group(1)} cm、寬 {m.group(2)} cm、池高 {m.group(3)} cm、有效水深 {m.group(4)} cm）", formatted)


def _limit(formatted: str, contexts: list[str]) -> str:
    quantities = list(_QTY.finditer(formatted))
    if len(quantities) != 1 or _COMPARATOR.search(formatted):
        return formatted
    qty = quantities[0]
    bounds = [m for text in contexts for m in _BOUND.finditer(text) if m.group(2) == qty.group(1)]
    same = {m.group(1) for m in bounds if m.group(3) == qty.group(2)}
    if len(same) != 1 or {m.group(1) for m in bounds} != same:
        return formatted
    return formatted[:qty.start()] + _LIMIT_WORD[same.pop()] + " " + formatted[qty.start():]
