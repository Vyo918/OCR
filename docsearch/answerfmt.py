"""Conservative formatting of answers against retrieved evidence."""
import re


_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = re.compile(rf"(?<![\w.])({_NUMBER})[ \t]*(v|hz|m3|m³)(?![\w])", re.IGNORECASE)
_RATING = re.compile(rf"(?<![\w.])({_NUMBER})[ \t]*V[ \t]+({_NUMBER})[ \t]*Hz(?![\w])", re.IGNORECASE)
_ROW = re.compile(rf"電源[ \t]+([13])[ \t]+({_NUMBER})[ \t]*V[ \t]+({_NUMBER})[ \t]*Hz(?![\w])", re.IGNORECASE)
_PHASE = re.compile(r"單相|单相|三相|single[ -]?phase|three[ -]?phase|\b[13][ -]?phase\b", re.IGNORECASE)


def normalize(answer: str, question: str, contexts: list[str]) -> str:
    """Format units and add an unambiguous phase supported by an evidence row."""
    units = {"v": "V", "hz": "Hz", "m3": "m3", "m³": "m³"}
    formatted = _UNIT.sub(lambda m: m.group(1) + (" " if m.group(2).lower().startswith("m") else "") + units[m.group(2).lower()], answer)
    if not any(word in question for word in ("電源", "電壓", "相")) or _PHASE.search(answer):
        return formatted
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
