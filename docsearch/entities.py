"""Extract drawing unit identifiers and retain matching retrieval evidence."""
import re
import unicodedata

_FAMILY = r"接觸曝氣池|污泥貯留池|調整池|終沉池|消毒池|放流池|初沉池|處理槽|污水泵"
_UNIT = re.compile(rf"(?P<family>{_FAMILY})\s*(?:\(\s*(?P<paren>\d+(?:\s*-\s*\d+)?)\s*\)?|(?P<pair>\d+\s*-\s*\d+)\)?|(?P<letter>[A-Za-z])(?![A-Za-z]))?")
_GROUP = re.compile(r"第\s*(\d+)\s*組")
_ITEM = re.compile(r"(?m)^\s*\((\d+)\)\s*(?=L\s*\d)")


def question_entities(question: str) -> list:
    text = unicodedata.normalize("NFKC", question).translate(str.maketrans("‐‑‒–—−﹣", "-------"))
    entities = []
    for match in _UNIT.finditer(text):
        family = match.group("family")
        ident = match.group("paren") or match.group("pair")
        if family == "污水泵" and match.group("letter"):
            ident = match.group("letter").upper()
        entities.append((family, re.sub(r"\s+", "", ident) if ident else None))
    entities.extend(("組", m.group(1)) for m in _GROUP.finditer(text))
    return list(dict.fromkeys(entities))


def chunk_entities(text: str) -> list:
    entities = question_entities(text)
    text = unicodedata.normalize("NFKC", text)
    for item in _ITEM.finditer(text):
        families = list(_UNIT.finditer(text, 0, item.start()))
        if families:
            entities.append((families[-1].group("family"), item.group(1)))
        entities.append(("組", item.group(1)))
    return list(dict.fromkeys(entities))


def anchor(question: str, best: list, pool: list, chunks: list, top: int) -> list:
    wanted = {entity for entity in question_entities(question) if entity[1] is not None}
    if not wanted:
        return best
    matching = {i for i in set(best) | set(pool) if wanted.intersection(chunk_entities(chunks[i]["text"]))}
    if not matching:
        return best
    result = list(best)
    count = sum(i in matching for i in result)
    for i in pool:
        if count >= 3:
            break
        if i not in matching or i in result:
            continue
        if len(result) >= top:
            removable = next((j for j in range(len(result) - 1, -1, -1) if result[j] not in matching), None)
            if removable is None:
                break
            result.pop(removable)
        result.append(i)
        count += 1
    return result
