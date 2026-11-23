"""Text helpers shared by sections.py and chunk.py so offsets always agree."""
from __future__ import annotations

import re

PARA = re.compile(r"\n\s*\n")


def spans(text: str, start: int, end: int, sep: re.Pattern) -> list[tuple[int, int]]:
    """Split text[start:end] on `sep`, returning whitespace-trimmed (start, end) spans."""
    out, pos = [], start
    raw = []
    for m in sep.finditer(text, start, end):
        raw.append((pos, m.start()))
        pos = m.end()
    raw.append((pos, end))
    for s, e in raw:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    return spans(text, 0, len(text), PARA)


def squash(text: str) -> str:
    """Collapse all whitespace (incl. nbsp) to single spaces, for fuzzy matching."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
