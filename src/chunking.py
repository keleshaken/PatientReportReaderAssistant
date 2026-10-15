"""Section-aware splitting with stable source-line references."""
from __future__ import annotations

import re
from dataclasses import dataclass


HEADINGS = re.compile(
    r"^(?:diagnos(?:is|es)|discharge diagnosis|chief complaints?|medications?|"
    r"prescriptions?|investigations?|tests?(?: and results?)?|lab(?:oratory)? results?|"
    r"follow[ -]?up|advice|instructions?|warning signs?|when to return|"
    r"diet(?: and activity)?|activity|clinical summary|patient details?)\s*:?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class SourceLine:
    id: str
    page: int
    section: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    section: str
    text: str
    line_ids: tuple[str, ...]


def _tokens(text: str) -> int:
    """Conservative approximation, used without an extra tokenizer download."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def split_pages(pages: list[PageText], max_tokens: int = 400, overlap: int = 50) -> tuple[list[Chunk], dict[str, SourceLine]]:
    """Split at headings and pages, then at source lines near the token limit."""
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("Invalid chunk size or overlap")
    chunks: list[Chunk] = []
    source_lines: dict[str, SourceLine] = {}
    for page in pages:
        section = "General"
        groups: list[tuple[str, list[SourceLine]]] = []
        current: list[SourceLine] = []
        for number, raw in enumerate(page.text.splitlines(), 1):
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                continue
            if HEADINGS.fullmatch(line):
                if current:
                    groups.append((section, current))
                section = line.rstrip(":").strip()
                current = []
                continue
            item = SourceLine(f"p{page.page}-l{number}", page.page, section, line)
            source_lines[item.id] = item
            current.append(item)
        if current:
            groups.append((section, current))
        for section_name, lines in groups:
            start = 0
            while start < len(lines):
                end = start
                count = 0
                while end < len(lines) and (count + _tokens(lines[end].text) <= max_tokens or end == start):
                    count += _tokens(lines[end].text)
                    end += 1
                selected = lines[start:end]
                chunk_id = f"p{page.page}-c{len(chunks) + 1}"
                chunks.append(Chunk(chunk_id, page.page, section_name,
                                    "\n".join(f"[{x.id}] {x.text}" for x in selected),
                                    tuple(x.id for x in selected)))
                if end == len(lines):
                    break
                back = 0
                new_start = end
                while new_start > start + 1 and back < overlap:
                    new_start -= 1
                    back += _tokens(lines[new_start].text)
                start = new_start
    return chunks, source_lines
