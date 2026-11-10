"""Two chunking strategies, kept side by side so the RAG project can compare them.

- fixed: slide a window of N words with overlap across the section text.
  Simple, but a chunk can cut a rule in half.
- section: pack whole paragraphs into chunks up to a word limit, and start
  every chunk with the section number and title, so each chunk still says
  which rule it belongs to when it is retrieved on its own.
"""
from dataclasses import dataclass

from .parse import Section


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    part: str
    strategy: str
    position: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def fixed_chunks(section: Section, size: int = 200, overlap: int = 40) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = section.text.split()
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(Chunk(f"fixed:{section.section_id}:{i}", section.section_id, section.part,
                            "fixed", i, " ".join(piece)))
    return chunks


def section_chunks(section: Section, max_words: int = 250) -> list[Chunk]:
    header = f"§ {section.section_id} {section.title}."
    budget = max_words - len(header.split())
    groups, current, count = [], [], 0
    for para in section.paragraphs:
        n = len(para.split())
        if current and count + n > budget:
            groups.append(current)
            current, count = [], 0
        if n > budget:
            # A single very long paragraph: split it by words.
            words = para.split()
            for s in range(0, len(words), budget):
                groups.append([" ".join(words[s:s + budget])])
            continue
        current.append(para)
        count += n
    if current:
        groups.append(current)
    return [Chunk(f"section:{section.section_id}:{i}", section.section_id, section.part,
                  "section", i, header + "\n" + "\n".join(g))
            for i, g in enumerate(groups)]


STRATEGIES = {"fixed": fixed_chunks, "section": section_chunks}
