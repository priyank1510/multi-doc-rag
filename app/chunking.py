"""Step 2: split pages into chunks small enough to embed and retrieve precisely.

Two strategies, so we can compare them in eval/evaluate.py:
- fixed:    slide a fixed-size character window with overlap (simple baseline)
- sentence: pack whole sentences up to a size budget, carrying the last few
            sentences forward as overlap (never cuts a sentence in half).
            Sentences are joined with newlines so headings stay separable.
"""
import re
from dataclasses import dataclass

from app.ingest import Page

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str


def split_sentences(text: str) -> list[str]:
    parts = []
    for paragraph in re.split(r"\n+", text):
        parts.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return parts


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[i : i + size].strip() for i in range(0, max(len(text) - overlap, 1), step)]


def sentence_chunks(text: str, size: int, overlap: int) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + len(sentence) > size:
            chunks.append("\n".join(current))
            # Carry trailing sentences (up to `overlap` chars) into the next chunk
            # so a fact spanning a boundary is still retrievable in one piece.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, carried_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_pages(pages: list[Page], strategy: str, size: int, overlap: int) -> list[Chunk]:
    splitter = {"fixed": fixed_chunks, "sentence": sentence_chunks}[strategy]
    chunks = []
    for page in pages:
        for i, text in enumerate(splitter(page.text, size, overlap)):
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{page.doc_id}:p{page.page}:c{i}",
                        doc_id=page.doc_id,
                        doc_name=page.doc_name,
                        page=page.page,
                        text=text,
                    )
                )
    return chunks
