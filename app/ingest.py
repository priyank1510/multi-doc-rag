"""Step 1 of the pipeline: turn a PDF into clean text, one entry per page.

We keep the page number so every answer can cite exactly where it came from.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    doc_id: str
    doc_name: str
    page: int  # 1-indexed, matches what a human sees in a PDF viewer
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace and re-join words hyphenated across line breaks."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "retri-\neval" -> "retrieval"
    text = re.sub(r"[ \t]+", " ", text)
    # A short line with no closing punctuation, followed by a capitalized line, is
    # a heading: give it its own paragraph so it isn't glued onto the next sentence.
    # (Wrapped body lines are left alone: they continue in lowercase.)
    text = re.sub(r"^([^\n.!?:;]{1,60})\n(?=[A-Z])", r"\1\n\n", text, flags=re.M)
    text = re.sub(r"\n{2,}", "\n\n", text)  # keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines are just line wraps
    return text.strip()


def load_pdf(path: Path, doc_id: str, doc_name: str | None = None) -> list[Page]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:  # skip blank / image-only pages
            pages.append(Page(doc_id=doc_id, doc_name=doc_name or path.name, page=i, text=text))
    return pages
