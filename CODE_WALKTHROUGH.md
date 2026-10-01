# Complete Code Walkthrough: Multi-Document RAG

This document explains **every file and every block of code** in the project, in the order you should build it. Each section shows the full code for one file, followed by a block-by-block explanation of what it does and why.

**How to use this document**
1. Read Part 1 (concepts) once, so the vocabulary makes sense.
2. Follow Part 2 to set up your machine.
3. In Part 3, build each file yourself: type the code (don't copy-paste), read the explanation, run the "Try it" check.
4. Part 4 traces one real question through the whole system, which ties it all together.
5. Parts 5 to 8 cover the Python features used, troubleshooting, commands, and GitHub.

---

## Table of contents
- [Part 1: Concepts you need first](#part-1-concepts-you-need-first)
- [Part 2: Setting up your machine](#part-2-setting-up-your-machine)
- [Part 3: Every file, explained](#part-3-every-file-explained)
  - [3.1 Folder structure](#31-folder-structure)
  - [3.2 `.gitignore`](#32-gitignore)
  - [3.3 `.env.example`](#33-envexample)
  - [3.4 `requirements.txt`](#34-requirementstxt)
  - [3.5 `__init__.py` files](#35-__init__py-files)
  - [3.6 `app/config.py`](#36-appconfigpy)
  - [3.7 `scripts/make_sample_pdfs.py`](#37-scriptsmake_sample_pdfspy)
  - [3.8 `app/ingest.py`](#38-appingestpy)
  - [3.9 `app/chunking.py`](#39-appchunkingpy)
  - [3.10 `app/embeddings.py`](#310-appembeddingspy)
  - [3.11 `app/vector_store.py`](#311-appvector_storepy)
  - [3.12 `eval/questions.json`](#312-evalquestionsjson)
  - [3.13 `eval/evaluate.py`](#313-evalevaluatepy)
  - [3.14 `app/llm.py`](#314-appllmpy)
  - [3.15 `app/rag.py`](#315-appragpy)
  - [3.16 `app/main.py`](#316-appmainpy)
  - [3.17 `app/static/index.html`](#317-appstaticindexhtml)
  - [3.18 `tests/conftest.py`](#318-testsconftestpy)
  - [3.19 `tests/test_chunking.py`](#319-teststest_chunkingpy)
  - [3.20 `tests/test_api.py`](#320-teststest_apipy)
- [Part 4: Following one question through the code](#part-4-following-one-question-through-the-code)
- [Part 5: Python features used, explained](#part-5-python-features-used-explained)
- [Part 6: Troubleshooting](#part-6-troubleshooting)
- [Part 7: Command cheat sheet](#part-7-command-cheat-sheet)
- [Part 8: Uploading to GitHub](#part-8-uploading-to-github)

---

## Part 1: Concepts you need first

| Term | Plain-English meaning |
|---|---|
| **LLM** | Large Language Model (like Claude). Generates text. It does **not** know what's inside your PDFs. |
| **RAG** | Retrieval-Augmented Generation. Find the relevant passages first, put them in the prompt, then have the LLM answer using only those. |
| **Chunk** | A small piece of a document (here about 250 characters, two or three sentences). We search over chunks, not whole documents. |
| **Embedding** | A list of numbers (here 384 of them) that represents the *meaning* of a text. Texts with similar meaning get similar lists. |
| **Vector** | Same thing as an embedding: a list of numbers, thought of as a point in 384-dimensional space. |
| **Cosine similarity** | A score from -1 to 1 for how similar two vectors are. 1 = same meaning, about 0 = unrelated. |
| **Normalization** | Scaling a vector so its length is exactly 1. Then cosine similarity = dot product, which is fast to compute. |
| **Dot product** | Multiply the two vectors element by element, then add it all up. `[1,2]·[3,4] = 1×3 + 2×4 = 11`. |
| **FAISS** | A library from Meta that stores vectors and quickly finds the ones closest to a query vector. |
| **Top-k** | "Give me the k best matches." We use k = 4. |
| **Threshold (`MIN_SCORE`)** | A minimum similarity score. Chunks below it are treated as irrelevant. |
| **Grounding** | Forcing the answer to come only from the provided sources, never the LLM's memory. |
| **Citation** | `[1]`, `[2]` markers pointing to the source chunk each claim came from. |
| **Query rewriting** | Turning a follow-up like "who is eligible for it?" into a standalone question before searching. |
| **API / endpoint** | A URL your program responds to, such as `POST /chat`. |
| **FastAPI** | A Python web framework for building APIs. |
| **Uvicorn** | The server program that actually runs a FastAPI app. |
| **venv** | A virtual environment: a private folder of Python packages just for this project. |
| **Hit@k / MRR** | Evaluation metrics. Did we find the right chunk, and how high did we rank it? (Explained in 3.13.) |

### The whole system in one picture

```
INGESTION (when you upload a PDF)
  PDF file
    → ingest.py      read the text of each page, clean it up
    → chunking.py    cut each page into ~250-char chunks (whole sentences)
    → embeddings.py  turn each chunk into a 384-number vector
    → vector_store.py save vectors in FAISS + save chunk text/page to disk

QUESTION ANSWERING (when you ask something)
  "Who is eligible for it?"
    → llm.rewrite_question  → "How long is parental leave? Who is eligible for it?"
    → embeddings.embed      → query vector
    → vector_store.search   → the 4 most similar chunks + scores
    → rag.retrieve          → drop chunks scoring below 0.30
    → llm.generate_answer   → Claude answer with [1] citations (or extractive fallback)
    → main.py /chat         → JSON back to the browser
```

---

## Part 2: Setting up your machine

### Step 1: Create the folder
```bash
mkdir multi-doc-rag
cd multi-doc-rag
```
`mkdir` makes a directory and `cd` moves into it. Everything below happens inside this folder.

### Step 2: Create a virtual environment
```bash
python -m venv .venv
```
- `python -m venv` runs Python's built-in `venv` module.
- `.venv` is the folder name it creates. The leading dot is a convention meaning "tooling, not source code".
- Why: without it, packages install globally and different projects can break each other with conflicting versions.

### Step 3: Activate it
```bash
.venv\Scripts\activate
```
On macOS/Linux it's `source .venv/bin/activate`. Your prompt now starts with `(.venv)`, and `python`/`pip` refer to the project's private copies.
*PowerShell blocked it?* See Part 6.

### Step 4: Install packages
```bash
pip install -r requirements.txt
```
`-r` means "read the package list from this file". It downloads PyTorch, sentence-transformers, FAISS, FastAPI and the rest (explained in 3.4).

### Step 5: Make sample PDFs, run tests, start the server
```bash
python scripts/make_sample_pdfs.py
pytest -q
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000.

---

## Part 3: Every file, explained

### 3.1 Folder structure

```
multi-doc-rag/
├── .gitignore              files git should never commit
├── .env.example            template for your secret settings
├── requirements.txt        list of packages to install
├── README.md               project overview (for GitHub visitors)
├── LEARNING_GUIDE.md       short build-order guide
├── CODE_WALKTHROUGH.md     this file
├── app/                    the application (a Python "package")
│   ├── __init__.py         marks app/ as a package (empty)
│   ├── config.py           all settings
│   ├── ingest.py           PDF → text
│   ├── chunking.py         text → chunks
│   ├── embeddings.py       chunks → vectors
│   ├── vector_store.py     vectors → FAISS search
│   ├── llm.py              chunks → answer
│   ├── rag.py              connects all the steps
│   ├── main.py             web API
│   └── static/index.html   chat web page
├── eval/
│   ├── __init__.py
│   ├── questions.json      test questions with known answers
│   ├── evaluate.py         measures retrieval quality
│   └── results.md          output of evaluate.py
├── scripts/
│   └── make_sample_pdfs.py creates demo PDFs
├── sample_docs/            the generated PDFs
├── tests/
│   ├── __init__.py
│   ├── conftest.py         test setup
│   ├── test_chunking.py    unit tests
│   └── test_api.py         end-to-end tests
└── data/                   (created at runtime, git-ignored) uploads + index
```

**Why split into so many files?** Each file does one job (the "single responsibility" principle). If retrieval is bad, you look in `chunking.py` or `embeddings.py`. If the answer wording is bad, you look in `llm.py`. Each piece can also be tested alone.

---

### 3.2 `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
data/
```

| Line | Why it's ignored |
|---|---|
| `.venv/` | Hundreds of MB of installed packages. Others recreate it from `requirements.txt`. |
| `__pycache__/`, `*.pyc` | Compiled Python bytecode that Python regenerates automatically. |
| `.pytest_cache/` | pytest's internal cache. |
| `.env` | **Your secret API key.** Never commit secrets to GitHub; bots scan for leaked keys within minutes. |
| `data/` | Your uploaded PDFs and generated index: personal data, and rebuildable anyway. |

---

### 3.3 `.env.example`

```ini
# Optional: enables Claude-generated answers. Without it the app uses extractive answers.
ANTHROPIC_API_KEY=
# RAG_LLM_MODEL=claude-opus-5-5
# RAG_TOP_K=4
# RAG_MIN_SCORE=0.30
```

- This is a **template**. You copy it to `.env` and fill in your key. `.env` is git-ignored; `.env.example` is committed so others know which settings exist.
- `#` lines are comments. The commented settings show what can be overridden.
- `python-dotenv` (used in `config.py`) reads `.env` and loads each line as an environment variable.

---

### 3.4 `requirements.txt`

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.4

sentence-transformers>=3.0
faiss-cpu>=1.8
pypdf>=5.0
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
anthropic>=1.11
python-dotenv>=1.0
numpy>=1.26

# dev / tooling
fpdf2>=2.8
pytest>=8.0
httpx>=0.27
```

| Package | What it does in this project |
|---|---|
| `--extra-index-url ...cpu` | Tells pip it can also look at PyTorch's CPU-only package server. The CPU build is ~200 MB; the GPU build is ~2.5 GB. |
| `torch` | PyTorch, the deep learning engine the embedding model runs on. |
| `sentence-transformers` | Loads the MiniLM model and turns text into embeddings with one function call. |
| `faiss-cpu` | Vector index and similarity search. |
| `pypdf` | Extracts text from PDFs. |
| `fastapi` | Web framework for the API. |
| `uvicorn[standard]` | The server that runs FastAPI. `[standard]` adds faster optional extras. |
| `python-multipart` | Needed by FastAPI to receive file uploads (multipart form data). |
| `anthropic` | Official SDK for calling Claude. |
| `python-dotenv` | Loads `.env` files. |
| `numpy` | Fast math on arrays (vectors are numpy arrays). |
| `fpdf2` | Creates PDFs (only used to make the sample documents). |
| `pytest` | Test runner. |
| `httpx` | HTTP client used by FastAPI's `TestClient` in the tests. |

`>=2.4` means "version 2.4 or newer". It's a minimum, not an exact pin, so installs keep working as packages release fixes.

---

### 3.5 `__init__.py` files

`app/__init__.py`, `eval/__init__.py` and `tests/__init__.py` are **empty files**. Their presence tells Python "this folder is a package", which makes imports like `from app.chunking import Chunk` and commands like `python -m eval.evaluate` work.

---

### 3.6 `app/config.py`

**Purpose:** one place for every setting. To change chunk size, you never edit logic, only this file (or an environment variable).

```python
"""Central configuration. Every knob lives here so experiments only touch one file.

Values can be overridden with environment variables (or a .env file).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

# Embedding model: small (80MB), fast on CPU, 384-dim vectors.
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking: chosen from eval/evaluate.py results (see README).
CHUNK_STRATEGY = os.getenv("RAG_CHUNK_STRATEGY", "sentence")  # "sentence" | "fixed"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "250"))  # characters
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))  # characters

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# Chunks scoring below this cosine similarity are treated as irrelevant.
# If nothing passes, the assistant says it doesn't know instead of guessing.
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))

# LLM: used only if an Anthropic credential is configured; otherwise the app
# falls back to an extractive answer built from the retrieved text.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("RAG_LLM_EFFORT", "low")  # low effort = faster chat replies
MAX_HISTORY_TURNS = int(os.getenv("RAG_MAX_HISTORY_TURNS", "6"))
```

**Block by block**

- **The `"""..."""` at the top** is a *docstring*, documentation for the module. Python stores it but doesn't run it.
- **`import os`** gives access to environment variables (`os.getenv`). **`from pathlib import Path`** is the modern way to handle file paths; `/` joins paths and works on Windows and Mac alike.
- **`load_dotenv()`** reads the `.env` file (if present) and puts its values into the environment. It must run *before* the `os.getenv` calls below so they can see those values.
- **`ROOT_DIR = Path(__file__).resolve().parent.parent`**
  - `__file__` is this file's path: `.../multi-doc-rag/app/config.py`
  - `.resolve()` makes it absolute.
  - `.parent` → `.../app`, then `.parent` again → `.../multi-doc-rag`, the project root.
  - Why: paths built from the root work no matter which folder you run commands from.
- **`os.getenv("RAG_DATA_DIR", ROOT_DIR / "data")`** reads the environment variable `RAG_DATA_DIR`. If it isn't set, it uses the default `<root>/data`. *Every* setting follows this pattern: a sensible default that can be overridden. The tests use it to point at a temporary folder.
- **`int(...)` and `float(...)`**: environment variables are always strings, so numbers must be converted.
- **Why these values?**
  - `all-MiniLM-L6-v2`: tiny (80 MB), fast on CPU, good quality, and it outputs 384-dimensional vectors.
  - `CHUNK_SIZE=250`, `CHUNK_OVERLAP=50`, `"sentence"`: picked from the evaluation results (3.13), not guessed.
  - `TOP_K=4`: enough context for an answer without flooding the prompt.
  - `MIN_SCORE=0.30`: real questions scored ≥ 0.39 and junk ≤ 0.14, so 0.30 sits in the gap.
  - `LLM_EFFORT="low"`: answering from four short chunks is easy, and low effort makes replies faster.
  - `MAX_HISTORY_TURNS=6`: send only the last 6 question/answer pairs to Claude, so long chats don't grow the prompt forever.

---

### 3.7 `scripts/make_sample_pdfs.py`

**Purpose:** create three fictional PDFs whose answers we know, so we can test and evaluate.

The file has two parts: a big `DOCS` dictionary holding the document text, and a `build` function that turns it into PDFs.

**The data structure (shortened; the real file has all the sections):**
```python
DOCS = {
    "orion_employee_handbook.pdf": {           # output file name
        "title": "Orion Labs Employee Handbook",
        "pages": [                              # list of pages
            [                                   # page 1 = list of (heading, body) pairs
                ("Working Hours", "Orion Labs operates on a flexible schedule. ..."),
                ("Paid Time Off", "Full-time employees accrue 1.75 days ..."),
            ],
            [ ... page 2 sections ... ],
        ],
    },
    "nimbus_storage_manual.pdf": { ... },
    "helix_security_policy.pdf": { ... },
}
```
Nesting: **dict of documents → each has a list of pages → each page is a list of (heading, body) tuples.** This shape mirrors a real document, and the code below just walks through it.

**The builder:**
```python
from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def build(name: str, spec: dict) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, sections in enumerate(spec["pages"]):
        pdf.add_page()
        if i == 0:
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 10, spec["title"], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
        for heading, body in sections:
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
    pdf.output(str(OUT_DIR / name))


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name, spec in DOCS.items():
        build(name, spec)
        print("wrote", OUT_DIR / name)
```

- `FPDF()` creates an empty PDF in memory.
- `set_auto_page_break(auto=True, margin=15)`: if text reaches 15 mm from the bottom, start a new page automatically.
- `enumerate(spec["pages"])` loops over pages and gives the index `i` too, so we know when we're on the first page (`i == 0`) and should print the title.
- `set_font("Helvetica", "B", 18)`: font, style (`"B"` = bold, `""` = regular), and size in points.
- `multi_cell(0, 10, text, ...)`: writes text that wraps automatically.
  - `0` = width: use the full line width.
  - `10` = the height of each line, in mm.
  - **`new_x="LMARGIN", new_y="NEXT"`**: after writing, move the cursor back to the left margin on the next line. **This fixed the first bug.** By default fpdf2 leaves the cursor at the right edge, so the next `multi_cell` had zero width and crashed with *"Not enough horizontal space to render a single character"*.
- `pdf.ln(4)` adds a 4 mm blank line.
- `pdf.output(path)` writes the file.
- **`if __name__ == "__main__":`**: this code runs only when the file is executed directly (`python scripts/make_sample_pdfs.py`), not when another file imports it. This is a standard Python pattern.
- `OUT_DIR.mkdir(exist_ok=True)`: create `sample_docs/`, and don't error if it already exists.

**Try it:** `python scripts/make_sample_pdfs.py`, then open one of the PDFs in `sample_docs/`.

---

### 3.8 `app/ingest.py`

**Purpose:** pipeline step 1. Read a PDF and return clean text, one entry per page, keeping the page number for citations.

```python
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
```

#### The `Page` dataclass
`@dataclass` auto-generates the boring code (`__init__`, `__repr__`, `__eq__`) for a class that just holds data. Writing `Page(doc_id="x", doc_name="a.pdf", page=1, text="...")` just works. (More in Part 5.)
- `doc_id`: a unique id, so two files with the same name don't collide.
- `doc_name`: the human-readable file name, used in citations.
- `page`: **1-indexed** (the first page is 1, not 0), because that's what people see in a PDF viewer.

#### Why `clean_text` is needed
Here's what pypdf actually returned for page 2 of the handbook (I printed it while debugging):
```
'Parental Leave\nOrion Labs provides 16 weeks of fully paid parental leave to all parents, including birth, adoptive and foster\nparents. Leave may be taken ...'
```
The `\n` characters are line breaks. Some are real structure (after the heading "Parental Leave"); most are just where the PDF wrapped the line ("foster\nparents"). `clean_text` keeps the first kind and removes the second.

#### Each regex, symbol by symbol
`re.sub(pattern, replacement, text)` finds every match of `pattern` and replaces it. The `r"..."` prefix means "raw string", so backslashes are passed to the regex engine as-is.

**1. Re-join hyphenated words:** `r"(\w)-\n(\w)"` → `r"\1\2"`
| Piece | Meaning |
|---|---|
| `(\w)` | one letter/digit, *captured* as group 1 |
| `-\n` | a hyphen followed by a line break |
| `(\w)` | one letter/digit, captured as group 2 |
| `\1\2` | replacement: group 1 directly followed by group 2 (hyphen and newline removed) |

`"retri-\neval"` → `"retrieval"`.

**2. Collapse spaces:** `r"[ \t]+"` → `" "`
`[ \t]` = a space or a tab; `+` = one or more. Any run of spaces/tabs becomes one space.

**3. Detect headings:** `r"^([^\n.!?:;]{1,60})\n(?=[A-Z])"` → `r"\1\n\n"`, with `flags=re.M`
| Piece | Meaning |
|---|---|
| `re.M` (multiline flag) | makes `^` match at the start of **every line**, not just the start of the text |
| `^` | start of a line |
| `[^\n.!?:;]` | any character that is **not** a newline or sentence punctuation (`^` inside `[]` means NOT) |
| `{1,60}` | 1 to 60 of those, so a short line |
| `(...)` | capture the line as group 1 |
| `\n` | the line break at its end |
| `(?=[A-Z])` | *lookahead*: the next character must be a capital letter, but don't consume it |
| `\1\n\n` | replacement: the line plus **two** newlines (a paragraph break) |

In words: a short line with no punctuation, followed by a line starting with a capital, is a heading, so make it its own paragraph.

*Why the lookahead matters (bug #2):* my first version didn't have `(?=[A-Z])`, and it also matched wrapped body lines like `"Orion provides 16 weeks\nof leave"`. A wrapped line continues in **lowercase** (`of`), so requiring a capital next fixes it.

**4. Normalize paragraph breaks:** `r"\n{2,}"` → `"\n\n"`
Two or more newlines in a row become exactly two.

**5. Remove line-wrap newlines:** `r"(?<!\n)\n(?!\n)"` → `" "`
| Piece | Meaning |
|---|---|
| `(?<!\n)` | *negative lookbehind*: the previous character is NOT a newline |
| `\n` | a newline |
| `(?!\n)` | *negative lookahead*: the next character is NOT a newline |

This matches only **single** newlines (word-wrap) and replaces them with a space. Double newlines (paragraph breaks) survive.

**Order matters.** Headings must be turned into double newlines (step 3) *before* step 5 removes single newlines; otherwise the heading line break is gone.

`.strip()` removes leading and trailing whitespace.

#### `load_pdf`
- `PdfReader(str(path))` opens the PDF.
- `enumerate(reader.pages, start=1)` loops over pages with numbering starting at 1.
- `page.extract_text() or ""`: `extract_text()` can return `None` for image-only pages, and `or ""` turns that into an empty string so `clean_text` doesn't crash.
- `if text:` skips empty pages (an empty string counts as False).
- `doc_name or path.name`: use the given name, or fall back to the file name.
- Returns a `list[Page]`.

**Try it:**
```python
from pathlib import Path
from app.ingest import load_pdf, clean_text
print(repr(clean_text("Parental Leave\nOrion provides 16 weeks\nof leave.")))
# 'Parental Leave\n\nOrion provides 16 weeks of leave.'
pages = load_pdf(Path("sample_docs/helix_security_policy.pdf"), "d1")
print(len(pages), pages[0].text[:150])
```
(Run `python` in the project folder with the venv active, then type the lines.)

---

### 3.9 `app/chunking.py`

**Purpose:** pipeline step 2. Split page text into small pieces for precise search.

**Why chunk?** One embedding represents a whole text. Embed a full page covering PTO, holidays and hours, and its vector is a blurry average that matches none of those topics strongly. A 250-character chunk is about one topic, so its vector is sharp.

```python
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
```

#### `_SENTENCE_SPLIT` regex: `(?<=[.!?])\s+(?=[A-Z0-9\"'(])`
| Piece | Meaning |
|---|---|
| `(?<=[.!?])` | lookbehind: the previous character is `.`, `!` or `?` |
| `\s+` | one or more whitespace characters (this is what we split on) |
| `(?=[A-Z0-9\"'(])` | lookahead: the next character is a capital, digit, quote or `(` |

It splits on the space **between** sentences. The lookarounds don't consume characters, so the period stays attached to its sentence. Requiring a capital next avoids splitting "e.g. something".
`re.compile(...)` pre-builds the regex once, which is faster when it's reused many times. The leading `_` in the name is a convention for "private to this module".

#### `Chunk` dataclass
Like `Page`, plus `chunk_id`. Every chunk carries `doc_name` and `page`, and **that's what makes citations possible**: when a chunk is retrieved, we know exactly where it came from.

#### `split_sentences`
- `re.split(r"\n+", text)` first splits on newlines (headings and paragraphs are always separate).
- Then each paragraph is split into sentences with `_SENTENCE_SPLIT`.
- `parts.extend(... for s in ... if s.strip())` is a *generator expression* that strips each piece and skips empty ones.

Example: `"First sentence here. Second one. Third!"` → `["First sentence here.", "Second one.", "Third!"]`.

#### `fixed_chunks`: the simple baseline
- `step = size - overlap`: how far the window moves each time.
- `range(0, max(len(text) - overlap, 1), step)`: start positions. `max(..., 1)` guarantees at least one chunk even for very short text.
- `text[i : i + size]`: a slice of `size` characters starting at `i`.

Worked example, `fixed_chunks("abcdefghij", size=4, overlap=2)`:
- step = 2, start positions = range(0, 8, 2) = 0, 2, 4, 6
- → `"abcd"`, `"cdef"`, `"efgh"`, `"ghij"`. Each neighbouring pair shares 2 characters (the overlap).

**Weakness:** it cuts wherever the count lands, often mid-sentence or mid-word. The evaluation showed this: fixed-500 completely missed one question because the answer sentence was cut in half.

#### `sentence_chunks`: the strategy we use
The idea: put whole sentences into a "bucket" until the next one would overflow `size`. Then close the bucket and start a new one, carrying the last sentence or two forward as overlap.

Variables:
- `chunks`: finished chunks.
- `current`: sentences in the bucket being filled.
- `length`: characters in `current` (+1 per sentence for the separator).

Walk-through with `size=50, overlap=20` on the sentences
`S1="First sentence here."` (20 chars), `S2="Second one is a bit longer than the first."` (42), `S3="Third!"` (6), `S4="Fourth?"` (7), `S5="Fifth and final sentence."` (25):

| Sentence | Check `length + len > 50`? | What happens | `current` after | `length` |
|---|---|---|---|---|
| S1 | bucket empty → skip check | add S1 | [S1] | 21 |
| S2 | 21 + 42 = 63 > 50 → **yes** | emit chunk "S1"; carry S1 (20 ≤ 20); add S2 | [S1, S2] | 64 |
| S3 | 64 + 6 = 70 > 50 → **yes** | emit "S1\nS2"; S2 is 42 > 20 so nothing is carried; add S3 | [S3] | 7 |
| S4 | 7 + 7 = 14 → no | add S4 | [S3, S4] | 15 |
| S5 | 15 + 25 = 40 → no | add S5 | [S3, S4, S5] | 41 |
| end | | emit "S3\nS4\nS5" | | |

Result: `["S1", "S1\nS2", "S3\nS4\nS5"]`

Details:
- `if current and ...`: never emit an empty chunk. A single sentence longer than `size` still becomes its own chunk (sentences are never split).
- The carry loop walks `reversed(current)` (newest sentence first) and keeps sentences while they fit in `overlap`. `carried.insert(0, prev)` puts each at the front so the original order is preserved.
- `"\n".join(current)` joins sentences with **newlines, not spaces. This was bug #1's second half.** Joining with spaces glued headings back onto sentences ("Parental Leave Orion Labs provides…") even after `clean_text` had separated them.

> **Quirk worth noticing (good to discuss in an interview):** in the table, the first chunk "S1" is fully contained in the second chunk "S1\nS2". This happens when a long sentence follows a short carried one. It costs a little duplicate storage but doesn't hurt answers, because `extractive_answer` de-duplicates sentences. An improvement exercise: skip emitting a chunk that would be a subset of the next one.

#### `chunk_pages`
- `{"fixed": fixed_chunks, "sentence": sentence_chunks}[strategy]` is a **dictionary of functions**. Functions are values in Python, so this picks the right one by name instead of writing `if/else`.
- `f"{page.doc_id}:p{page.page}:c{i}"` builds an id like `"79e7e71f1d20:p2:c0"` (document, page 2, chunk 0), handy for debugging.
- Chunking happens **per page**, so a chunk never spans two pages and its page number is always exact.

**Try it:**
```python
from app.chunking import sentence_chunks, fixed_chunks
print(fixed_chunks("abcdefghij", 4, 2))
for c in sentence_chunks("One fact here. Another fact follows. A third one. And a fourth.", 40, 20):
    print(repr(c))
```

---

### 3.10 `app/embeddings.py`

**Purpose:** pipeline step 3. Turn text into vectors that capture meaning.

```python
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # Loaded once per process; first run downloads the model (~80MB).
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    vectors = get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")
```

- **`@lru_cache(maxsize=1)`** remembers the function's return value. The first call loads the model (a few seconds, plus a one-time download into `~/.cache/huggingface`); every later call returns the same loaded model instantly. Without it, every request would reload the model.
- **`SentenceTransformer(name)`** downloads (first time) and loads the MiniLM transformer.
- **`.encode(texts, ...)`** runs the model:
  - `texts`: a **list** of strings, embedded together in batches (much faster than one at a time).
  - `batch_size=32`: process 32 texts per forward pass.
  - **`normalize_embeddings=True`**: scale every vector to length 1. Then `a · b` (dot product) **equals** the cosine similarity of a and b. That lets FAISS use its fast inner-product index and makes the scores easy to read (1 = identical, 0 = unrelated).
  - `show_progress_bar=False` keeps the logs clean.
- **`np.asarray(..., dtype="float32")`**: FAISS requires 32-bit floats.
- Output shape: `(number_of_texts, 384)`, one row per text.

**Try it:**
```python
from app.embeddings import embed
v = embed(["Can I use SMS codes?", "SMS codes are not an accepted second factor.", "Bake bread at 220C."])
print(v.shape)      # (3, 384)
print(v @ v[0])     # similarity to the first sentence: [1.0, 0.66, 0.06]
```
`v @ v[0]` is matrix multiplication: the dot product of every row with the first row. This one line is the core of semantic search.

---

### 3.11 `app/vector_store.py`

**Purpose:** pipeline step 4. Store vectors, find the closest ones to a query, save to disk, and delete documents.

```python
import json
import threading
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.chunking import Chunk


class VectorStore:
    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = index_dir
        self.dim = dim
        self._lock = threading.Lock()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.chunks: dict[int, Chunk] = {}  # faiss id -> chunk
        self.next_id = 0
        self._load()
```

**Constructor**
- `faiss.IndexFlatIP(dim)`: a "flat" index does **exact** search by comparing the query to every stored vector. `IP` = inner product, which equals cosine similarity because our vectors are normalized. Brute force sounds slow, but it handles tens of thousands of vectors in milliseconds.
- `faiss.IndexIDMap2(...)` wraps it so **we** assign each vector's id. Plain FAISS numbers vectors 0, 1, 2… and renumbers after deletions; with our own ids we can always map a result back to its chunk, and delete by id.
- `self.chunks`: a dictionary from FAISS id → `Chunk` (the text, doc and page). **FAISS only stores numbers**, so this dictionary is how a search result becomes readable text.
- `self.next_id`: the next unused id. It only ever increases, so ids are never reused after deletes.
- `threading.Lock()`: FastAPI runs normal (`def`) endpoints in a thread pool, so two uploads could modify the index at the same moment. The lock makes writes happen one at a time.
- `self._load()` restores a saved index from disk, if one exists.

```python
    @property
    def _index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / "chunks.json"

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self.chunks = {int(k): Chunk(**v) for k, v in meta["chunks"].items()}
            self.next_id = meta["next_id"]

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))
        meta = {"next_id": self.next_id, "chunks": {k: asdict(c) for k, c in self.chunks.items()}}
        self._meta_path.write_text(json.dumps(meta), encoding="utf-8")
```

**Persistence (save/load)**
- `@property` lets you write `self._index_path` (no parentheses) while it's computed by a method.
- Two files are saved: `faiss.index` (the vectors, FAISS's binary format) and `chunks.json` (the metadata, human-readable).
- `asdict(c)` converts a dataclass to a dictionary so it can be written as JSON. `Chunk(**v)` does the reverse: `**v` unpacks the dict into keyword arguments.
- `int(k)`: JSON object keys are always strings, so convert them back to ints when loading.
- `mkdir(parents=True, exist_ok=True)` creates `data/index/` and any missing parent folders, without erroring if they exist.
- Result: **restart the server and your documents are still there.**

```python
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        with self._lock:
            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype="int64")
            self.index.add_with_ids(vectors, ids)
            for i, chunk in zip(ids, chunks):
                self.chunks[int(i)] = chunk
            self.next_id += len(chunks)
            self._save()
```

**Adding**
- `with self._lock:` only one thread at a time can run this block. The lock is released automatically at the end, even if an error occurs.
- `np.arange(10, 13)` → `[10, 11, 12]`: new ids for the new chunks. FAISS needs `int64`.
- `add_with_ids(vectors, ids)`: row 0 of `vectors` gets `ids[0]`, and so on.
- `zip(ids, chunks)` pairs each id with its chunk for the metadata dictionary.

```python
    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [i for i, c in self.chunks.items() if c.doc_id == doc_id]
            if ids:
                self.index.remove_ids(np.array(ids, dtype="int64"))
                for i in ids:
                    del self.chunks[i]
                self._save()
            return len(ids)
```

**Deleting:** find every chunk id belonging to the document, remove them from FAISS *and* from the metadata, then save. It returns how many were removed, so the API can return 404 when that's 0 (document not found).

```python
    def search(
        self, query_vector: np.ndarray, k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        # When filtering by document, over-fetch so we still end up with k hits.
        fetch = self.index.ntotal if doc_ids else min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vector.reshape(1, -1), fetch)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            chunk = self.chunks[int(i)]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            results.append((chunk, float(score)))
            if len(results) == k:
                break
        return results
```

**Searching**
- `ntotal` is the number of vectors stored. An empty index means no results.
- `query_vector.reshape(1, -1)`: FAISS expects a 2-D array (a batch of queries), so turn shape `(384,)` into `(1, 384)`. `-1` means "work out this dimension yourself".
- `index.search(queries, n)` returns two arrays, `scores` and `ids`, each of shape `(1, n)`, **already sorted best-first**. `[0]` takes the first (only) query's row.
- `i == -1`: FAISS pads with -1 when it has fewer results than requested.
- **Document filter:** if `doc_ids` is given, fetch *all* results, skip chunks from other documents, and stop at k. Simple and correct for a small index (a large system would use a database that supports metadata filters).
- Returns a list of `(Chunk, score)` pairs.

```python
    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks.values():
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "name": c.doc_name, "chunks": 0, "pages": set()})
            d["chunks"] += 1
            d["pages"].add(c.page)
        return [{**d, "pages": len(d["pages"])} for d in docs.values()]
```

**Listing documents:** there's no separate documents table, so this groups chunks by `doc_id`.
- `docs.setdefault(key, default)` returns `docs[key]`, creating it with `default` first if missing.
- `pages` is a `set` so each page number is counted once.
- `{**d, "pages": len(...)}` copies `d` but replaces the set with its size (sets can't be converted to JSON).

---

### 3.12 `eval/questions.json`

```json
{
  "answerable": [
    {"q": "How many PTO days can I carry over to next year?", "doc": "orion_employee_handbook.pdf", "evidence": "rolls over up to a maximum of 10 days"},
    {"q": "Can I use text message codes for two-factor login?", "doc": "helix_security_policy.pdf", "evidence": "SMS codes are not an accepted second factor"},
    ...
  ],
  "unanswerable": [
    "What is the capital of Australia?",
    "What is the stock price of Orion Labs?",
    ...
  ]
}
```

- **`answerable`** has 23 questions. Each lists the document holding the answer (`doc`) and an exact phrase from the answer (`evidence`).
- **Phrasing differs on purpose:** "text message codes" vs. "SMS", "food per day" vs. "meals". This tests *semantic* search, not keyword matching.
- **`unanswerable`** has 5 off-topic questions the system should refuse. "Stock price of Orion Labs" is deliberately tricky because it names a company that is in the documents.

---

### 3.13 `eval/evaluate.py`

**Purpose:** measure retrieval quality for six chunking configs, so the chunk size is chosen from data.

```python
import json
from pathlib import Path

import faiss
import numpy as np

from app import config
from app.chunking import chunk_pages
from app.embeddings import embed
from app.ingest import load_pdf

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "sample_docs"
K = config.TOP_K

CONFIGS = [
    ("fixed", 200, 40),
    ("fixed", 500, 100),
    ("fixed", 1000, 200),
    ("sentence", 250, 50),
    ("sentence", 500, 100),
    ("sentence", 1000, 200),
]


def normalize(s: str) -> str:
    return " ".join(s.lower().split())
```
- `CONFIGS`: each tuple is `(strategy, size, overlap)`. Overlap is always 20% of size, so only one thing varies at a time.
- `normalize`: lowercase and collapse whitespace, so `"Rolls  over"` matches `"rolls over"` when checking evidence. `s.split()` with no argument splits on any whitespace.

```python
def evaluate(pages, questions, strategy, size, overlap):
    chunks = chunk_pages(pages, strategy, size, overlap)
    vectors = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
```
For each config, build a **fresh, in-memory** index (no ids, no saving) using exactly the same functions the app uses. That means we're measuring the real pipeline.

```python
    hits, reciprocal_ranks = 0, []
    q_vecs = embed([item["q"] for item in questions["answerable"]])
    _, ids = index.search(q_vecs, K)
    for item, row in zip(questions["answerable"], ids):
        rank = next(
            (r for r, i in enumerate(row, start=1)
             if chunks[i].doc_name == item["doc"] and normalize(item["evidence"]) in normalize(chunks[i].text)),
            None,
        )
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
```
- All questions are embedded and searched in one batch. `ids` has shape `(23, 4)`: the top-4 chunk positions for each question.
- `next((generator), None)` returns the **first** rank `r` (1 to 4) where the chunk is from the right doc **and** contains the evidence phrase, or `None` if no top-4 chunk qualifies.
- `hits += rank is not None` works because `True` counts as 1 and `False` as 0.

**The metrics:**
- **Hit@4** = hits ÷ questions: "how often is the right chunk somewhere in the top 4?"
- **MRR (Mean Reciprocal Rank)** = the average of 1/rank (0 if not found). Ranked 1st → 1.0, 2nd → 0.5, 3rd → 0.33, missing → 0.
  - Example: three questions found at ranks 1, 2 and not found → (1 + 0.5 + 0) / 3 = **0.5**.
  - MRR rewards putting the right chunk **first**, which matters because the LLM pays most attention to the top source.

```python
    off_vecs = embed(questions["unanswerable"])
    top_scores, _ = index.search(off_vecs, 1)
    refused = int((top_scores[:, 0] < config.MIN_SCORE).sum())
```
**Off-topic refusal:** for each junk question, look at its single best score. Below `MIN_SCORE` means the app would correctly say "I couldn't find that".
`top_scores[:, 0] < 0.30` produces an array of True/False values, and `.sum()` counts the Trues.

```python
    n = len(questions["answerable"])
    return {
        "strategy": strategy, "size": size, "overlap": overlap, "chunks": len(chunks),
        "hit@k": hits / n, "mrr": float(np.mean(reciprocal_ranks)),
        "refusal": refused / len(questions["unanswerable"]),
    }


def main():
    questions = json.loads((Path(__file__).parent / "questions.json").read_text())
    pages = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        pages += load_pdf(pdf, doc_id=pdf.stem, doc_name=pdf.name)
    if not pages:
        raise SystemExit("No PDFs found. Run: python scripts/make_sample_pdfs.py")

    rows = [evaluate(pages, questions, *cfg) for cfg in CONFIGS]
    ...  # builds and prints a markdown table, writes eval/results.md
```
- `DOCS_DIR.glob("*.pdf")` finds all PDFs. `pdf.stem` is the name without its extension.
- `*cfg` unpacks the tuple `("fixed", 200, 40)` into three separate arguments.
- `raise SystemExit("...")` exits with a helpful message instead of a confusing crash.
- The table format: `{r['hit@k']:.0%}` prints 0.96 as `96%`, and `{r['mrr']:.3f}` prints 3 decimals.

**Results and what they mean**

| strategy | size | Hit@4 | MRR |
|---|---|---|---|
| fixed | 200 | 100% | 0.949 |
| fixed | 500 | **96%** (missed one) | 0.870 |
| fixed | 1000 | 100% | 0.859 |
| **sentence** | **250** | **100%** | **0.971** ← chosen |
| sentence | 500 | 100% | 0.880 |
| sentence | 1000 | 100% | 0.859 |

1. **Smaller is sharper:** MRR drops as chunks grow, because big chunks mix topics.
2. **Sentence boundaries help:** sentence-250 beats fixed-200, and fixed-500 missed a question entirely by cutting the evidence sentence.
3. **Refusal was 80% for every config:** the "stock price of Orion Labs" question always slipped through. More on this in 3.14.

---

### 3.14 `app/llm.py`

**Purpose:** pipeline step 5. (a) Rewrite follow-up questions, (b) generate a grounded, cited answer with Claude, (c) fall back to an extractive answer with no API key.

```python
import logging
import os
import re

import numpy as np

from app import config
from app.chunking import Chunk, split_sentences
from app.embeddings import embed

log = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions about the user's uploaded documents.

Rules:
- Use ONLY the numbered sources in the user message. Do not use outside knowledge.
- Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise: a few sentences or a short list."""

REWRITE_PROMPT = """Rewrite the user's latest message as a standalone search query, resolving
pronouns and references using the conversation. Reply with the query only."""

_FOLLOW_UP = re.compile(r"\b(it|its|that|this|those|these|they|them|their|he|she|there)\b", re.I)
```

**Constants**
- `log = logging.getLogger(__name__)` creates a logger named after the module (`app.llm`). `log.exception(...)` prints the error *and* its full traceback to the server log.
- **`NO_ANSWER`** is one exact string used everywhere: in the prompt, in the extractive fallback, and in `rag.py` (to hide sources when nothing was found). One constant means they can never drift apart.
- **`SYSTEM_PROMPT`**, the grounding rules. The system prompt is the instruction set Claude follows for the whole conversation:
  - "Use ONLY the numbered sources" prevents answers from the model's general knowledge (hallucination).
  - "Cite every claim" makes every statement checkable.
  - "reply exactly: ..." gives a predictable refusal the code can detect.
  - It's an f-string, so `{NO_ANSWER}` is inserted into the text.
- **`REWRITE_PROMPT`**: instructions for turning a follow-up into a standalone search query.
- **`_FOLLOW_UP` regex:** `\b` = word boundary (so "it" matches but "item" doesn't); `(it|its|...)` = any of these words; `re.I` = ignore case. It detects pronoun-based follow-ups.

```python
def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def _client():
    import anthropic  # imported lazily so the app runs without the SDK configured

    return anthropic.Anthropic()
```
- `llm_available()`: is a Claude credential set? `bool(...)` turns a string or `None` into True/False.
- `_client()`: `anthropic.Anthropic()` automatically reads the key from the environment, so the key never appears in code. The import sits inside the function (a "lazy import") so the module loads quickly even if Claude is never used.

```python
def _call_claude(system: str, messages: list[dict], max_tokens: int) -> str:
    response = _client().beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={"effort": config.LLM_EFFORT},
        # If a safety classifier declines, the API retries on a fallback model
        # inside the same call instead of returning an empty refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":  # the whole fallback chain declined
        return NO_ANSWER
    return "".join(b.text for b in response.content if b.type == "text").strip()
```

**The Claude call, parameter by parameter**
| Parameter | Meaning |
|---|---|
| `model` | which Claude model to use (`claude-opus-5-5` by default) |
| `max_tokens` | upper limit on the reply length (a token is about ¾ of a word) |
| `system` | the system prompt (rules) |
| `messages` | the conversation: a list of `{"role": "user"/"assistant", "content": "..."}` |
| `output_config={"effort": "low"}` | how much thinking the model does. Low is fastest, and fine for answering from short sources. |
| `betas` + `fallbacks="default"` | if a safety filter declines the request, the API automatically retries on another model inside the same call |

- `response.stop_reason == "refusal"` means even the fallback declined, so return the standard "couldn't find" message.
- `response.content` is a list of *blocks* (text blocks, and possibly thinking blocks). We keep only the `text` blocks and join them.

```python
def format_sources(contexts: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.doc_name}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(contexts, start=1)
    )
```
This turns the retrieved chunks into the numbered list Claude sees:
```
[1] (orion_employee_handbook.pdf, page 2)
Parental Leave
Orion Labs provides 16 weeks of fully paid parental leave ...

[2] (orion_employee_handbook.pdf, page 1)
...
```
`for i, (c, _) in enumerate(...)` unpacks each `(chunk, score)` pair; `_` is a convention for "I don't need this value".

```python
def rewrite_question(question: str, history: list[dict]) -> str:
    """Make follow-ups like "what about its refund policy?" searchable on their own."""
    if not history:
        return question
    if llm_available():
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        try:
            return _call_claude(
                REWRITE_PROMPT,
                [{"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"}],
                max_tokens=200,
            ) or question
        except Exception:
            log.exception("Query rewrite failed; using raw question")
    # Heuristic fallback: short or pronoun-heavy questions borrow the previous user turn.
    if len(question.split()) <= 4 or _FOLLOW_UP.search(question):
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{previous} {question}".strip()
    return question
```

**Query rewriting.** The problem: embed "Who is eligible for **it**?" and the vector has no idea what "it" is, so the search returns junk.
1. No history (first question): nothing to resolve, return as-is.
2. **With Claude:** send the last 4 messages (`history[-4:]`) and ask for a standalone query. `... or question` falls back to the original if Claude returns an empty string.
3. `try/except`: if the API fails (network error, bad key), log it and keep going with the heuristic. **The app never crashes because an external service is down.**
4. **Heuristic (no key):** if the question is 4 words or fewer, or contains a pronoun, prepend the previous user question.
   - "Who is eligible for it?" → "How long is parental leave? Who is eligible for it?"
   - `next(generator, "")` returns the most recent user message (`reversed(history)` walks newest-first), or `""` if there isn't one.

Important: the rewritten query is used **only for searching**. Claude still answers the user's original question, with the history for context.

```python
def extractive_answer(
    question: str, contexts: list[tuple[Chunk, float]], max_sentences: int = 3, margin: float = 0.15
) -> str:
    first_seen: dict[str, int] = {}  # sentence -> source number; dedupes chunk overlap
    for i, (c, _) in enumerate(contexts, start=1):
        for s in split_sentences(c.text):
            if len(s.split()) >= 4:  # skip bare headings
                first_seen.setdefault(s, i)
    candidates = [(i, s) for s, i in first_seen.items()]
    if not candidates:
        return NO_ANSWER
    scores = embed([text for _, text in candidates]) @ embed([question])[0]
    ranked = np.argsort(-scores)[:max_sentences]
    # Keep only sentences nearly as relevant as the best one, so a single-fact
    # answer isn't padded with loosely related filler.
    keep = [j for j in ranked if scores[j] >= scores[ranked[0]] - margin]
    # Keep the selected sentences in document order so the answer reads naturally.
    return " ".join(f"{candidates[j][1]} [{candidates[j][0]}]" for j in sorted(keep))
```

**Extractive answer (no API key needed).** It picks the best existing sentences instead of writing new text.
1. **Collect candidate sentences** from the retrieved chunks.
   - `len(s.split()) >= 4` skips headings like "Paid Time Off" (fewer than 4 words).
   - `first_seen.setdefault(s, i)`: overlapping chunks contain the same sentence twice. Storing each sentence once, with the **first** source number it appeared in, removes the duplicates.
2. **Score each sentence** against the question: the embedding matrix `@` the question vector gives one cosine score per sentence.
3. **Rank:** `np.argsort(-scores)` returns positions sorted by score, highest first (negating sorts descending). `[:3]` keeps at most 3.
4. **Relative cutoff (the fix for bug #3):** keep only sentences within `margin = 0.15` of the best score. Real numbers from debugging:
   - "How long is parental leave?" scores: 0.67, 0.67, **0.42** (a PTO sentence). Cutoff = 0.67 - 0.15 = 0.52, so the PTO filler is dropped. ✔
   - The follow-up: 0.63, 0.62, **0.50** ("Employees become eligible after 90 days"). Cutoff = 0.48, so it's kept. ✔
5. **`sorted(keep)`** puts the chosen sentences back in document order so the answer reads naturally. Each sentence gets its citation, like `"... 16 weeks ... [1]"`.

```python
def generate_answer(question: str, contexts: list[tuple[Chunk, float]], history: list[dict]) -> tuple[str, str]:
    """Returns (answer, mode) where mode is "claude" or "extractive"."""
    if not contexts:
        return NO_ANSWER, "none"
    if llm_available():
        messages = history[-config.MAX_HISTORY_TURNS * 2 :] + [
            {"role": "user", "content": f"Sources:\n{format_sources(contexts)}\n\nQuestion: {question}"}
        ]
        try:
            return _call_claude(SYSTEM_PROMPT, messages, max_tokens=2000), "claude"
        except Exception:
            log.exception("Claude call failed; falling back to extractive answer")
    return extractive_answer(question, contexts), "extractive"
```

**Choosing the answer mode**
1. **No relevant chunks** (all scored below `MIN_SCORE`) → refuse immediately. No LLM call, no cost, no chance to hallucinate.
2. **Claude available** → build the messages: recent history (`MAX_HISTORY_TURNS * 2` because each turn is 2 messages) plus a new user message with the numbered sources and the question. History lets Claude understand "it" in follow-ups.
3. If Claude fails → log it and fall through to extractive.
4. The function returns a **tuple** `(answer, mode)`, so the API can tell the user which mode answered.

#### The known limitation
"What is the stock price of Orion Labs?" scores 0.42 at chunk level and 0.58 at sentence level, **higher** than five genuine questions (lowest 0.50). Embeddings measure "same topic", not "contains the answer". Any threshold that blocked this question would also block real ones. So:
- With Claude: the system prompt rule ("if the sources don't contain the answer, say so") handles it, because Claude can read the text and see there's no stock price.
- Extractive mode can't make that judgment. It's documented in the README as a known limitation rather than hidden by over-tuning.

---

### 3.15 `app/rag.py`

**Purpose:** the orchestrator. It calls the other modules in the right order and keeps chat history.

```python
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app import config
from app.chunking import chunk_pages
from app.embeddings import embed, get_model
from app.ingest import load_pdf
from app.llm import NO_ANSWER, generate_answer, rewrite_question
from app.vector_store import VectorStore


@dataclass
class Answer:
    answer: str
    mode: str
    search_query: str
    sources: list[dict] = field(default_factory=list)
```
- `Answer` is the result object returned by `ask()`.
- `field(default_factory=list)`: a default value of `[]`. You can't write `= []` directly in a dataclass, because one list would be shared by every instance (a classic Python trap). `default_factory` creates a new list each time.

```python
class RAGService:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        dim = get_model().get_embedding_dimension()
        self.store = VectorStore(index_dir, dim)
        self.sessions: dict[str, list[dict]] = {}  # session_id -> chat history
```
- Asks the model for its vector size (384) instead of hard-coding it. Swap in a different embedding model in `config.py` and everything still works.
- `self.sessions` maps `session_id` → a list of messages. It's stored in memory, so restarting the server clears conversations (documents persist; chats don't). A production version would use Redis or a database.
- `index_dir` is a parameter (default from config) so the tests and the eval can use a separate folder.

```python
    def ingest_pdf(self, path: Path, name: str) -> dict:
        doc_id = uuid.uuid4().hex[:12]
        pages = load_pdf(path, doc_id, name)
        chunks = chunk_pages(pages, config.CHUNK_STRATEGY, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if chunks:
            self.store.add(chunks, embed([c.text for c in chunks]))
        return {"doc_id": doc_id, "name": name, "pages": len(pages), "chunks": len(chunks)}
```
**Ingestion pipeline:** a new random id → read pages → chunk → embed all chunks in one batch → store. `uuid.uuid4().hex[:12]` is a random hex string cut to 12 characters, unique enough here. The returned summary is what the API sends back after an upload.

```python
    def retrieve(self, query: str, k: int = config.TOP_K, doc_ids: list[str] | None = None):
        hits = self.store.search(embed([query])[0], k, doc_ids)
        return [(c, s) for c, s in hits if s >= config.MIN_SCORE]
```
**Retrieval:** embed the query (`embed` takes a list, and `[0]` takes the single resulting vector), search, then **drop anything below the relevance threshold**. This one line is what makes "Who won the World Cup?" return nothing.

```python
    def ask(self, question: str, session_id: str, doc_ids: list[str] | None = None, k: int = config.TOP_K) -> Answer:
        history = self.sessions.setdefault(session_id, [])
        search_query = rewrite_question(question, history)
        contexts = self.retrieve(search_query, k, doc_ids)
        answer, mode = generate_answer(question, contexts, history)
        if answer == NO_ANSWER:
            contexts = []  # don't show sources for an answer that used none
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return Answer(
            answer=answer,
            mode=mode,
            search_query=search_query,
            sources=[
                {"n": i, "doc_name": c.doc_name, "page": c.page, "score": round(s, 3), "text": c.text}
                for i, (c, s) in enumerate(contexts, start=1)
            ],
        )

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
```
**Question pipeline:**
1. `setdefault` gets this session's history, creating an empty list for a new session.
2. Rewrite the question for searching.
3. Retrieve the relevant chunks.
4. Generate the answer from the **original** question.
5. If the answer is the refusal, don't show sources (showing "sources" for "I couldn't find that" would be misleading).
6. **Append** the question and answer to the history. `history` is the same list object stored in `self.sessions`, so `+=` updates the session too.
7. Build the response. `"n": i` matches the `[1]`, `[2]` citation numbers in the answer, and `round(s, 3)` gives scores like 0.69.

`reset_session` deletes a conversation. `pop(key, None)` doesn't error if the key is missing.

---

### 3.16 `app/main.py`

**Purpose:** the web layer. It turns HTTP requests into calls on `RAGService`.

```python
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.llm import llm_available
from app.rag import RAGService

STATIC_DIR = Path(__file__).parent / "static"
rag: RAGService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rag = RAGService()  # loads the embedding model + any saved index once at startup
    yield


app = FastAPI(title="Multi-Document RAG", version="1.0.0", lifespan=lifespan)
```

**Startup (`lifespan`)**
- Code **before** `yield` runs once when the server starts; code after it would run at shutdown.
- It creates the upload folder and builds the `RAGService`, which loads the model (slow) **once** instead of per request.
- `global rag` assigns to the module-level variable `rag`, so every endpoint can use it.
- Why not create `RAGService()` at import time? Then simply importing the module (in tests, for example) would load the model. `lifespan` ties it to the server actually starting.
- `FastAPI(title=..., version=...)`: these appear on the auto-generated docs page at `/docs`.

```python
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    doc_ids: list[str] | None = None  # restrict search to these documents
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class Source(BaseModel):
    n: int
    doc_name: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    mode: str
    search_query: str
    sources: list[Source]
```

**Pydantic models (request/response shapes)**
- FastAPI uses these to **validate** incoming JSON automatically. Send `{"question": ""}` and you get a 422 error without writing a single `if`.
- `Field(min_length=1, max_length=2000)`: the question can't be empty or huge.
- `Field(default=4, ge=1, le=20)`: `ge` = greater-or-equal, `le` = less-or-equal.
- `str | None = None`: optional, defaults to None.
- `ChatResponse` documents exactly what `/chat` returns and filters the output to these fields.
- All of this appears on the `/docs` page, where you can try requests in the browser.

```python
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "llm": "claude" if llm_available() else "extractive", "chunks": rag.store.index.ntotal}
```
- `@app.get("/")` is a **decorator**: when a browser requests `GET /`, run this function. It serves the chat page. `include_in_schema=False` hides it from `/docs`.
- `/health` is a quick status check (used by the UI header, and by monitoring in real deployments). Returning a dict is automatically converted to JSON.

```python
@app.post("/documents", status_code=201)
def upload_documents(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"{f.filename}: only PDF files are supported")
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        try:
            results.append(rag.ingest_pdf(dest, f.filename))
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, f"{f.filename}: could not read PDF ({e})")
    return {"documents": results}
```

**Upload endpoint**
- `status_code=201` means "Created", the correct HTTP code for making something new.
- `files: list[UploadFile] = File(...)` accepts one or more uploaded files from a multipart form. `...` means required.
- **Validation:** reject anything that isn't `.pdf` with a 400 (Bad Request).
- **Security:** the file is saved under a **random name** (`uuid4().hex`), not the user's filename. A malicious name like `../../evil.py` could otherwise write outside the folder ("path traversal"). The original name is only used as a label.
- `with dest.open("wb") as out:` opens the file for **w**riting **b**inary. `with` guarantees it's closed afterwards.
- `shutil.copyfileobj` streams the upload to disk without loading it all into memory.
- If ingestion fails (corrupt PDF): delete the saved file (`unlink`) and return 422 (Unprocessable).
- `def`, not `async def`: ingestion is CPU-heavy and blocking, and FastAPI runs plain `def` endpoints in a thread pool, so the server stays responsive. (This is also why the lock in `VectorStore` matters.)

```python
@app.get("/documents")
def list_documents():
    return {"documents": rag.store.documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = rag.store.delete_document(doc_id)
    if not removed:
        raise HTTPException(404, "document not found")
    return {"doc_id": doc_id, "chunks_removed": removed}
```
- `{doc_id}` in the path is a **path parameter**: FastAPI passes that part of the URL as the `doc_id` argument.
- HTTP verbs follow REST conventions: `GET` = read, `POST` = create, `DELETE` = remove.
- 404 = Not Found.

```python
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if rag.store.index.ntotal == 0:
        raise HTTPException(400, "Upload at least one PDF first")
    session_id = req.session_id or uuid.uuid4().hex
    result = rag.ask(req.question, session_id, req.doc_ids, req.top_k)
    return ChatResponse(session_id=session_id, **result.__dict__)


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    rag.reset_session(session_id)
    return {"session_id": session_id, "reset": True}
```
**Chat endpoint**
- `req: ChatRequest`: FastAPI parses and validates the JSON body into this object.
- If no documents are indexed yet, return a helpful 400 error.
- **Sessions:** if the client didn't send a `session_id`, create one. The client stores it and sends it back on follow-ups, which is how the server knows which history to use. (The same idea as a cookie.)
- `**result.__dict__` unpacks the `Answer` dataclass's fields (`answer`, `mode`, `search_query`, `sources`) as keyword arguments.
- `DELETE /chat/{id}` is the "New chat" button.

**Try it:** start the server, open http://127.0.0.1:8000/docs, and click any endpoint → "Try it out".

---

### 3.17 `app/static/index.html`

**Purpose:** a single-file chat interface. No framework, no build step: HTML for structure, CSS for looks, JavaScript for behavior.

#### HTML structure
```html
<div class="app">
  <aside>                                    <!-- left sidebar -->
    <h1>Multi-Doc RAG</h1>
    <div class="muted" id="mode">…</div>     <!-- "Answer mode: extractive · 33 chunks" -->
    <label class="drop" id="drop">Drop PDFs here or click to upload
      <input type="file" id="file" accept="application/pdf" multiple hidden>
    </label>
    <div class="muted" id="status"></div>    <!-- "Indexing…" / errors -->
    <div id="docs"></div>                    <!-- document list with × buttons -->
  </aside>
  <main>
    <div id="log">...</div>                  <!-- chat messages -->
    <form id="ask">
      <input type="text" id="q" placeholder="Ask about your documents…">
      <button class="primary">Send</button>
      <button type="button" id="reset">New chat</button>
    </form>
  </main>
</div>
```
- Wrapping the hidden `<input type="file">` in a `<label>` means clicking anywhere on the label opens the file picker.
- `accept="application/pdf"` filters the picker to PDFs, and `multiple` allows selecting several.
- `type="button"` on "New chat" stops it from submitting the form.

#### CSS highlights
```css
:root { --bg:#f6f7f9; --panel:#fff; --text:#1c1f24; --accent:#2f6fde; ... }
@media (prefers-color-scheme: dark) { :root { --bg:#14161a; ... } }
.app { display:grid; grid-template-columns:280px 1fr; height:100vh; }
.msg.user { align-self:flex-end; background:var(--user); }
@media (max-width:720px) { .app { grid-template-columns:1fr; ... } }
```
- **CSS variables** (`--bg`, etc.) define colors once, so dark mode only needs to redefine the variables.
- `prefers-color-scheme: dark` follows the operating system's dark mode automatically.
- `grid-template-columns: 280px 1fr` gives a 280px sidebar, and the chat takes the rest (`1fr` = one fraction of the remaining space).
- `.msg.user` aligns the user's own messages to the right, like a chat app.
- `max-width: 720px` stacks the sidebar above the chat on phones.

#### JavaScript, function by function
```js
let sessionId = null;
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
```
- `sessionId` remembers the conversation id the server gave us.
- `$` is a shorthand helper for getting elements by id.
- **`esc`** HTML-escapes text before inserting it into the page. **Security:** a PDF containing `<script>...` must be shown as text, never executed (that would be an XSS attack). Every server-provided string goes through `esc`.

```js
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json();
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
```
- One wrapper for every server call. `fetch` makes the HTTP request; `await` waits for it without freezing the page.
- If the status isn't 2xx, it throws FastAPI's error message (`detail`) so it can be shown to the user.

```js
async function refresh() {
  const h = await api("/health");
  $("mode").textContent = `Answer mode: ${h.llm} · ${h.chunks} chunks indexed`;
  const { documents } = await api("/documents");
  $("docs").innerHTML = documents.map(d => `<div class="doc">...${esc(d.name)}...
     <button class="x" data-id="${d.doc_id}">×</button></div>`).join("") || '<p class="muted">No documents yet.</p>';
}
```
- Updates the header and document list. `documents.map(...)` turns each document into an HTML string, and `.join("")` combines them.
- `|| '<p>No documents yet.</p>'`: an empty string is falsy, so this shows a placeholder when there are no documents.
- `data-id` stores the document id on the × button for the delete handler.

```js
async function upload(files) {
  const fd = new FormData();
  [...files].forEach(f => fd.append("files", f));
  $("status").textContent = "Indexing…";
  try { await api("/documents", { method: "POST", body: fd }); $("status").textContent = ""; }
  catch (e) { $("status").textContent = e.message; }
  refresh();
}
```
- `FormData` builds a multipart upload. Each file is appended under the name `"files"`, which matches the FastAPI parameter name `files`.
- `[...files]` converts the browser's FileList into a real array.

```js
$("file").onchange = e => upload(e.target.files);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); };
```
- File picker and drag-and-drop both call `upload`. `preventDefault()` stops the browser's default behavior (opening the dropped PDF in the tab).

```js
$("docs").onclick = async e => {
  if (!e.target.dataset.id) return;
  await api(`/documents/${e.target.dataset.id}`, { method: "DELETE" });
  refresh();
};
```
- **Event delegation:** one click handler on the whole list. If the clicked element has a `data-id`, it was a × button. This also works for buttons added later.

```js
$("ask").onsubmit = async e => {
  e.preventDefault();
  const question = $("q").value.trim();
  if (!question) return;
  $("q").value = "";
  add("user", esc(question));
  const pending = add("bot", "<span class='muted'>Thinking…</span>");
  try {
    const r = await api("/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }) });
    sessionId = r.session_id;
    const sources = r.sources.map(s => `<div class="src"><b>[${s.n}] ${esc(s.doc_name)} p.${s.page}</b> (score ${s.score})<br>${esc(s.text)}</div>`).join("");
    pending.innerHTML = esc(r.answer) + (sources ? `<details><summary>Sources · searched: “${esc(r.search_query)}”</summary>${sources}</details>` : "");
  } catch (err) { pending.textContent = "Error: " + err.message; }
};
```
**Sending a question**
1. `e.preventDefault()` stops the form from reloading the page (the default form behavior).
2. Show the user's message, plus a "Thinking…" placeholder.
3. POST JSON to `/chat` with the stored `session_id` (`null` the first time).
4. **Save `r.session_id`**: this is what makes follow-up questions work.
5. Replace the placeholder with the answer, plus a collapsible `<details>` showing the sources and the rewritten search query. Showing the query makes the rewriting visible, which is useful for learning and debugging.

```js
$("reset").onclick = async () => {
  if (sessionId) await api(`/chat/${sessionId}`, { method: "DELETE" });
  sessionId = null;
  $("log").innerHTML = '<div class="msg">New conversation started.</div>';
};
refresh();
```
- "New chat" deletes the server-side history and forgets the id.
- `refresh()` at the end loads the document list when the page opens.

---

### 3.18 `tests/conftest.py`

```python
import os
import tempfile

# Must run before `app.config` is imported: isolate test data and force the
# deterministic extractive mode so tests never call (or pay for) the Claude API.
os.environ["RAG_DATA_DIR"] = tempfile.mkdtemp(prefix="rag-test-")
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
```
- pytest automatically runs `conftest.py` **before** importing any test file.
- `tempfile.mkdtemp()` creates a fresh empty folder. Pointing `RAG_DATA_DIR` there means **tests never touch your real uploads or index**.
- Removing the API key forces extractive mode, so tests are:
  - **free** (no API charges),
  - **deterministic** (the same answer every run; LLM output varies),
  - **offline** (they work without internet once the model is downloaded).
- Timing matters: `config.py` reads environment variables *at import time*, so these must be set before anything imports `app.config`.

---

### 3.19 `tests/test_chunking.py`

**Unit tests:** small, fast, one function each, no model needed.

```python
from app.chunking import chunk_pages, fixed_chunks, sentence_chunks, split_sentences
from app.ingest import Page, clean_text

TEXT = "First sentence here. Second one is a bit longer than the first. Third! Fourth? Fifth and final sentence."


def test_clean_text_joins_hyphenated_words_and_line_wraps():
    assert clean_text("retri-\neval is\nuseful") == "retrieval is useful"


def test_clean_text_separates_headings_but_not_wrapped_lines():
    text = clean_text("Parental Leave\nOrion provides 16 weeks\nof leave.")
    assert text == "Parental Leave\n\nOrion provides 16 weeks of leave."
```
- pytest finds every function whose name starts with `test_` and runs it. **`assert`** fails the test if the condition is False and shows both values.
- Test 1 checks the hyphen rule and the line-wrap rule.
- Test 2 is a **regression test** for bug #2: it pins down that headings split but wrapped lines don't. If anyone breaks the regex later, this test fails immediately.

```python
def test_split_sentences():
    assert split_sentences(TEXT) == ["First sentence here.", "Second one is a bit longer than the first.", "Third!", "Fourth?", "Fifth and final sentence."]


def test_fixed_chunks_overlap():
    chunks = fixed_chunks("abcdefghij", size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]
```
- Checks that `.`, `!` and `?` all end sentences.
- Checks the exact sliding-window output worked out in 3.9.

```python
def test_sentence_chunks_never_split_sentences():
    sentences = set(split_sentences(TEXT))
    for chunk in sentence_chunks(TEXT, size=50, overlap=20):
        # every chunk is made of whole sentences only
        rebuilt = chunk
        for s in sentences:
            rebuilt = rebuilt.replace(s, "")
        assert rebuilt.strip() == ""
```
**A property test:** remove every known whole sentence from each chunk. If anything except whitespace is left, a sentence was cut. This checks the *guarantee* rather than one exact output.

```python
def test_sentence_chunks_carry_overlap():
    chunks = sentence_chunks(TEXT, size=50, overlap=30)
    assert len(chunks) > 1
    # the last sentence of one chunk reappears at the start of the next
    assert any(a.split(". ")[-1] in b for a, b in zip(chunks, chunks[1:]))


def test_chunk_pages_keeps_citation_metadata():
    pages = [Page(doc_id="d1", doc_name="a.pdf", page=3, text=TEXT)]
    chunks = chunk_pages(pages, "sentence", 60, 10)
    assert all(c.doc_name == "a.pdf" and c.page == 3 for c in chunks)
    assert chunks[0].chunk_id == "d1:p3:c0"
```
- Overlap test: `zip(chunks, chunks[1:])` pairs each chunk with the next one, and `any(...)` checks that at least one pair shares text.
- Metadata test: every chunk keeps its document name and page. Citations depend on this.

---

### 3.20 `tests/test_api.py`

**End-to-end tests:** exercise the real app through HTTP, with real PDFs and the real model.

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm import NO_ANSWER
from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def client():
    if not any(SAMPLES.glob("*.pdf")):
        pytest.skip("run scripts/make_sample_pdfs.py first")
    with TestClient(app) as c:
        files = [("files", (p.name, p.read_bytes(), "application/pdf")) for p in sorted(SAMPLES.glob("*.pdf"))]
        r = c.post("/documents", files=files)
        assert r.status_code == 201, r.text
        yield c
```
- A **fixture** is setup code that tests receive as an argument (each test below takes `client`).
- `scope="module"`: run the setup **once** for the whole file, since loading the model and uploading PDFs is slow.
- `TestClient(app)` calls the app directly in-process, with no real server needed. Using it in a `with` block runs `lifespan` (startup).
- Each upload entry is a tuple `("files", (filename, bytes, content_type))`, the same shape the browser sends.
- `assert ..., r.text`: if it fails, pytest shows the response body, which makes debugging easy.
- `yield c` hands the client to the tests. Code after `yield` would be cleanup.

```python
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "extractive" and body["chunks"] > 0
```
Server up, extractive mode (proving conftest removed the key), chunks indexed.

```python
def test_answer_cites_correct_document(client):
    r = client.post("/chat", json={"question": "What is the minimum password length?"}).json()
    assert "14 characters" in r["answer"]
    assert r["sources"][0]["doc_name"] == "helix_security_policy.pdf"
    assert "[1]" in r["answer"] or "[2]" in r["answer"]
```
Checks the core promise: right answer, from the right document, with a citation.

```python
def test_off_topic_question_is_refused(client):
    r = client.post("/chat", json={"question": "Who won the 2018 FIFA World Cup?"}).json()
    assert r["answer"] == NO_ANSWER and r["sources"] == []
```
Hallucination guard: an off-topic question gets the refusal and no sources.

```python
def test_follow_up_uses_conversation_context(client):
    first = client.post("/chat", json={"question": "How long is parental leave at Orion Labs?"}).json()
    follow = client.post(
        "/chat", json={"question": "Who is eligible for it?", "session_id": first["session_id"]}
    ).json()
    assert "parental leave" in follow["search_query"].lower()
    assert "90 days" in " ".join(s["text"] for s in follow["sources"])
```
Conversational memory: the follow-up reuses the `session_id`, the rewritten query includes "parental leave", and the eligibility fact is retrieved.

```python
def test_rejects_non_pdf(client):
    r = client.post("/documents", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400


def test_delete_document(client):
    docs = client.get("/documents").json()["documents"]
    target = next(d for d in docs if d["name"] == "nimbus_storage_manual.pdf")
    assert client.delete(f"/documents/{target['doc_id']}").status_code == 200
    r = client.post("/chat", json={"question": "How much does the Nimbus Team plan cost?"}).json()
    assert all(s["doc_name"] != "nimbus_storage_manual.pdf" for s in r["sources"])
```
- Input validation: a `.txt` upload is rejected with a 400.
- Deletion: after deleting the Nimbus manual, no answer can cite it. This test is **last in the file** on purpose, because it changes the shared state the other tests use.

**Run:** `pytest -q` → `14 passed`. `-q` = quiet output. Use `pytest -v` to see each test name.

---

## Part 4: Following one question through the code

Real values from the running app. The user has already asked "How long is parental leave?" and now types **"Who is eligible for it?"**

1. **Browser (`index.html`)**: `onsubmit` sends
   `POST /chat {"question": "Who is eligible for it?", "session_id": "a1b2…"}`
2. **`main.py` `chat()`**: Pydantic validates the body into a `ChatRequest`. The index isn't empty and a session_id exists, so it calls `rag.ask(...)`.
3. **`rag.py` `ask()`**: `history` = the two previous messages (question + answer).
4. **`llm.py` `rewrite_question()`**: no API key, so the heuristic runs. The question contains "it", so it prepends the previous user question:
   → `"How long is parental leave? Who is eligible for it?"`
5. **`rag.py` `retrieve()`** → **`embeddings.embed()`**: MiniLM turns the query into 384 numbers (normalized).
6. **`vector_store.search()`**: FAISS compares it against all 33 chunk vectors and returns the top 4 with their scores. In the live run: **0.642** (page 2, the parental leave chunk), 0.322 (page 2, the next chunk, which contains "eligible after 90 days"), 0.319 (page 1), 0.309 (page 3).
7. **Threshold filter**: keep chunks scoring ≥ 0.30. All four pass here, just barely for the last three.
8. **`llm.generate_answer()`**: no key → **`extractive_answer()`**:
   - splits the chunks into sentences, skips headings, de-duplicates
   - scores each sentence: 0.63 "Orion Labs provides 16 weeks…", 0.62 "Leave may be taken…", 0.50 "Employees become eligible after 90 days…", then lower
   - keeps sentences within 0.15 of the best, so the irrelevant ones are dropped
   - the live run returned → `"Employees become eligible after 90 days of employment. [2]"` (the `[2]` is because that sentence came from the second chunk)
9. **Back in `ask()`**: the answer isn't the refusal, so the sources are kept. The Q&A is appended to the history. An `Answer` is returned.
10. **`main.py`** wraps it in a `ChatResponse` and FastAPI sends JSON:
    ```json
    {"session_id": "a1b2…", "answer": "Employees become eligible after 90 days of employment. [2]",
     "mode": "extractive", "search_query": "How long is parental leave? Who is eligible for it?",
     "sources": [{"n": 1, "doc_name": "orion_employee_handbook.pdf", "page": 2, "score": 0.66, "text": "..."}, ...]}
    ```
11. **Browser** shows the answer, and the collapsible "Sources · searched: …" section.

With an API key, steps 4 and 8 call Claude instead: the rewrite is smarter, and the answer is a fluent sentence with `[n]` citations.

---

## Part 5: Python features used, explained

| Feature | Example in this project | What it does |
|---|---|---|
| **Type hints** | `def embed(texts: list[str]) -> np.ndarray:` | Document the expected types. Python doesn't enforce them, but editors use them for autocomplete and warnings. |
| **`X \| None`** | `doc_ids: list[str] \| None = None` | "Either a list of strings or None" (an optional value). |
| **Dataclass** | `@dataclass class Chunk:` | Auto-generates `__init__`, `__repr__` and `__eq__` for data-holding classes. |
| **f-string** | `f"[{i}] ({c.doc_name}, page {c.page})"` | Inserts values into a string. `{x:.3f}` formats a number to 3 decimals. |
| **List comprehension** | `[c.text for c in chunks]` | Builds a list in one line. Equivalent to a for-loop with `.append`. |
| **Generator expression** | `"".join(b.text for b in content if ...)` | Like a list comprehension but lazy (no intermediate list). |
| **Dict comprehension** | `{int(k): Chunk(**v) for k, v in ...}` | Builds a dictionary in one line. |
| **Tuple unpacking** | `for i, (c, s) in enumerate(contexts, start=1):` | Splits pairs into separate variables. |
| **`*` / `**` unpacking** | `evaluate(pages, q, *cfg)`, `Chunk(**v)` | `*` spreads a list into positional arguments; `**` spreads a dict into keyword arguments. |
| **`enumerate`** | `enumerate(reader.pages, start=1)` | Loop with a counter. |
| **`zip`** | `zip(ids, chunks)` | Loop over two lists in parallel. |
| **`next(gen, default)`** | `next((r for r ... if ...), None)` | First matching item, or the default if none. |
| **Decorator** | `@lru_cache`, `@app.get("/")`, `@property` | Wraps a function to add behavior (caching, routing, attribute access). |
| **`lru_cache`** | `get_model()` | Remembers results, so the model loads only once. |
| **Context manager (`with`)** | `with self._lock:`, `with dest.open("wb")` | Guarantees cleanup (unlock, close the file) even on errors. |
| **`try / except`** | around Claude calls | Catch errors and fall back instead of crashing. |
| **`global`** | `global rag` in `lifespan` | Assign to a module-level variable from inside a function. |
| **`if __name__ == "__main__":`** | in scripts | Run code only when the file is executed directly, not imported. |
| **Lazy import** | `import anthropic` inside `_client()` | Import only when needed. |
| **`dict.setdefault`** | `self.sessions.setdefault(id, [])` | Get a key's value, or insert the default first if the key is missing. |
| **Regex lookarounds** | `(?<=...)`, `(?=...)`, `(?<!...)`, `(?!...)` | Check what's before/after without consuming it (explained in 3.8). |
| **NumPy `@`** | `matrix @ vector` | Matrix multiplication: many dot products at once. |
| **`np.argsort(-x)`** | ranking sentences | Positions that would sort the array, highest first. |

---

## Part 6: Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell's execution policy blocks scripts | Skip activation and call the venv's Python directly, e.g. `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`. Or use Command Prompt and `.venv\Scripts\activate.bat`. Changing the execution policy is a security setting; only do it if you understand it. |
| `ModuleNotFoundError: No module named 'app'` | Running from the wrong folder | `cd` into `multi-doc-rag` (the folder that contains `app/`) first. |
| `ModuleNotFoundError: No module named 'faiss'` (or another package) | The venv isn't active, so the global Python is being used | Activate the venv, or use `.venv\Scripts\python.exe`. |
| First run is slow / downloads | The embedding model (~80 MB) downloads once | Wait; later runs use the cached copy. |
| `Warning: You are sending unauthenticated requests to the HF Hub` | The model is downloaded from Hugging Face without an account token | Harmless; ignore it. A free HF token only raises download rate limits. |
| `[Errno 10048] address already in use` | Port 8000 is taken (an old server is still running) | Stop the other server, or run `uvicorn app.main:app --port 8001`. |
| Answer mode shows "extractive" even with a key | `.env` is missing or in the wrong folder, or the server was started before you added the key | Put `.env` in the project root and restart the server. |
| `400 Upload at least one PDF first` | Empty index | Upload a PDF in the UI, or via `POST /documents`. |
| Old behavior after changing chunk settings | The saved index was built with the old chunking | Delete the `data/` folder and re-upload. |
| `FPDFException: Not enough horizontal space` | The fpdf2 cursor stayed at the right margin | Use `new_x="LMARGIN", new_y="NEXT"` (already in the code). |
| `pytest` says "skipped" | The sample PDFs don't exist yet | `python scripts/make_sample_pdfs.py` |

---

## Part 7: Command cheat sheet

| Goal | Command |
|---|---|
| Create venv | `python -m venv .venv` |
| Activate (Windows) | `.venv\Scripts\activate` |
| Install packages | `pip install -r requirements.txt` |
| Make sample PDFs | `python scripts/make_sample_pdfs.py` |
| Run tests | `pytest -q` (or `pytest -v` for names) |
| Run one test file | `pytest tests/test_chunking.py -v` |
| Run evaluation | `python -m eval.evaluate` |
| Start server | `uvicorn app.main:app --reload` (`--reload` restarts on code changes) |
| Chat UI | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |
| Upload via curl | `curl -F files=@sample_docs/helix_security_policy.pdf http://127.0.0.1:8000/documents` |
| Ask via curl | `curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is the minimum password length?\"}"` |
| Try another chunk size | set `RAG_CHUNK_SIZE=500` in `.env`, delete `data/`, restart |
| Reset everything | stop the server, delete `data/` |

---

## Part 8: Uploading to GitHub

```bash
git init
git add .
git status
```
- `git init` turns the folder into a git repository.
- `git add .` stages every file, except those listed in `.gitignore`.
- **`git status` — check it!** You should NOT see `.venv/`, `data/` or `.env`. If you do, fix `.gitignore` before committing.

```bash
git commit -m "Conversational multi-document RAG with FastAPI, FAISS and Claude"
```
A commit is a saved snapshot, with a message describing it.

On github.com: **New repository** → name it `multi-doc-rag` → leave "Add README" **unchecked** (you already have one) → Create. Then:
```bash
git remote add origin https://github.com/priyank1510/multi-doc-rag.git
git branch -M main
git push -u origin main
```
- `remote add origin` tells git where the GitHub copy lives.
- `branch -M main` names your branch `main`.
- `push -u origin main` uploads it. `-u` remembers the destination, so next time just `git push`.

**Making it look good on GitHub:** in the repo's About section (the gear icon), add a description and topics: `rag`, `fastapi`, `faiss`, `llm`, `claude`, `nlp`, `python`.

**Resume line that matches this code:**
> Built a conversational multi-document RAG system (FastAPI, sentence-transformer embeddings, FAISS, Claude) with cited, page-level answers and follow-up query rewriting. Evaluated 6 chunking strategies on a labeled question set, and sentence-aware chunking raised MRR from 0.86 to 0.97.
