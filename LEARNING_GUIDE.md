# How this project was built: a follow-along guide

> This is the short version. For every file's full code with a block-by-block explanation, see **[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)**.

This guide rebuilds the project in the same order I built it. Each step says **what** the file does, **why** it's designed that way, and gives a **try it** command so you can watch the piece work alone before moving on. Type the code yourself rather than copying. That's how it sticks.

---

## Step 0: The mental model

RAG = **R**etrieval-**A**ugmented **G**eneration. An LLM doesn't know your PDFs, so for each question you:
1. **Retrieve** the few passages most relevant to the question.
2. **Augment** the prompt with those passages.
3. **Generate** an answer that may only use them.

Every file in `app/` is one stage of that pipeline:

| Stage | File | Input → Output |
|---|---|---|
| Read | `ingest.py` | PDF → list of `Page(text, page_no)` |
| Split | `chunking.py` | pages → list of `Chunk` (~250 chars) |
| Embed | `embeddings.py` | text → 384-dim vectors |
| Store/search | `vector_store.py` | vectors → FAISS index → top-k chunks |
| Answer | `llm.py` | question + chunks → cited answer |
| Wire up | `rag.py` | runs the stages in order + chat memory |
| Serve | `main.py` | HTTP API + UI |

Build them **bottom-up** and test each one in isolation. When something breaks, you'll know which stage it's in.

---

## Step 1: Project setup

```bash
mkdir multi-doc-rag && cd multi-doc-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Why a venv?** It keeps this project's library versions separate from every other project on your machine, and `requirements.txt` lets anyone recreate them.
**Why CPU torch?** sentence-transformers needs PyTorch. The CPU build is ~200MB versus ~2.5GB for CUDA, and a small embedding model runs fast on CPU.

Then create `app/config.py`. Every number you might tune (chunk size, top-k, threshold, model names) lives there and can be overridden with an env var. That way experiments touch one file, never the logic.

---

## Step 2: Sample data (`scripts/make_sample_pdfs.py`)

You can't test a RAG system without documents **whose answers you know**. I wrote three fictional documents (handbook, product manual, security policy) and generate them as real PDFs with `fpdf2`.

```bash
python scripts/make_sample_pdfs.py
```

🐞 **First bug I hit:** `FPDFException: Not enough horizontal space`. In fpdf2, `multi_cell` leaves the cursor at the *right* edge, so the next cell had zero width. Fix: `new_x="LMARGIN", new_y="NEXT"`. Lesson: read the exception, then check the library's defaults.

---

## Step 3: Read PDFs (`app/ingest.py`)

`pypdf` extracts text per page. Raw PDF text is messy, so `clean_text` fixes three things:
- `retri-\neval` → `retrieval` (words hyphenated across lines)
- single `\n` → space (a PDF line break is usually just word-wrap)
- headings get their own paragraph (more on this in Step 9)

We keep the **page number** on every `Page`. That's what makes citations like "page 2" possible later.

**Try it:**
```python
from pathlib import Path
from app.ingest import load_pdf
pages = load_pdf(Path("sample_docs/helix_security_policy.pdf"), "d1")
print(len(pages), pages[0].text[:200])
```

---

## Step 4: Chunking (`app/chunking.py`)

**Why chunk at all?** An embedding is one vector for a whole piece of text. Embed a full page and the vector becomes an average of ten topics, matching none of them well. Small chunks give precise matches.

Two strategies, so we can *measure* which is better instead of guessing:
- **fixed:** a sliding character window, e.g. 500 chars with 100 overlap. Simple, but it can cut a sentence in half.
- **sentence:** packs whole sentences up to the size budget, then carries the last sentence or two into the next chunk.

**Why overlap?** If a fact sits on a chunk boundary, overlap means at least one chunk contains it whole.

**Try it:**
```python
from app.chunking import sentence_chunks
text = "One fact here. Another fact follows. A third one. And a fourth."
for c in sentence_chunks(text, size=40, overlap=20): print(repr(c))
```
Notice how sentences repeat across neighbouring chunks: that's the overlap.

---

## Step 5: Embeddings (`app/embeddings.py`)

`all-MiniLM-L6-v2` is a small transformer that maps text to a 384-number vector. Texts with similar meaning get nearby vectors, even with no shared words ("text message codes" ≈ "SMS").

Two design details:
- `normalize_embeddings=True` gives every vector length 1, so **dot product = cosine similarity**. That lets us use FAISS's fastest index type.
- `@lru_cache` loads the model once per process. Loading takes seconds, and you don't want to repeat it on every request.

**Try it:**
```python
from app.embeddings import embed
v = embed(["Can I use SMS codes?", "SMS codes are not an accepted second factor.", "Bake bread at 220C."])
print(v @ v[0])   # similarity of each sentence to the first
```
The related sentence scores far higher than the bread one. That one line is the whole idea behind semantic search.

---

## Step 6: Vector store (`app/vector_store.py`)

FAISS stores vectors and finds the nearest ones to a query vector.
- `IndexFlatIP` is exact inner-product search. Brute force, but fine up to roughly 100k chunks.
- `IndexIDMap2` wraps it so **we** choose the ids. Then we can map an id back to its chunk (text, doc, page) and later delete every chunk of one document.
- The index plus a `chunks.json` metadata file are written to `data/index/`, so uploads survive a restart.
- A `threading.Lock` guards writes, because FastAPI runs sync endpoints in a thread pool.

Filtering by document (`doc_ids`) over-fetches and then filters. That's simple and correct at this scale; a large system would use one index per tenant or a vector DB with metadata filters.

---

## Step 7: Measure before tuning (`eval/`)

Before writing the LLM part, I built the **evaluation**. `questions.json` holds 23 questions, each with the correct document and an evidence phrase, plus 5 off-topic questions.

Metrics:
- **Hit@k:** did any of the top-k chunks contain the evidence?
- **MRR (mean reciprocal rank):** 1 if the right chunk ranked first, ½ if second, and so on. It rewards good *ranking*, not just presence.
- **Off-topic refusal:** for junk questions, did every chunk score below `MIN_SCORE`?

```bash
python -m eval.evaluate
```

Results: sentence-250 won (MRR 0.971). Big chunks dropped to 0.86, and fixed-500 missed a question completely because it cut the evidence sentence in half. **This is the kind of finding to talk about in interviews:** you didn't pick a chunk size from a blog post, you measured it.

I also looked at the raw scores to set `MIN_SCORE`. Real questions scored ≥ 0.39 and junk ≤ 0.14, so 0.30 sits safely in the gap.

---

## Step 8: Generation (`app/llm.py`)

Three jobs:

**a) Query rewriting.** "Who is eligible for *it*?" retrieves garbage, because "it" has no meaning on its own. Before retrieval, the follow-up is rewritten into a standalone query using chat history. Claude does the rewrite when a key is set. Otherwise a heuristic kicks in: a short or pronoun-containing question gets the previous question prepended.

**b) Grounded answering with Claude.** The system prompt says: use only the numbered sources, cite them as `[1]`, and if the answer isn't there, reply with an exact "I couldn't find that" sentence. Sources are formatted as `[n] (file, page N)` followed by the text. The call uses `effort: "low"` because this is fast chat over short context, and server-side `fallbacks` so a rare safety-classifier refusal gets retried on another model.

**c) Extractive fallback.** No API key? Split the retrieved chunks into sentences, embed them, and return the top sentences with citations. The answer is always grounded, just less fluent. This means the project **runs for anyone who clones it**, which matters for a GitHub portfolio.

---

## Step 9: Debugging with real usage

Once the server ran, I asked real questions in the browser. Tests passed, but three problems showed up:

| Symptom | Root cause | Fix |
|---|---|---|
| "Parental Leave Orion Labs provides…" | Two causes: the newline after a heading was flattened to a space, *and* chunking re-joined sentences with spaces | Detect a heading (short line, no punctuation, next line starts with a capital) and join chunk sentences with `\n` |
| My first heading regex split wrapped body lines | A wrapped line is also "short with no punctuation" | Require the *next* line to start with a capital letter. Wrapped lines continue in lowercase |
| Answers padded with an unrelated 3rd sentence | Extractive mode always returned 3 sentences | Keep only sentences within 0.15 of the best score, and skip headings (< 4 words) |

And one I deliberately **didn't** "fix": *"What is the stock price of Orion Labs?"* scored 0.58, higher than five real questions. A threshold strict enough to block it would block real answers too. The honest answer is that embeddings measure *topic similarity*, not *whether the answer exists*. That judgment is the LLM's job. It's documented as a known limitation instead of being hidden by over-fitting a threshold.

Lessons: **test with real usage, not just unit tests**, and **look at the actual numbers before changing a threshold.**

---

## Step 10: The API (`app/main.py`) and UI

- `lifespan` loads the model and index **once** at startup.
- Pydantic models (`ChatRequest`, `ChatResponse`) validate input and generate the `/docs` page automatically.
- Uploads are checked for `.pdf`, saved under a random name (never trust the user's filename on disk), and ingested. Unreadable PDFs return 422.
- `session_id` keys the chat history. The first `/chat` call creates one, and the client sends it back on later calls.
- `static/index.html` is a single file with no framework: upload, document list, chat, and expandable sources.

```bash
uvicorn app.main:app --reload
```

---

## Step 11: Tests (`tests/`)

- `conftest.py` sets a temporary data dir and **removes the API key** before the app is imported. Tests are deterministic, isolated from your real index, and free.
- `test_chunking.py` holds pure unit tests (fast, no model).
- `test_api.py` is end-to-end through FastAPI's `TestClient`: upload → cited answer → off-topic refusal → follow-up rewrite → reject non-PDF → delete.

```bash
pytest -q        # 14 passed
```

---

## Step 12: Put it on GitHub

```bash
git init
git add .
git status        # check: no .venv/, data/, or .env in the list
git commit -m "Conversational multi-document RAG with FastAPI, FAISS and Claude"
```
Create an empty repo on github.com (no README, since you have one), then:
```bash
git remote add origin https://github.com/priyank1510/multi-doc-rag.git
git branch -M main
git push -u origin main
```
`.gitignore` already excludes `.venv/`, `data/` and `.env`, so your API key never gets committed.

---

## Interview talking points
1. **Why RAG over fine-tuning?** Documents change. RAG updates instantly, cites sources, and costs nothing to "retrain".
2. **How did you choose chunk size?** I built an eval set and measured Hit@k and MRR across 6 configs. Sentence-aware 250-char chunks won.
3. **How do you reduce hallucination?** A relevance threshold (refuse when nothing matches), a prompt that only allows the numbered sources plus mandatory citations, and an exact refusal string.
4. **What would you improve?** A cross-encoder reranker on the top-20, hybrid BM25 + vector search for exact terms like error codes, streaming responses, an answer-level eval (faithfulness) using Claude as judge, and a hosted vector DB for scale.
5. **A limitation you found:** embedding similarity can't tell "a question about Orion Labs" from "a question the Orion docs answer". That's why the LLM grounding rule matters.

## Exercises to make it your own
- [ ] Add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and re-run the eval. Does MRR improve?
- [ ] Add `.txt` / `.md` upload support in `ingest.py`.
- [ ] Stream Claude's answer to the UI with `client.messages.stream(...)`.
- [ ] Add a `Dockerfile` so it runs with one command.
