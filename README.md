# Conversational Multi-Document RAG

Chat with several PDFs at once. Answers are grounded in the documents and cite the exact file and page they came from. Follow-up questions keep the conversation's context.

**Stack:** Python · FastAPI · sentence-transformers (transformer embeddings) · FAISS · Anthropic Claude (optional) · pytest

```
                ┌────────────── ingestion ───────────────┐
 PDF upload ──▶ pypdf text ──▶ clean ──▶ sentence chunks ──▶ MiniLM embeddings ──▶ FAISS index (+ metadata on disk)

                ┌────────────────── question answering ──────────────────┐
 question ──▶ rewrite follow-up ──▶ embed ──▶ FAISS top-k ──▶ score filter ──▶ Claude (grounded, cited)
              ("it" → "parental leave")                     (refuse if      └▶ or extractive fallback
                                                             nothing relevant)
```

## Features
- **Multi-document ingestion:** upload many PDFs. Each chunk keeps its document name and page number, so answers can cite them.
- **Semantic retrieval:** `all-MiniLM-L6-v2` embeddings, normalized so FAISS inner-product search equals cosine similarity.
- **Conversational:** follow-ups like "who is eligible for *it*?" are rewritten into standalone search queries before retrieval.
- **Grounded answers:** Claude answers only from the numbered sources and must cite them. If no chunk passes the relevance threshold, the system says it doesn't know instead of guessing.
- **Runs without an API key:** without `ANTHROPIC_API_KEY`, an extractive mode answers with the most relevant source sentences.
- **Evaluated:** `eval/evaluate.py` compares chunking strategies on a labeled question set (Hit@k, MRR, off-topic refusal).
- **Persistent:** the FAISS index and metadata are saved to `data/` and reloaded on restart. You can delete a document and remove all of its chunks.

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_sample_pdfs.py
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000, drop in the PDFs from `sample_docs/` (or your own), and ask questions. Interactive API docs are at http://127.0.0.1:8000/docs.

To use Claude for answers, copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## API
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload one or more PDFs (multipart `files`) |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document and its chunks |
| `POST` | `/chat` | `{"question", "session_id?", "doc_ids?", "top_k?"}` → answer, sources, rewritten query |
| `DELETE` | `/chat/{session_id}` | Reset conversation memory |
| `GET` | `/health` | Status, answer mode, number of chunks |

```bash
curl -F files=@sample_docs/helix_security_policy.pdf http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is the minimum password length?\"}"
```

## Evaluation
`python -m eval.evaluate` builds a fresh index for each chunking config and scores 23 answerable and 5 off-topic questions:

| strategy | size | overlap | chunks | Hit@4 | MRR | Off-topic refusal |
|---|---|---|---|---|---|---|
| fixed | 200 | 40 | 41 | 100% | 0.949 | 80% |
| fixed | 500 | 100 | 19 | 96% | 0.870 | 80% |
| fixed | 1000 | 200 | 9 | 100% | 0.859 | 80% |
| **sentence** | **250** | **50** | 33 | **100%** | **0.971** | 80% |
| sentence | 500 | 100 | 19 | 100% | 0.880 | 80% |
| sentence | 1000 | 200 | 9 | 100% | 0.859 | 80% |

**Takeaways**
- Smaller, sentence-aligned chunks rank the right passage first most often. Large chunks dilute the embedding with unrelated facts. Sentence-250 is the default.
- Fixed windows can cut the key fact in half (fixed-500 missed one question entirely). Sentence chunking never splits a sentence.
- The one off-topic miss is *"What is the stock price of Orion Labs?"*. It names a company in the docs, so it scores as high as real questions, and no similarity threshold can separate the two. The threshold catches clearly unrelated questions. Near-misses are caught by Claude's instruction to answer only from the sources. **Known limitation:** extractive mode (no API key) can't make that judgment and will return the closest Orion Labs sentences.

## Tests
```bash
pytest -q
```
The tests cover text cleaning and chunking, plus end-to-end API flows: upload, cited answers, off-topic refusal, follow-up rewriting, non-PDF rejection and deletion. They always run in extractive mode, so they never call the paid API.

## Project layout
```
app/
  config.py        all tunable settings (env-overridable)
  ingest.py        PDF → cleaned page text
  chunking.py      fixed vs sentence chunking
  embeddings.py    transformer embeddings (normalized)
  vector_store.py  FAISS index + metadata persistence + delete
  llm.py           query rewriting, Claude answers, extractive fallback
  rag.py           orchestrates the pipeline + chat sessions
  main.py          FastAPI routes
  static/index.html  chat UI
eval/              labeled questions + chunking evaluation
scripts/           sample PDF generator
tests/             pytest suite
```
