"""The orchestrator: wires ingestion, chunking, embedding, retrieval and generation.

    PDF -> pages -> chunks -> vectors -> FAISS          (ingest_pdf)
    question -> rewrite -> vector -> top-k chunks -> LLM (ask)
"""
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


class RAGService:
    def __init__(self, index_dir: Path = config.INDEX_DIR):
        dim = get_model().get_embedding_dimension()
        self.store = VectorStore(index_dir, dim)
        self.sessions: dict[str, list[dict]] = {}  # session_id -> chat history

    def ingest_pdf(self, path: Path, name: str) -> dict:
        doc_id = uuid.uuid4().hex[:12]
        pages = load_pdf(path, doc_id, name)
        chunks = chunk_pages(pages, config.CHUNK_STRATEGY, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if chunks:
            self.store.add(chunks, embed([c.text for c in chunks]))
        return {"doc_id": doc_id, "name": name, "pages": len(pages), "chunks": len(chunks)}

    def retrieve(self, query: str, k: int = config.TOP_K, doc_ids: list[str] | None = None):
        hits = self.store.search(embed([query])[0], k, doc_ids)
        return [(c, s) for c, s in hits if s >= config.MIN_SCORE]

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
