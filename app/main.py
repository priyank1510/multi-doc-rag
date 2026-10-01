"""Step 7: expose the pipeline as a REST API (and serve a small chat UI).

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "llm": "claude" if llm_available() else "extractive", "chunks": rag.store.index.ntotal}


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


@app.get("/documents")
def list_documents():
    return {"documents": rag.store.documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = rag.store.delete_document(doc_id)
    if not removed:
        raise HTTPException(404, "document not found")
    return {"doc_id": doc_id, "chunks_removed": removed}


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
