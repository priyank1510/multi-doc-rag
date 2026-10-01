"""End-to-end test through the HTTP API, using the generated sample PDFs."""
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


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "extractive" and body["chunks"] > 0


def test_list_documents(client):
    names = {d["name"] for d in client.get("/documents").json()["documents"]}
    assert "nimbus_storage_manual.pdf" in names


def test_answer_cites_correct_document(client):
    r = client.post("/chat", json={"question": "What is the minimum password length?"}).json()
    assert "14 characters" in r["answer"]
    assert r["sources"][0]["doc_name"] == "helix_security_policy.pdf"
    assert "[1]" in r["answer"] or "[2]" in r["answer"]


def test_off_topic_question_is_refused(client):
    r = client.post("/chat", json={"question": "Who won the 2018 FIFA World Cup?"}).json()
    assert r["answer"] == NO_ANSWER and r["sources"] == []


def test_follow_up_uses_conversation_context(client):
    first = client.post("/chat", json={"question": "How long is parental leave at Orion Labs?"}).json()
    follow = client.post(
        "/chat", json={"question": "Who is eligible for it?", "session_id": first["session_id"]}
    ).json()
    assert "parental leave" in follow["search_query"].lower()
    assert "90 days" in " ".join(s["text"] for s in follow["sources"])


def test_rejects_non_pdf(client):
    r = client.post("/documents", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400


def test_delete_document(client):
    docs = client.get("/documents").json()["documents"]
    target = next(d for d in docs if d["name"] == "nimbus_storage_manual.pdf")
    assert client.delete(f"/documents/{target['doc_id']}").status_code == 200
    r = client.post("/chat", json={"question": "How much does the Nimbus Team plan cost?"}).json()
    assert all(s["doc_name"] != "nimbus_storage_manual.pdf" for s in r["sources"])
