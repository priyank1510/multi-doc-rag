"""Compare chunking strategies by retrieval quality.

For each config we build a fresh in-memory index of sample_docs/ and measure:
- Hit@k:  share of questions where a top-k chunk from the right doc contains the evidence
- MRR:    mean reciprocal rank of the first correct chunk (rewards ranking it first)
- Refusal: share of off-topic questions where no chunk passes MIN_SCORE
           (i.e. the system correctly says "I don't know" instead of guessing)

Run: python -m eval.evaluate
"""
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


def evaluate(pages, questions, strategy, size, overlap):
    chunks = chunk_pages(pages, strategy, size, overlap)
    vectors = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

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

    off_vecs = embed(questions["unanswerable"])
    top_scores, _ = index.search(off_vecs, 1)
    refused = int((top_scores[:, 0] < config.MIN_SCORE).sum())

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
    header = f"| strategy | size | overlap | chunks | Hit@{K} | MRR | Off-topic refusal |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | {r['size']} | {r['overlap']} | {r['chunks']} | "
            f"{r['hit@k']:.0%} | {r['mrr']:.3f} | {r['refusal']:.0%} |"
        )
    table = "\n".join(lines)
    print(table)
    (Path(__file__).parent / "results.md").write_text(
        f"# Retrieval evaluation\n\n{len(questions['answerable'])} answerable + "
        f"{len(questions['unanswerable'])} off-topic questions, model `{config.EMBEDDING_MODEL}`, "
        f"MIN_SCORE={config.MIN_SCORE}\n\n{table}\n"
    )


if __name__ == "__main__":
    main()
