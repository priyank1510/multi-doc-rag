"""Step 4: store chunk vectors in FAISS and find the nearest ones to a query.

IndexFlatIP does exact (brute-force) inner-product search, which is plenty fast
for tens of thousands of chunks. Wrapping it in IndexIDMap2 lets us attach our
own integer ids, so we can delete every chunk of a document later.

The index and chunk metadata are saved to disk, so uploads survive a restart.
"""
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

    # ---------- persistence ----------
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

    # ---------- writes ----------
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

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [i for i, c in self.chunks.items() if c.doc_id == doc_id]
            if ids:
                self.index.remove_ids(np.array(ids, dtype="int64"))
                for i in ids:
                    del self.chunks[i]
                self._save()
            return len(ids)

    # ---------- reads ----------
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

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks.values():
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "name": c.doc_name, "chunks": 0, "pages": set()})
            d["chunks"] += 1
            d["pages"].add(c.page)
        return [{**d, "pages": len(d["pages"])} for d in docs.values()]
