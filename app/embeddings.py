"""Step 3: turn text into vectors with a transformer embedding model.

Vectors are L2-normalized, so a dot product between two of them equals their
cosine similarity. That lets FAISS use a fast inner-product index.
"""
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
