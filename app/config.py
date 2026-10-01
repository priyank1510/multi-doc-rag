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
