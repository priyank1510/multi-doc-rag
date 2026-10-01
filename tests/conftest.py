import os
import tempfile

# Must run before `app.config` is imported: isolate test data and force the
# deterministic extractive mode so tests never call (or pay for) the Claude API.
os.environ["RAG_DATA_DIR"] = tempfile.mkdtemp(prefix="rag-test-")
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
