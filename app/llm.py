"""Step 6: generate a grounded answer from the retrieved chunks.

Two modes:
- Claude (if ANTHROPIC_API_KEY is set): writes a conversational answer that may
  only use the numbered sources and must cite them like [1], [2].
- Extractive fallback (no key needed): picks the sentences from the retrieved
  chunks that are most similar to the question. Always grounded, never fluent.
"""
import logging
import os
import re

import numpy as np

from app import config
from app.chunking import Chunk, split_sentences
from app.embeddings import embed

log = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions about the user's uploaded documents.

Rules:
- Use ONLY the numbered sources in the user message. Do not use outside knowledge.
- Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise: a few sentences or a short list."""

REWRITE_PROMPT = """Rewrite the user's latest message as a standalone search query, resolving
pronouns and references using the conversation. Reply with the query only."""

_FOLLOW_UP = re.compile(r"\b(it|its|that|this|those|these|they|them|their|he|she|there)\b", re.I)


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def _client():
    import anthropic  # imported lazily so the app runs without the SDK configured

    return anthropic.Anthropic()


def _call_claude(system: str, messages: list[dict], max_tokens: int) -> str:
    response = _client().beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={"effort": config.LLM_EFFORT},
        # If a safety classifier declines, the API retries on a fallback model
        # inside the same call instead of returning an empty refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":  # the whole fallback chain declined
        return NO_ANSWER
    return "".join(b.text for b in response.content if b.type == "text").strip()


def format_sources(contexts: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.doc_name}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(contexts, start=1)
    )


# ---------- conversational query rewriting ----------
def rewrite_question(question: str, history: list[dict]) -> str:
    """Make follow-ups like "what about its refund policy?" searchable on their own."""
    if not history:
        return question
    if llm_available():
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        try:
            return _call_claude(
                REWRITE_PROMPT,
                [{"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"}],
                max_tokens=200,
            ) or question
        except Exception:
            log.exception("Query rewrite failed; using raw question")
    # Heuristic fallback: short or pronoun-heavy questions borrow the previous user turn.
    if len(question.split()) <= 4 or _FOLLOW_UP.search(question):
        previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{previous} {question}".strip()
    return question


# ---------- answer generation ----------
def extractive_answer(
    question: str, contexts: list[tuple[Chunk, float]], max_sentences: int = 3, margin: float = 0.15
) -> str:
    first_seen: dict[str, int] = {}  # sentence -> source number; dedupes chunk overlap
    for i, (c, _) in enumerate(contexts, start=1):
        for s in split_sentences(c.text):
            if len(s.split()) >= 4:  # skip bare headings
                first_seen.setdefault(s, i)
    candidates = [(i, s) for s, i in first_seen.items()]
    if not candidates:
        return NO_ANSWER
    scores = embed([text for _, text in candidates]) @ embed([question])[0]
    ranked = np.argsort(-scores)[:max_sentences]
    # Keep only sentences nearly as relevant as the best one, so a single-fact
    # answer isn't padded with loosely related filler.
    keep = [j for j in ranked if scores[j] >= scores[ranked[0]] - margin]
    # Keep the selected sentences in document order so the answer reads naturally.
    return " ".join(f"{candidates[j][1]} [{candidates[j][0]}]" for j in sorted(keep))


def generate_answer(question: str, contexts: list[tuple[Chunk, float]], history: list[dict]) -> tuple[str, str]:
    """Returns (answer, mode) where mode is "claude" or "extractive"."""
    if not contexts:
        return NO_ANSWER, "none"
    if llm_available():
        messages = history[-config.MAX_HISTORY_TURNS * 2 :] + [
            {"role": "user", "content": f"Sources:\n{format_sources(contexts)}\n\nQuestion: {question}"}
        ]
        try:
            return _call_claude(SYSTEM_PROMPT, messages, max_tokens=2000), "claude"
        except Exception:
            log.exception("Claude call failed; falling back to extractive answer")
    return extractive_answer(question, contexts), "extractive"
