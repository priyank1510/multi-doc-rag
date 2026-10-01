from app.chunking import chunk_pages, fixed_chunks, sentence_chunks, split_sentences
from app.ingest import Page, clean_text

TEXT = "First sentence here. Second one is a bit longer than the first. Third! Fourth? Fifth and final sentence."


def test_clean_text_joins_hyphenated_words_and_line_wraps():
    assert clean_text("retri-\neval is\nuseful") == "retrieval is useful"


def test_clean_text_separates_headings_but_not_wrapped_lines():
    text = clean_text("Parental Leave\nOrion provides 16 weeks\nof leave.")
    assert text == "Parental Leave\n\nOrion provides 16 weeks of leave."


def test_split_sentences():
    assert split_sentences(TEXT) == [
        "First sentence here.",
        "Second one is a bit longer than the first.",
        "Third!",
        "Fourth?",
        "Fifth and final sentence.",
    ]


def test_fixed_chunks_overlap():
    chunks = fixed_chunks("abcdefghij", size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_sentence_chunks_never_split_sentences():
    sentences = set(split_sentences(TEXT))
    for chunk in sentence_chunks(TEXT, size=50, overlap=20):
        # every chunk is made of whole sentences only
        rebuilt = chunk
        for s in sentences:
            rebuilt = rebuilt.replace(s, "")
        assert rebuilt.strip() == ""


def test_sentence_chunks_carry_overlap():
    chunks = sentence_chunks(TEXT, size=50, overlap=30)
    assert len(chunks) > 1
    # the last sentence of one chunk reappears at the start of the next
    assert any(a.split(". ")[-1] in b for a, b in zip(chunks, chunks[1:]))


def test_chunk_pages_keeps_citation_metadata():
    pages = [Page(doc_id="d1", doc_name="a.pdf", page=3, text=TEXT)]
    chunks = chunk_pages(pages, "sentence", 60, 10)
    assert all(c.doc_name == "a.pdf" and c.page == 3 for c in chunks)
    assert chunks[0].chunk_id == "d1:p3:c0"
