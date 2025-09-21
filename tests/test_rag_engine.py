from __future__ import annotations

from typing import List

from backend.core.rag_engine import RAGEngine
from langchain.docstore.document import Document


class _DummyVectorStore:
    def __init__(self, documents: List[Document]) -> None:
        self._documents = documents

    def similarity_search(self, question: str, k: int = 5) -> List[Document]:
        return self._documents


class _DummyLLM:
    def __init__(self) -> None:
        self.seen_documents: List[Document] | None = None

    @property
    def is_ready(self) -> bool:  # pragma: no cover - trivial property
        return True

    def generate(self, question: str, documents: List[Document]) -> str:
        self.seen_documents = documents
        return "dummy-answer"


def test_query_deduplicates_documents_and_sources() -> None:
    engine = RAGEngine()

    duplicate_documents = [
        Document(
            page_content="Primeiro chunk do doc1",
            metadata={"source": "doc1.pdf", "doc_id": "doc1", "chunk_id": 0},
        ),
        Document(
            page_content="Duplicado chunk do doc1",
            metadata={"source": "doc1.pdf", "doc_id": "doc1", "chunk_id": 0},
        ),
        Document(
            page_content="Segundo chunk doc1",
            metadata={"source": "doc1.pdf", "doc_id": "doc1", "chunk_id": 1},
        ),
        Document(
            page_content="Primeiro chunk doc2",
            metadata={"source": "doc2.pdf", "doc_id": "doc2", "chunk_id": 0},
        ),
        Document(
            page_content="Duplicado chunk doc2",
            metadata={"source": "doc2.pdf", "doc_id": "doc2", "chunk_id": 0},
        ),
    ]

    engine.vectorstore = _DummyVectorStore(duplicate_documents)

    dummy_llm = _DummyLLM()
    engine.llm = dummy_llm

    response = engine.query("Qual e o conteudo?")

    assert "sources" in response
    assert len(response["sources"]) == 3

    assert dummy_llm.seen_documents is not None
    assert len(dummy_llm.seen_documents) == 3
    assert [doc.page_content for doc in dummy_llm.seen_documents] == [
        "Primeiro chunk do doc1",
        "Segundo chunk doc1",
        "Primeiro chunk doc2",
    ]

    texts_in_order = [source["text"] for source in response["sources"]]
    assert texts_in_order[0].startswith("Primeiro chunk do doc1")
    assert texts_in_order[1].startswith("Segundo chunk doc1")
    assert texts_in_order[2].startswith("Primeiro chunk doc2")
    assert all(text.endswith("...") for text in texts_in_order)

    unique_pairs = {(source["source"], source["text"]) for source in response["sources"]}
    assert len(unique_pairs) == len(response["sources"])
