from __future__ import annotations

import hashlib
import io

import fitz
import httpx
import pytest

from backend import main
from backend.core.document_processor import DocumentProcessor


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize(
    ("filename", "content", "content_type"),
    [
        ("DOCUMENT.PDF", b"%PDF-1.4 fake pdf content", "application/pdf"),
        ("NOTES.TXT", b"Plain text content", "text/plain"),
    ],
)
async def test_upload_accepts_uppercase_extensions(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content: bytes,
    content_type: str,
) -> None:
    processed_calls: list[tuple[bytes, str]] = []
    indexed_calls: list[list[dict[str, str]]] = []

    def fake_process_document(file_bytes: bytes, received_filename: str):
        processed_calls.append((file_bytes, received_filename))
        return [{"text": "chunk", "source": received_filename, "doc_id": "dummy"}]

    def fake_index_documents(chunks: list[dict[str, str]]):
        indexed_calls.append(chunks)

    monkeypatch.setattr(main.doc_processor, "process_document", fake_process_document)
    monkeypatch.setattr(main.rag_engine, "index_documents", fake_index_documents)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/documents",
            files={"file": (filename, io.BytesIO(content), content_type)},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"

    assert processed_calls[-1][1] == filename
    assert indexed_calls[-1] == [{"text": "chunk", "source": filename, "doc_id": "dummy"}]


async def test_document_processor_extracts_uppercase_pdf_with_pymupdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_content = "Uppercase PDF Content"

    pdf_document = fitz.open()
    try:
        page = pdf_document.new_page()
        page.insert_text((72, 72), text_content)
        pdf_bytes = pdf_document.tobytes()
    finally:
        pdf_document.close()

    processor = DocumentProcessor()
    expected_doc_id = hashlib.sha256(pdf_bytes).hexdigest()
    opened_with_pymupdf = False
    real_open = fitz.open

    def tracking_open(*args, **kwargs):
        nonlocal opened_with_pymupdf
        opened_with_pymupdf = True
        return real_open(*args, **kwargs)

    monkeypatch.setattr("backend.core.document_processor.fitz.open", tracking_open)

    chunks = processor.process_document(pdf_bytes, "SAMPLE.PDF")

    assert opened_with_pymupdf is True
    assert any(text_content in chunk["text"] for chunk in chunks)
    assert all(chunk["source"] == "SAMPLE.PDF" for chunk in chunks)
    assert all(chunk["doc_id"] == expected_doc_id for chunk in chunks)
