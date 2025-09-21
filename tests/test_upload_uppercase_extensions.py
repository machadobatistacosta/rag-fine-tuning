from __future__ import annotations

import io

import httpx
import pytest

from backend import main


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
