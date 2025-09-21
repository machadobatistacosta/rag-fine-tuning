from __future__ import annotations

import io
import time

import fitz
import requests


API_ROOT = "http://127.0.0.1:8000/api/v1"
PDF_FILENAME = "integration-test.pdf"
PDF_TEXT = "Integration test document about deduplication."


def _create_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


def _wait_for_source(question: str, expected_source: str, attempts: int = 10, delay: float = 0.2):
    last_response = None
    for _ in range(attempts):
        response = requests.post(
            f"{API_ROOT}/query",
            json={"question": question, "top_k": 5},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        last_response = data
        sources = data.get("sources", [])
        if any(source.get("source") == expected_source for source in sources):
            return sources
        time.sleep(delay)

    raise AssertionError(f"Uploaded document not found in query results: {last_response}")


def test_duplicate_pdf_upload_returns_unique_sources():
    # Garante que a API esta acessivel
    health_response = requests.get(f"{API_ROOT}/health", timeout=10)
    health_response.raise_for_status()

    pdf_bytes = _create_pdf_bytes(PDF_TEXT)

    # Faz upload do mesmo PDF duas vezes
    for _ in range(2):
        files = {
            "file": (PDF_FILENAME, io.BytesIO(pdf_bytes), "application/pdf"),
        }
        upload_response = requests.post(f"{API_ROOT}/documents", files=files, timeout=10)
        upload_response.raise_for_status()
        body = upload_response.json()
        assert body.get("status") == "success"

    sources = _wait_for_source(PDF_TEXT, PDF_FILENAME)

    pdf_sources = [source for source in sources if source.get("source") == PDF_FILENAME]
    assert pdf_sources, "No sources returned for uploaded document"

    unique_entries = {(source["source"], source["text"]) for source in pdf_sources}
    assert len(unique_entries) == len(pdf_sources), "Duplicate snippets returned for the same document"
