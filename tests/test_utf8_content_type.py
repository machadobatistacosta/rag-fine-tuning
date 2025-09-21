from __future__ import annotations

import io
import time

import requests


API_ROOT = "http://127.0.0.1:8000/api/v1"
ACCENTED_FILENAME = "integration-utf8.txt"
ACCENTED_TEXT = "Informação estratégica: João lidera a operação diária."


def _wait_for_accented_fragment(
    question: str,
    expected_fragment: str,
    attempts: int = 10,
    delay: float = 0.2,
):
    last_response: requests.Response | None = None
    for _ in range(attempts):
        response = requests.post(
            f"{API_ROOT}/query",
            json={"question": question, "top_k": 5},
            timeout=10,
        )
        response.raise_for_status()
        last_response = response
        if expected_fragment in response.text:
            return response
        time.sleep(delay)

    detail = None if last_response is None else last_response.text
    raise AssertionError(
        f"Accented fragment '{expected_fragment}' not found in query response: {detail}"
    )


def test_json_responses_advertise_utf8_and_preserve_accents():
    health_response = requests.get(f"{API_ROOT}/health", timeout=10)
    health_response.raise_for_status()
    assert (
        health_response.headers.get("content-type")
        == "application/json; charset=utf-8"
    )

    files = {
        "file": (
            ACCENTED_FILENAME,
            io.BytesIO(ACCENTED_TEXT.encode("utf-8")),
            "text/plain",
        )
    }
    upload_response = requests.post(
        f"{API_ROOT}/documents", files=files, timeout=10
    )
    upload_response.raise_for_status()

    query_response = _wait_for_accented_fragment(
        ACCENTED_TEXT, "Informação"
    )
    assert (
        query_response.headers.get("content-type")
        == "application/json; charset=utf-8"
    )

    data = query_response.json()
    sources_text = " ".join(
        source.get("text", "") for source in data.get("sources", [])
    )
    assert "Informação" in sources_text
    assert "João" in sources_text
