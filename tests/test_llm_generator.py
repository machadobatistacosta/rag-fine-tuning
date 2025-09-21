from __future__ import annotations

from unittest.mock import MagicMock

from backend.core.llm_generator import LLMGenerator
from langchain.docstore.document import Document


def test_generate_uses_zero_temperature(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_LLM", "0")

    generator = LLMGenerator(temperature=0.0)
    mock_pipeline = MagicMock(return_value=[{"generated_text": "resultado"}])
    generator._pipeline = mock_pipeline

    documents = [Document(page_content="conteudo", metadata={"source": "fonte"})]

    result = generator.generate("pergunta?", documents)

    assert result == "resultado"

    assert mock_pipeline.call_count == 1
    _, kwargs = mock_pipeline.call_args
    assert kwargs["temperature"] == 0.0
    assert kwargs["do_sample"] is False
