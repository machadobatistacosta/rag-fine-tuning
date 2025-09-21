import hashlib
import logging
import os
import random
from typing import Any, Dict, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:  # compatibilidade ao importar via "backend.core" ou diretamente de "core"
    from backend.core.llm_generator import LLMGenerator
except ModuleNotFoundError:  # pragma: no cover - caminho utilizado na execucao direta
    from core.llm_generator import LLMGenerator  # type: ignore


class _DeterministicFallbackEmbeddings:
    """Simple deterministic embeddings used when HuggingFace models are unavailable."""

    def __init__(self, embedding_size: int = 768) -> None:
        # Keep this dimension aligned with the primary HuggingFace embedding model.
        self.embedding_size = embedding_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.embedding_size)]


class RAGEngine:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Embeddings em portugues (com fallback deterministico offline)
        self.embeddings = self._load_embeddings()

        # Vector store persistente
        persist_directory = "./data/chroma_db"
        os.makedirs(persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        # Gerador LLM para respostas finais
        self.llm = LLMGenerator()
        if not self.llm.is_ready:
            self.logger.warning(
                "LLM nao inicializado. Motivo: %s", self.llm.load_error or "modelo nao configurado"
            )

    def index_documents(self, chunks: List[Dict[str, str]]) -> None:
        """Indexa chunks de documentos no vectorstore."""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {"source": chunk["source"], "chunk_id": idx} for idx, chunk in enumerate(chunks)
        ]

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        self.vectorstore.persist()

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Busca documentos relevantes e gera resposta."""
        docs = self.vectorstore.similarity_search(question, k=top_k)

        sources = [
            {
                "text": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ]

        if self.llm.is_ready:
            try:
                answer = self.llm.generate(question, docs)
            except Exception as exc:  # noqa: BLE001 - queremos informar o erro ao usuario
                self.logger.exception("Falha ao gerar resposta com o LLM")
                answer = (
                    "Encontrei {count} documentos relevantes, mas o LLM falhou: {erro}."
                ).format(count=len(docs), erro=str(exc))
        else:
            motivo = self.llm.load_error or "modelo nao configurado"
            answer = (
                "Encontrei {count} documentos relevantes, mas o LLM nao esta disponivel ({motivo})."
            ).format(count=len(docs), motivo=motivo)

        return {"answer": answer, "sources": sources}

    def _load_embeddings(self):
        use_hf = os.getenv("RAG_ENABLE_HF_EMBEDDINGS", "0").lower() in {"1", "true", "yes"}
        if not use_hf:
            self.logger.info(
                "Usando embeddings deterministicas em modo offline. Defina RAG_ENABLE_HF_EMBEDDINGS=1 para usar HuggingFace."
            )
            return _DeterministicFallbackEmbeddings()
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
            )
        except Exception as exc:  # noqa: BLE001 - fallback para execucao offline
            self.logger.warning(
                "Falha ao carregar embeddings HuggingFace, usando fallback deterministico: %s",
                exc,
            )
            return _DeterministicFallbackEmbeddings()
