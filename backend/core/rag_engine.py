import logging
import os
from typing import Any, Dict, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from core.llm_generator import LLMGenerator


class RAGEngine:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Embeddings em portugues
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

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
