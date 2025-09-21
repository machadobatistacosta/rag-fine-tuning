from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

try:  # Permite executar como pacote ou script isolado
    from backend.core.rag_engine import RAGEngine
    from backend.core.document_processor import DocumentProcessor
except ModuleNotFoundError:  # pragma: no cover - compatibilidade para execucao direta
    from core.rag_engine import RAGEngine  # type: ignore
    from core.document_processor import DocumentProcessor  # type: ignore

app = FastAPI(title="IA Corporativa PMEs", version="0.1.0")

# CORS para desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
rag_engine = RAGEngine()
doc_processor = DocumentProcessor()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Busca informações nos documentos indexados"""
    try:
        results = rag_engine.query(request.question, request.top_k)
        return QueryResponse(
            answer=results["answer"],
            sources=results["sources"],
        )
    except Exception as e:  # noqa: BLE001 - expor erro simplificado via HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents")
async def upload_document(file: UploadFile = File(...)):
    """Upload e indexação de novo documento"""
    try:
        # Validar tipo de arquivo
        if not file.filename.endswith((".pdf", ".txt")):
            raise HTTPException(400, "Apenas PDF e TXT são suportados")

        # Processar e indexar
        content = await file.read()
        chunks = doc_processor.process_document(content, file.filename)
        rag_engine.index_documents(chunks)

        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
