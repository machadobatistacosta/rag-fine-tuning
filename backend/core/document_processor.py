import hashlib
from typing import Dict, List

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " "],
            keep_separator=True
        )
    
    def process_document(self, content: bytes, filename: str) -> List[Dict[str, str]]:
        """Processa documento e retorna chunks"""
        text = ""

        doc_id = hashlib.sha256(content).hexdigest()
        
        if filename.endswith('.pdf'):
            # Extrair texto do PDF
            pdf_document = fitz.open(stream=content, filetype="pdf")
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        else:
            # Arquivo texto simples
            text = content.decode('utf-8', errors='ignore')
        
        # Dividir em chunks
        chunks = self.text_splitter.split_text(text)
        
        return [
            {"text": chunk, "source": filename, "doc_id": doc_id}
            for chunk in chunks
        ]
