"""Utilities for loading a local Transformers pipeline used by the RAG engine."""
from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer

import os, torch
from typing import List, Optional

import torch
from langchain.docstore.document import Document
from transformers import Pipeline, pipeline

DEFAULT_PROMPT = (
    "Voce e um assistente corporativo especializado em PMEs. "
    "Utilize apenas o texto em CONTEXTO para responder em portugues claro. "
    "Se a informacao nao estiver no contexto, indique que ela nao foi encontrada.\n\n"
    "CONTEXTO:\n{context}\n\n"
    "PERGUNTA: {question}\n\n"
    "RESPOSTA:"
)


class LLMGenerator:
    """Carrega um modelo local via Transformers para gerar respostas."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_context_chars: Optional[int] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("RAG_LLM_MODEL", "google/flan-t5-base")
        self.task = task or os.getenv("RAG_LLM_TASK", "text2text-generation")
        self.max_new_tokens = max_new_tokens or int(os.getenv("RAG_LLM_MAX_NEW_TOKENS", "256"))
        self.temperature = temperature or float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
        self.top_p = top_p or float(os.getenv("RAG_LLM_TOP_P", "0.9"))
        self.max_context_chars = max_context_chars or int(os.getenv("RAG_LLM_MAX_CONTEXT_CHARS", "6000"))
        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self._pipeline: Optional[Pipeline] = None
        self._load_error: Optional[str] = None
        self._init_pipeline()

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _init_pipeline(self) -> None:
        try:
            self._pipeline = pipeline(
                task=self.task,
                model=self.model_name,
                device=self._resolve_device(),
            )
            self._load_error = None
        except Exception as exc:  # noqa: BLE001 - propagamos via load_error
            self._pipeline = None
            self._load_error = str(exc)

    def _resolve_device(self) -> int:
        device_pref = os.getenv("RAG_LLM_DEVICE", "auto").lower()
        if device_pref == "cpu":
            return -1
        if device_pref == "cuda":
            return 0 if torch.cuda.is_available() else -1
        try:
            return int(device_pref)
        except ValueError:
            return 0 if torch.cuda.is_available() else -1

    def generate(self, question: str, documents: List[Document]) -> str:
        if not self.is_ready:
            raise RuntimeError(
                "LLM nao foi inicializado. Verifique se o modelo esta disponivel e configurado."
            )

        if not documents:
            return "Nenhum documento relevante foi encontrado para responder a pergunta."

        context = self._build_context(documents)
        prompt = self.prompt_template.format(context=context, question=question.strip())
        do_sample = self.temperature > 0

        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=do_sample,
        )

        if isinstance(outputs, list) and outputs:
            generated = outputs[0].get("generated_text") or outputs[0].get("summary_text", "")
        else:
            generated = ""

        if self.task == "text-generation" and generated.startswith(prompt):
            generated = generated[len(prompt) :]

        return generated.strip() or "Nao consegui gerar uma resposta com o modelo configurado."

    def _build_context(self, documents: List[Document]) -> str:
        snippets = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", f"fonte_{idx}")
            snippet = doc.page_content.strip()
            snippets.append(f"[Fonte {idx} | {source}]\n{snippet}")
        context = "\n\n".join(snippets)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]
        return context
