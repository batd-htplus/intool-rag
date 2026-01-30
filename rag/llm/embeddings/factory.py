import os
from typing import Optional

from rag.logging import logger
from rag.llm.embeddings.base import EmbeddingProvider

_PROVIDER: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """
    Resolve and return the global EmbeddingProvider singleton.

    Priority:
    1. Explicit config via EMBEDDING_PROVIDER
    2. Gemini (if GEMINI_API_KEY exists)
    3. HuggingFace (default / offline / Ollama)
    """
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    provider_name = os.getenv("EMBEDDING_PROVIDER", "").lower()

    if provider_name == "gemini":
        from rag.providers.gemini.embeddings import GeminiEmbeddingProvider
        _PROVIDER = GeminiEmbeddingProvider()
        return _PROVIDER

    if provider_name in ("hf", "huggingface"):
        from rag.providers.hf.embeddings import HuggingFaceEmbeddingProvider
        _PROVIDER = HuggingFaceEmbeddingProvider()
        return _PROVIDER

    if provider_name == "ollama":
        from rag.providers.ollama.embeddings import OllamaEmbeddingProvider
        _PROVIDER = OllamaEmbeddingProvider()
        return _PROVIDER

    if os.getenv("GEMINI_API_KEY"):
        try:
            from rag.providers.gemini.embeddings import GeminiEmbeddingProvider
            _PROVIDER = GeminiEmbeddingProvider()
            return _PROVIDER
        except Exception as e:
            logger.warning(f"[EMBED] Gemini unavailable, fallback to HF: {e}")

    from rag.providers.hf.embeddings import HuggingFaceEmbeddingProvider
    _PROVIDER = HuggingFaceEmbeddingProvider()
    return _PROVIDER
