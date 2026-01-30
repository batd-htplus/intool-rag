"""
Ollama Embeddings Provider
=========================

This ensures:
- Offline support
- Stable dimensions
- FAISS compatibility
"""

from rag.providers.hf.embeddings import HuggingFaceEmbeddingProvider


class OllamaEmbeddingProvider(HuggingFaceEmbeddingProvider):
    """
    Alias for HuggingFaceEmbeddingProvider.

    Ollama is NOT used for embeddings.
    """
    pass
