"""
Local Embedding Model Service - provides singleton embeddings instance.

Uses LangChain embeddings wrapper to support both OpenAI and HuggingFace backends.
No external HTTP dependencies - all embeddings are computed locally or via async APIs.
"""
import threading
import os
from typing import List
from rag.logging import logger
from rag.config import config
from rag.providers.langchain_wrapper import LangChainEmbeddingWrapper

_embedding_model = None
_embedding_lock = threading.Lock()

def get_embedding_model():
    """
    Get embedding model instance (lazy loaded, singleton, thread-safe).
    
    Supports:
    - HuggingFace embeddings (local, EMBEDDING_PROVIDER="huggingface")
    - OpenAI embeddings (async, EMBEDDING_PROVIDER="openai")
    
    Returns:
        LangChainEmbeddingWrapper instance
        
    Raises:
        RuntimeError: If model initialization fails
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                try:
                    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
                    _embedding_model = LangChainEmbeddingWrapper(provider=provider)
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise
    return _embedding_model

def is_embedding_loaded():
    """Check if embedding model is loaded (non-blocking)"""
    return _embedding_model is not None


