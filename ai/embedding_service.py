"""
Embedding model wrapper for model service
"""
import threading
from typing import List
from ai.logging import logger
from ai.config import config

_embedding_model = None
_embedding_lock = threading.Lock()

def get_embedding_model():
    """Get embedding model instance (lazy loaded, singleton, thread-safe)"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                try:
                    from ai.embedding.bge_m3 import BGEEmbedding
                    _embedding_model = BGEEmbedding()
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise
    return _embedding_model

def is_embedding_loaded():
    """Check if embedding model is loaded (non-blocking)"""
    return _embedding_model is not None

