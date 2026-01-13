"""
Embedding model wrapper for model service
"""
from typing import List
from model_service.logging import logger
from model_service.config import config

_embedding_model = None

def get_embedding_model():
    """Get embedding model instance (lazy loaded, singleton)"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from model_service.embedding.bge_m3 import BGEEmbedding
            logger.info("Loading embedding model...")
            _embedding_model = BGEEmbedding()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    return _embedding_model

