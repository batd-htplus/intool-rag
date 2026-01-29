"""
Embedding Service - Gemini Embeddings (Primary) + HuggingFace Fallback
=====================================================================

Singleton service for embedding operations.

Default: Gemini embeddings (semantic consistency with LLM structure analysis)
Fallback: HuggingFace (if Gemini unavailable)

Used in both phases:
- BUILD: Embed semantic node contents → FAISS index
        (NOT full PDF, only node contents after LLM structure analysis)
- QUERY: Embed user query → FAISS search

Why Gemini Primary:
- Same model analyzes structure AND embeds content
- 100% semantic consistency (same neural network)
- Context-aware embeddings
"""
import threading
import os
from typing import List, Optional
from rag.logging import logger
from rag.providers.base import EmbeddingProvider

_embedding_provider: Optional[EmbeddingProvider] = None
_embedding_lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get embedding provider (lazy loaded, singleton, thread-safe).
    
    Priority:
    1. Gemini embeddings (primary, semantic consistency)
    2. HuggingFace embeddings (fallback, if Gemini unavailable)
    
    Returns:
        Initialized EmbeddingProvider
        
    Raises:
        RuntimeError: If both providers fail
    """
    global _embedding_provider
    if _embedding_provider is None:
        with _embedding_lock:
            if _embedding_provider is None:
                # Try Gemini first (semantic consistency)
                try:
                    logger.info("[EMBED] Initializing Gemini embedding provider (primary)...")
                    from rag.providers.gemini_embedding import GeminiEmbeddingProvider
                    _embedding_provider = GeminiEmbeddingProvider()
                    logger.info("[EMBED] ✓ Using Gemini embeddings (semantic consistency)")
                    return _embedding_provider
                except Exception as e:
                    logger.warning(f"[EMBED] Gemini embeddings unavailable: {e}")
                
                # Fallback to HuggingFace
                try:
                    logger.info("[EMBED] Initializing HuggingFace embedding provider (fallback)...")
                    from rag.providers.langchain_wrapper import HuggingFaceEmbeddingProvider
                    _embedding_provider = HuggingFaceEmbeddingProvider(
                        model_name="BAAI/bge-small-en-v1.5"
                    )
                    logger.info("[EMBED] ✓ Using HuggingFace embeddings (fallback)")
                    return _embedding_provider
                except Exception as e:
                    logger.error(f"[EMBED] Both Gemini and HuggingFace failed: {e}")
                    raise RuntimeError(
                        "Failed to initialize any embedding provider. "
                        "Check Gemini API key and/or HuggingFace installation."
                    )
    return _embedding_provider


async def embed(text: str, instruction: Optional[str] = None) -> List[float]:
    """
    Embed single text.
    
    Args:
        text: Text to embed (query or node content)
        instruction: Optional instruction
        
    Returns:
        Embedding vector
    """
    provider = get_embedding_provider()
    return await provider.embed_single(text, instruction)


async def embed_batch(
    texts: List[str],
    instruction: Optional[str] = None,
    batch_size: int = 32
) -> List[List[float]]:
    """
    Embed batch of texts (semantic node contents).
    
    Args:
        texts: List of texts to embed (node contents, NOT full PDF)
        instruction: Optional instruction
        batch_size: Batch size
        
    Returns:
        List of embedding vectors
    """
    provider = get_embedding_provider()
    return await provider.embed_batch(texts, instruction, batch_size)


def get_embedding_dimension() -> int:
    """
    Get embedding dimension.
    
    Returns:
        Dimension (768 for Gemini, 384 for HuggingFace bge-small)
    """
    provider = get_embedding_provider()
    return provider.get_dimension()


def is_embedding_loaded() -> bool:
    """Check if embedding provider is loaded"""
    return _embedding_provider is not None


# Backward compatibility
def get_embedding_service():
    """Alias for get_embedding_provider (backward compatibility)"""
    return get_embedding_provider()


def get_embedding_model():
    """Alias for get_embedding_provider (backward compatibility)"""
    return get_embedding_provider()



