"""Local embedding provider using in-process LLM service."""

from typing import List, Optional
from rag.logging import logger
from rag.core.exceptions import EmbeddingError
from .base import EmbeddingProvider
from rag.llm import get_embedding_model

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using LangChain embeddings."""

    def __init__(self):
        """Initialize local embedding provider"""
        try:
            self.model = get_embedding_model()
            self._dimension = None
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            raise

    async def embed_single(self, text: str, instruction: Optional[str] = None) -> List[float]:
        """Embed single text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embedding = self.model.embed_single(text)
            if not embedding:
                raise EmbeddingError("Empty embedding response")
            return embedding
        except Exception as e:
            logger.error(f"Embedding error for text: {e}")
            raise EmbeddingError(f"Failed to embed text: {str(e)}")

    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Embed batch of texts"""
        if not texts:
            return []
        
        texts = [str(t).strip() for t in texts if t and str(t).strip()]
        if not texts:
            raise ValueError("All texts are empty")
        
        try:
            all_embeddings = self.model.embed_batch(texts)
            
            if not all_embeddings or len(all_embeddings) == 0:
                raise EmbeddingError("Empty embedding response")
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise EmbeddingError(f"Failed to embed batch: {str(e)}")

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            try:
                self._dimension = self.model.get_dimension()
            except Exception as e:
                logger.warning(f"Could not get dimension from model: {e}")
                self._dimension = 768  # Default BGE dimension
        return self._dimension

    async def close(self):
        """Close resources (local embeddings don't need cleanup)"""
        pass
