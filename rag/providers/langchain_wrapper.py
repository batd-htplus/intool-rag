"""
HuggingFace Embeddings Wrapper (Fallback)
==========================================

Local embeddings provider using HuggingFace models.
No external API keys needed - all computation is local.

IMPORTANT: This is a FALLBACK provider.
Primary: Gemini embeddings (semantic consistency with LLM structure analysis)
Fallback: HuggingFace (if Gemini unavailable or for offline-only setups)

Models:
- BAAI/bge-small-en-v1.5 (384 dims, lightweight, recommended)
- BAAI/bge-base-en-v1.5 (768 dims, better quality)

When HuggingFace is used:
- BUILD: Embed semantic node contents (after LLM structure analysis)
- QUERY: Embed user query for FAISS search
"""

from typing import List, Optional
from rag.logging import logger

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

from rag.providers.base import EmbeddingProvider


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Local HuggingFace embeddings - no API keys needed.
    
    Recommended for:
    - BUILD phase: Embed documents offline
    - QUERY phase: Embed queries for FAISS search
    
    Supports:
    - BAAI/bge-small-en-v1.5 (384 dims, default)
    - BAAI/bge-base-en-v1.5 (768 dims, better quality)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize HuggingFace embeddings.
        
        Args:
            model_name: HuggingFace model ID
        """
        if not HAS_LANGCHAIN:
            raise RuntimeError(
                "LangChain not installed: pip install langchain-community"
            )
        
        self.model_name = model_name
        self.embeddings = None
        self.dimension = None
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize HuggingFace embeddings"""
        logger.info(f"[EMBED] Loading HuggingFace model: {self.model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
            
            test_embedding = self.embeddings.embed_query("test")
            self.dimension = len(test_embedding)
            
            logger.info(f"[EMBED] ✓ HuggingFace ready (dim={self.dimension})")
            
        except Exception as e:
            logger.error(f"[EMBED] Failed to initialize HuggingFace: {e}")
            raise
    
    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Embed single text.
        
        Args:
            text: Text to embed
            instruction: Optional instruction (ignored for HuggingFace)
            
        Returns:
            Embedding vector (list of floats)
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")
        
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        try:
            embedding = self.embeddings.embed_query(text.strip())
            return embedding
        except Exception as e:
            logger.error(f"[EMBED] Single embedding failed: {e}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Embed batch of texts.
        
        Args:
            texts: List of texts to embed
            instruction: Optional instruction (ignored for HuggingFace)
            batch_size: Batch size for processing (advisory, HF handles internally)
            
        Returns:
            List of embedding vectors
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")
        
        if not texts:
            return []
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"[EMBED] Batch embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            raise RuntimeError("Embeddings not initialized")
        return self.dimension


def create_huggingface_embedding_provider(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> HuggingFaceEmbeddingProvider:
    """
    Create HuggingFace embedding provider.
    
    Args:
        model_name: HuggingFace model ID
        
    Returns:
        Initialized provider
    """
    return HuggingFaceEmbeddingProvider(model_name)
