"""
LangChain Embeddings Wrapper
=============================

Wraps LangChain embeddings for use with our provider system.

Allows swapping embedding implementations:
- OpenAI Embeddings
- HuggingFace (local)
- Cohere
- Others

Complies with EmbeddingProvider interface.
"""

from typing import List, Optional
import os

try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

from rag.logging import logger
from rag.config import config
from rag.providers.base import EmbeddingProvider

class LangChainEmbeddingWrapper(EmbeddingProvider):
    """
    Wraps LangChain embeddings to comply with our EmbeddingProvider interface.
    
    Supports:
    - OpenAI Embeddings (via API key)
    - HuggingFace Embeddings (local)
    """
    
    def __init__(
        self,
        embedding_type: str = "openai",
        **kwargs
    ):
        """
        Initialize LangChain embeddings.
        
        Args:
            embedding_type: "openai" or "huggingface"
            **kwargs: Additional args for embedding model
        """
        if not HAS_LANGCHAIN:
            raise RuntimeError(
                "LangChain not installed: pip install langchain"
            )
        
        self.embedding_type = embedding_type
        self.embeddings = None
        self.dimension = None
        
        self._init_embeddings(**kwargs)
    
    def _init_embeddings(self, **kwargs):
        """Initialize selected embedding model"""
        if self.embedding_type == "openai":
            self._init_openai(**kwargs)
        elif self.embedding_type == "huggingface":
            self._init_huggingface(**kwargs)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    def _init_openai(self, **kwargs):
        """Initialize OpenAI embeddings"""
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        model = kwargs.get("model", "text-embedding-3-small")
        
        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
        )
        
        if "3-small" in model:
            self.dimension = 1536
        elif "3-large" in model:
            self.dimension = 3072
        else:
            self.dimension = 1536  # default
        
        logger.info(f"Initialized OpenAI embeddings: {model} (dim={self.dimension})")
    
    def _init_huggingface(self, **kwargs):
        """Initialize HuggingFace embeddings"""
        model_name = kwargs.get("model_name", "BAAI/bge-small-en-v1.5")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        
        self.dimension = len(self.embeddings.embed_query("test"))
        
        logger.info(f"Initialized HuggingFace embeddings: {model_name} (dim={self.dimension})")
    
    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Embed single text.
        
        Args:
            text: Text to embed
            instruction: Optional instruction (used by some models)
            
        Returns:
            Embedding vector
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")
        
        if instruction:
            text = f"{instruction}\n{text}"
        
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
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
            instruction: Optional instruction
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")
        
        if instruction:
            texts = [f"{instruction}\n{text}" for text in texts]
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            raise RuntimeError("Embeddings not initialized")
        return self.dimension


def create_langchain_embedding_wrapper(
    embedding_type: str = "huggingface",
    **kwargs
) -> LangChainEmbeddingWrapper:
    """
    Create LangChain embedding wrapper.
    
    Args:
        embedding_type: "openai" or "huggingface"
        **kwargs: Model-specific arguments
        
    Returns:
        Wrapper instance
    """
    return LangChainEmbeddingWrapper(embedding_type, **kwargs)
