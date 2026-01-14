"""Base provider interfaces for abstracting model implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class EmbeddingProvider(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate text from prompt with streaming."""
        pass


class RerankerProvider(ABC):
    """Abstract base class for reranking models."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        Rerank documents based on relevance to query.
        
        Returns:
            List of (index, score) tuples sorted by score (descending)
        """
        pass
