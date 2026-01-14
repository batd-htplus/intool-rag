"""Abstract base classes for vector store implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Result from vector search operation."""
    id: str
    payload: Dict[str, Any]
    score: float


class VectorStore(ABC):
    """Abstract base class for vector store implementations.
    
    Supports:
    - Multiple implementations: Qdrant, Pinecone, Weaviate, Milvus, etc.
    - Consistent interface for RAG system
    - Easy switching without changing RAG code
    
    Example implementations:
    - QdrantVectorStore: rag/vector_store/qdrant.py
    - PineconeVectorStore: rag/vector_store/pinecone.py (future)
    - WeaviateVectorStore: rag/vector_store/weaviate.py (future)
    """
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            filters: Optional metadata filters (dict or query object)
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of VectorSearchResult with metadata and scores
        
        Raises:
            VectorStoreError: If search fails
        """
        pass
    
    @abstractmethod
    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Insert or update vectors with metadata.
        
        Args:
            ids: List of document IDs
            vectors: List of embedding vectors (same length as ids)
            payloads: List of metadata dicts (same length as ids)
        
        Raises:
            VectorStoreError: If upsert fails
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete vectors by ID or filters.
        
        Args:
            ids: List of document IDs to delete (empty = use filters)
            filters: Optional filter conditions (if no ids provided)
        
        Raises:
            VectorStoreError: If delete fails
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Delete all vectors (reset collection/index).
        
        Raises:
            VectorStoreError: If clear fails
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dict with keys:
            - vector_count: Number of vectors
            - dimension: Vector dimension
            - collection_name: Collection/index name
            - storage_size: Approximate storage size
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connection to vector store.
        
        Called at application shutdown.
        """
        pass


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Error connecting to vector store."""
    pass


class VectorStoreSearchError(VectorStoreError):
    """Error during search operation."""
    pass


class VectorStoreUpsertError(VectorStoreError):
    """Error during upsert operation."""
    pass


class VectorStoreDeleteError(VectorStoreError):
    """Error during delete operation."""
    pass
