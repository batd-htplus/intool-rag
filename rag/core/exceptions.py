"""Standardized exception hierarchy for RAG system."""


class RAGError(Exception):
    """Base exception for all RAG system errors."""
    pass


# Provider errors
class ProviderError(RAGError):
    """Base exception for provider-related errors."""
    pass


class EmbeddingError(ProviderError):
    """Error during embedding operation."""
    pass


class LLMError(ProviderError):
    """Error during LLM generation."""
    pass


class RerankerError(ProviderError):
    """Error during reranking operation."""
    pass


class ProviderConnectionError(ProviderError):
    """Cannot connect to provider service."""
    pass


class ProviderTimeoutError(ProviderError):
    """Provider request timed out."""
    pass


# Retrieval errors
class RetrievalError(RAGError):
    """Error during document retrieval."""
    pass


class VectorSearchError(RetrievalError):
    """Error during vector search."""
    pass


class FilterError(RetrievalError):
    """Error applying filters during retrieval."""
    pass


# Vector store errors
class VectorStoreError(RAGError):
    """Base exception for vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Cannot connect to vector store."""
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


# Ingestion errors
class IngestError(RAGError):
    """Error during document ingestion."""
    pass


class ChunkingError(IngestError):
    """Error during document chunking."""
    pass


class LoadingError(IngestError):
    """Error loading document."""
    pass


# Cache errors
class CacheError(RAGError):
    """Error in caching system."""
    pass


class CacheReadError(CacheError):
    """Error reading from cache."""
    pass


class CacheWriteError(CacheError):
    """Error writing to cache."""
    pass


# Configuration errors
class ConfigError(RAGError):
    """Error in configuration."""
    pass


class ConfigValueError(ConfigError):
    """Invalid configuration value."""
    pass


class ConfigMissingError(ConfigError):
    """Required configuration missing."""
    pass
