"""Dependency Injection Container for RAG system."""

from typing import Optional, Dict, Any
import httpx
from rag.config import config
from rag.logging import logger
from rag.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from rag.providers.embedding_provider import LocalEmbeddingProvider
from rag.providers.llm_provider import LocalLLMProvider
from rag.providers.reranker_provider import HTTPRerankerProvider

class Container:
    """Central dependency injection container for RAG system.
    
    Manages creation and lifecycle of:
    - HTTP client with connection pooling
    - Provider instances (embedding, LLM, reranker)
    - Caching layers
    
    Benefits:
    - Single HTTP client for all providers (connection pooling)
    - Lazy initialization (load on first use)
    - Centralized dependency management
    - Easy to mock for testing
    """
    
    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._reranker_provider: Optional[RerankerProvider] = None
        self._provider_configs: Dict[str, Any] = {}
    
    def get_http_client(self) -> httpx.AsyncClient:
        """Get shared HTTP client with connection pooling.
        
        Uses httpx.Limits for connection pooling:
        - max_connections=100: Max total connections
        - max_keepalive_connections=20: Max persistent connections per host
        
        All providers use this single client → better resource utilization
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    config.HTTP_READ_TIMEOUT,
                    connect=config.HTTP_CONNECT_TIMEOUT,
                    read=config.HTTP_READ_TIMEOUT,
                    write=config.HTTP_WRITE_TIMEOUT,
                    pool=config.HTTP_POOL_TIMEOUT
                ),
                limits=httpx.Limits(
                    max_connections=config.HTTP_MAX_CONNECTIONS,
                    max_keepalive_connections=config.HTTP_MAX_KEEPALIVE_CONNECTIONS
                )
            )
        return self._http_client
    
    def get_embedding_provider(
        self,
        provider_type: Optional[str] = None,
        **kwargs
    ) -> EmbeddingProvider:
        """Get embedding provider instance (lazy loaded, singleton).
        
        Args:
            provider_type: Type of provider ("local", "http").
                         Defaults to "local" (in-process LLM service)
            **kwargs: Additional configuration options (ignored for local)
        
        Returns:
            EmbeddingProvider instance
        
        Examples:
            # Use local embedding provider (default)
            provider = container.get_embedding_provider()
            
            # Local in-process embeddings
            provider = container.get_embedding_provider("local")
        """
        provider_type = provider_type or "local"
        
        # Generate stable config key
        sorted_kwargs = tuple(sorted(kwargs.items())) if kwargs else ()
        config_key = f"embedding_{provider_type}_{sorted_kwargs}"
        if self._embedding_provider and self._provider_configs.get("embedding") == config_key:
            return self._embedding_provider
        
        if provider_type == "local":
            self._embedding_provider = LocalEmbeddingProvider()
        else:
            raise ValueError(f"Unknown embedding provider type: {provider_type}")
        
        self._provider_configs["embedding"] = config_key
        return self._embedding_provider
    
    def get_llm_provider(
        self,
        provider_type: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """Get LLM provider instance (lazy loaded, singleton).
        
        Args:
            provider_type: Type of provider ("local", "http").
                         Defaults to "local" (in-process LLM service)
            **kwargs: Additional configuration options (ignored for local)
        
        Returns:
            LLMProvider instance
        
        Examples:
            # Use local LLM provider (default, no external services)
            provider = container.get_llm_provider()
            
            # Local in-process LLM (Ollama or HuggingFace)
            provider = container.get_llm_provider("local")
        """
        provider_type = provider_type or "local"
        
        sorted_kwargs = tuple(sorted(kwargs.items())) if kwargs else ()
        config_key = f"llm_{provider_type}_{sorted_kwargs}"
        if self._llm_provider and self._provider_configs.get("llm") == config_key:
            return self._llm_provider
        
        if provider_type == "local":
            self._llm_provider = LocalLLMProvider()
        else:
            raise ValueError(f"Unknown LLM provider type: {provider_type}")
        
        self._provider_configs["llm"] = config_key
        return self._llm_provider
    
    def get_reranker_provider(
        self,
        provider_type: Optional[str] = None,
        **kwargs
    ) -> Optional[RerankerProvider]:
        """Get reranker provider instance (lazy loaded, singleton).
        
        Args:
            provider_type: Type of provider ("http", "local").
                         Defaults to config.RERANKER_PROVIDER_TYPE
            **kwargs: Additional configuration options
        
        Returns:
            RerankerProvider instance or None if reranking disabled
        
        Examples:
            # Use configured reranker
            provider = container.get_reranker_provider()
            
            # Get None if reranking disabled
            if not config.RERANKER_ENABLED:
                provider = None
        """
        if not config.RERANKER_ENABLED:
            return None
        
        provider_type = provider_type or config.RERANKER_PROVIDER_TYPE
        
        sorted_kwargs = tuple(sorted(kwargs.items())) if kwargs else ()
        config_key = f"reranker_{provider_type}_{sorted_kwargs}"
        if self._reranker_provider and self._provider_configs.get("reranker") == config_key:
            return self._reranker_provider
        
        if provider_type == "http":
            base_url = kwargs.get("base_url", config.MODEL_SERVICE_URL)
            self._reranker_provider = HTTPRerankerProvider(
                base_url=base_url,
                http_client=self.get_http_client()
            )
        else:
            raise ValueError(f"Unknown reranker provider type: {provider_type}")
        
        self._provider_configs["reranker"] = config_key
        return self._reranker_provider
    
    async def shutdown(self):
        """Shutdown container and close all resources.
        
        Called at application shutdown to:
        - Close HTTP client connections
        - Close provider connections
        - Release resources
        """
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        
        if self._embedding_provider and hasattr(self._embedding_provider, "close"):
            await self._embedding_provider.close()
        
        if self._llm_provider and hasattr(self._llm_provider, "close"):
            await self._llm_provider.close()
        
        if self._reranker_provider and hasattr(self._reranker_provider, "close"):
            await self._reranker_provider.close()
        
        self._embedding_provider = None
        self._llm_provider = None
        self._reranker_provider = None
        self._provider_configs = {}


# Global container instance
_global_container: Optional[Container] = None

def get_container() -> Container:
    """Get global container instance (singleton).
    
    Returns:
        Global Container instance
    
    Usage:
        container = get_container()
        embedding_provider = container.get_embedding_provider()
    """
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container

async def shutdown_container():
    """Shutdown global container instance.
    
    Called at application shutdown.
    """
    global _global_container
    if _global_container:
        await _global_container.shutdown()
        _global_container = None
