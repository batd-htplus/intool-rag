"""Dependency Injection Container for RAG system."""

from typing import Optional, Dict, Any
import httpx
from rag.config import config
from rag.logging import logger
from rag.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from rag.providers.embedding_provider import HTTPEmbeddingProvider
from rag.providers.llm_provider import HTTPLLMProvider
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
                timeout=httpx.Timeout(120.0, connect=10.0, read=120.0, write=10.0, pool=10.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20
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
            provider_type: Type of provider ("http", "openai", "local"). 
                         Defaults to config.EMBEDDING_PROVIDER_TYPE
            **kwargs: Additional configuration options
        
        Returns:
            EmbeddingProvider instance
        
        Examples:
            # Use default HTTP provider
            provider = container.get_embedding_provider()
            
            # Use specific provider
            provider = container.get_embedding_provider("http")
            
            # With custom base URL
            provider = container.get_embedding_provider(base_url="http://localhost:8002")
        """
        provider_type = provider_type or config.EMBEDDING_PROVIDER_TYPE
        
        config_key = f"embedding_{provider_type}_{str(kwargs)}"
        if self._embedding_provider and self._provider_configs.get("embedding") == config_key:
            return self._embedding_provider
        
        if provider_type == "http":
            base_url = kwargs.get("base_url", config.MODEL_SERVICE_URL)
            self._embedding_provider = HTTPEmbeddingProvider(
                base_url=base_url,
                http_client=self.get_http_client()
            )
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
            provider_type: Type of provider ("http", "openai", "local").
                         Defaults to config.LLM_PROVIDER_TYPE
            **kwargs: Additional configuration options
        
        Returns:
            LLMProvider instance
        
        Examples:
            # Use default HTTP provider
            provider = container.get_llm_provider()
            
            # Use specific provider
            provider = container.get_llm_provider("http")
        """
        provider_type = provider_type or config.LLM_PROVIDER_TYPE
        
        config_key = f"llm_{provider_type}_{str(kwargs)}"
        if self._llm_provider and self._provider_configs.get("llm") == config_key:
            return self._llm_provider
        
        if provider_type == "http":
            base_url = kwargs.get("base_url", config.MODEL_SERVICE_URL)
            self._llm_provider = HTTPLLMProvider(
                base_url=base_url,
                http_client=self.get_http_client()
            )
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
        
        config_key = f"reranker_{provider_type}_{str(kwargs)}"
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
