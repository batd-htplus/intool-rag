"""Dependency Injection Container for RAG system."""

from typing import Optional, Dict, Any
import httpx
from rag.config import config
from rag.logging import logger
from rag.providers.base import LLMProvider


class Container:
    """Central dependency injection container for RAG system.
    
    Manages creation and lifecycle of:
    - HTTP client with connection pooling (for LLM calls)
    - LLM provider instance
    
    IMPORTANT: Embeddings are handled by embedding_service.py, NOT container
    
    Benefits:
    - Single HTTP client for all requests (connection pooling)
    - Lazy initialization (load on first use)
    - Centralized dependency management
    - Easy to mock for testing
    """
    
    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._provider_configs: Dict[str, Any] = {}
    
    def get_http_client(self) -> httpx.AsyncClient:
        """Get shared HTTP client with connection pooling.
        
        Uses httpx.Limits for connection pooling:
        - max_connections=100: Max total connections
        - max_keepalive_connections=20: Max persistent connections per host
        
        All HTTP-based providers use this single client → better resource utilization
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
    
    def get_llm_provider(
        self,
        provider_type: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """Get LLM provider instance (lazy loaded, singleton).
        
        Args:
            provider_type: Type of provider ("local").
                         Defaults to "local" (Ollama or HuggingFace backend)
            **kwargs: Additional configuration options (ignored for local)
        
        Returns:
            LLMProvider instance
        
        Examples:
            # Get LLM provider (configured in .env)
            provider = container.get_llm_provider()
        """
        provider_type = provider_type or "local"
        
        sorted_kwargs = tuple(sorted(kwargs.items())) if kwargs else ()
        config_key = f"llm_{provider_type}_{sorted_kwargs}"
        if self._llm_provider and self._provider_configs.get("llm") == config_key:
            return self._llm_provider
        
        if provider_type == "local":
            from rag.providers.llm_provider import LocalLLMProvider
            self._llm_provider = LocalLLMProvider()
        else:
            raise ValueError(f"Unknown LLM provider type: {provider_type}")
        
        self._provider_configs["llm"] = config_key
        return self._llm_provider
    
    async def shutdown(self):
        """Shutdown container and close all resources.
        
        Called at application shutdown to:
        - Close HTTP client connections
        - Close LLM provider connections
        - Release resources
        """
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        
        if self._llm_provider and hasattr(self._llm_provider, "close"):
            await self._llm_provider.close()
        
        self._llm_provider = None
        self._provider_configs = {}


# Global container instance
_global_container: Optional[Container] = None

def get_container() -> Container:
    """Get global container instance (singleton).
    
    Returns:
        Global Container instance
    
    Usage:
        container = get_container()
        llm_provider = container.get_llm_provider()
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
