"""HTTP-based embedding provider for calling remote model service."""

from typing import List, Optional, Dict, Any
import httpx
import asyncio
from rag.config import config
from rag.logging import logger
from rag.core.exceptions import EmbeddingError, ProviderConnectionError, ProviderTimeoutError
from .base import EmbeddingProvider


class HTTPEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that uses HTTP calls to model service.
    
    Can use either:
    1. Shared HTTP client (recommended for DI) - via DI container
    2. Own HTTP client (standalone mode) - creates new client
    
    Benefits of shared client:
    - Connection pooling across all providers
    - Better resource utilization
    - Reduced connection overhead
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ):
        """
        Initialize HTTP embedding provider.
        
        Args:
            base_url: Base URL for embedding service
            http_client: Optional shared HTTP client (for DI)
            max_retries: Maximum retry attempts (defaults to config)
            retry_delay: Base delay between retries in seconds (defaults to config)
        """
        self.base_url = base_url or config.MODEL_SERVICE_URL
        self.client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.HTTP_READ_TIMEOUT,
                connect=config.HTTP_CONNECT_TIMEOUT,
                read=config.HTTP_READ_TIMEOUT,
                write=config.HTTP_WRITE_TIMEOUT,
                pool=config.HTTP_POOL_TIMEOUT
            )
        )
        self._dimension = None
        self._owns_client = http_client is None
        self.max_retries = max_retries or config.HTTP_MAX_RETRIES
        self.retry_delay = retry_delay or config.HTTP_RETRY_DELAY

    async def _request_with_retry(
        self,
        url: str,
        payload: Dict[str, Any],
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry logic.
        
        Args:
            url: Request URL
            payload: Request payload
            retry_count: Current retry attempt (internal use)
            
        Returns:
            Response JSON data
            
        Raises:
            ProviderConnectionError: If connection fails after all retries
            ProviderTimeoutError: If request times out
            EmbeddingError: For other HTTP errors
        """
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Connection error (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                return await self._request_with_retry(url, payload, retry_count + 1)
            logger.error(f"Connection failed after {self.max_retries + 1} attempts: {str(e)}")
            raise ProviderConnectionError(f"Embedding service unavailable: {str(e)}")
        except httpx.TimeoutException as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Embedding timeout (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                return await self._request_with_retry(url, payload, retry_count + 1)
            logger.error(f"Embedding request timeout after {self.max_retries + 1} attempts: {str(e)}")
            raise ProviderTimeoutError("Embedding service timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if 500 <= status_code < 600 and retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Embedding server error {status_code} (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                return await self._request_with_retry(url, payload, retry_count + 1)
            logger.error(f"Embedding HTTP error {status_code}: {str(e)}")
            raise EmbeddingError(f"Embedding service error: {status_code}")
        except Exception as e:
            logger.error(f"Unexpected embedding error: {str(e)}")
            raise EmbeddingError(f"Unexpected error: {str(e)}")

    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding for a single text.
        
        Uses /embed/single endpoint for better performance when available,
        falls back to /embed endpoint if needed.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Try /embed/single endpoint first (more efficient)
        payload = {
            "text": text,
        }
        if instruction:
            payload["instruction"] = instruction

        try:
            result = await self._request_with_retry(f"{self.base_url}/embed/single", payload)
            if "embedding" in result:
                embedding = result["embedding"]
                if embedding and len(embedding) > 0:
                    return embedding
            raise EmbeddingError("Empty embedding response from /embed/single")
        except (EmbeddingError, httpx.HTTPStatusError) as e:
            # Fallback to /embed endpoint if /embed/single not available
            should_fallback = False
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
                should_fallback = True
            elif isinstance(e, EmbeddingError) and ("404" in str(e) or "not found" in str(e).lower() or "empty" in str(e).lower()):
                should_fallback = True
            
            if not should_fallback:
                raise
            
            logger.debug("Fallback to /embed endpoint for single embedding")
            
            payload = {
                "texts": [text],
            }
            if instruction:
                payload["instruction"] = instruction
            result = await self._request_with_retry(f"{self.base_url}/embed", payload)
            
            if "embeddings" in result and isinstance(result["embeddings"], list) and len(result["embeddings"]) > 0:
                return result["embeddings"][0]
            elif isinstance(result.get("data"), list) and len(result["data"]) > 0:
                return result["data"][0].get("embedding", [])
            
            raise EmbeddingError("Failed to get embedding: empty or invalid response")

    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts with support for large batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]

            payload = {
                "texts": chunk,
            }
            if instruction:
                payload["instruction"] = instruction

            result = await self._request_with_retry(f"{self.base_url}/embed", payload)

            if "embeddings" in result and isinstance(result["embeddings"], list):
                all_embeddings.extend(result["embeddings"])
            elif isinstance(result.get("data"), list):
                for item in result["data"]:
                    embedding = item.get("embedding", [])
                    if embedding:
                        all_embeddings.append(embedding)

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._dimension = config.VECTOR_DIMENSION
        return self._dimension

    async def close(self):
        """Close HTTP client (only if we own it, otherwise managed by DI container)."""
        if self._owns_client:
            await self.client.aclose()
