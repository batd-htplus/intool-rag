"""HTTP-based embedding provider for calling remote model service."""

from typing import List, Optional
import httpx
import asyncio
from rag.config import config
from rag.logging import logger
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
        http_client: Optional[httpx.AsyncClient] = None
    ):
        self.base_url = base_url or config.MODEL_SERVICE_URL
        self.client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0, pool=10.0)
        )
        self._dimension = None
        self._owns_client = http_client is None
        self.max_retries = 3
        self.retry_delay = 2.0

    async def _request_with_retry(self, url: str, payload: dict, retry_count: int = 0):
        """Make HTTP request with retry logic for connection errors."""
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(f"Connection error (attempt {retry_count + 1}/{self.max_retries + 1}), retrying in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
                return await self._request_with_retry(url, payload, retry_count + 1)
            logger.error(f"Connection failed after {self.max_retries + 1} attempts: {str(e)}")
            raise RuntimeError(f"Embedding service unavailable: {str(e)}")
        except httpx.TimeoutException as e:
            raise RuntimeError("Embedding service timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Embedding service error: {e.response.status_code}")

    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding for a single text."""
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

        return []

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
