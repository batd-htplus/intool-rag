"""HTTP-based embedding provider for calling remote model service."""

from typing import List, Optional
import httpx
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

    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding for a single text."""
        try:
            payload = {
                "texts": [text],
            }
            if instruction:
                payload["instruction"] = instruction

            response = await self.client.post(
                f"{self.base_url}/embed",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            if "embeddings" in result and isinstance(result["embeddings"], list) and len(result["embeddings"]) > 0:
                return result["embeddings"][0]
            elif isinstance(result.get("data"), list) and len(result["data"]) > 0:
                return result["data"][0].get("embedding", [])

            return []

        except httpx.TimeoutException as e:
            logger.error(f"Embedding timeout: {str(e)}")
            raise RuntimeError("Embedding service timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding HTTP error {e.response.status_code}: {str(e)}")
            raise RuntimeError(f"Embedding service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise

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

            try:
                payload = {
                    "texts": chunk,
                }
                if instruction:
                    payload["instruction"] = instruction

                response = await self.client.post(
                    f"{self.base_url}/embed",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                if "embeddings" in result and isinstance(result["embeddings"], list):
                    all_embeddings.extend(result["embeddings"])
                elif isinstance(result.get("data"), list):
                    for item in result["data"]:
                        embedding = item.get("embedding", [])
                        if embedding:
                            all_embeddings.append(embedding)

            except httpx.HTTPStatusError as e:
                logger.error(f"Batch embedding HTTP error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Batch embedding error: {str(e)}", exc_info=True)
                raise

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
