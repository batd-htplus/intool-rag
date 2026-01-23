"""HTTP-based reranker provider for calling remote model service."""

from typing import List, Optional, Dict, Any
import httpx
from rag.config import config
from rag.logging import logger
from rag.core.exceptions import RerankerError, ProviderConnectionError, ProviderTimeoutError
from .base import RerankerProvider


class HTTPRerankerProvider(RerankerProvider):
    """Reranker provider that uses HTTP calls to model service.
    
    Can use either:
    1. Shared HTTP client (recommended for DI) - via DI container
    2. Own HTTP client (standalone mode) - creates new client
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ):
        """
        Initialize HTTP reranker provider.
        
        Args:
            model_name: Reranker model name
            base_url: Base URL for reranker service
            http_client: Optional shared HTTP client (for DI)
            max_retries: Maximum retry attempts (defaults to config)
            retry_delay: Base delay between retries in seconds (defaults to config)
        """
        self.model_name = model_name or config.RERANKER_MODEL
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
        self._owns_client = http_client is None
        self.max_retries = max_retries or config.HTTP_MAX_RETRIES
        self.retry_delay = retry_delay or config.HTTP_RETRY_DELAY

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query text
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = all)
            
        Returns:
            List of (index, score) tuples sorted by score (descending)
            
        Raises:
            ValueError: If query or documents are empty
            ProviderTimeoutError: If request times out
            RerankerError: For other errors
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not documents:
            return []

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }

        return await self._rerank_with_retry(payload, top_k, retry_count=0)
    
    async def _rerank_with_retry(
        self,
        payload: Dict[str, Any],
        top_k: Optional[int],
        retry_count: int = 0
    ) -> List[tuple]:
        """Rerank with retry logic for connection errors."""
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Expected format: {"results": [{"index": 0, "score": 0.95}, ...]}
            if "results" in result:
                ranked = sorted(
                    result["results"],
                    key=lambda x: x.get("score", 0),
                    reverse=True,
                )
                if top_k:
                    ranked = ranked[:top_k]

                return [(r["index"], r["score"]) for r in ranked]

            logger.warning(f"Unexpected rerank response format: {result}")
            return []

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            if retry_count < self.max_retries:
                import asyncio
                delay = self.retry_delay * (2 ** retry_count)
                logger.warning(
                    f"Reranker connection error (attempt {retry_count + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                return await self._rerank_with_retry(payload, top_k, retry_count + 1)
            logger.error(f"Reranker connection failed after {self.max_retries + 1} attempts: {str(e)}")
            raise ProviderConnectionError(f"Reranker service unavailable: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Reranking timeout: {str(e)}")
            raise ProviderTimeoutError("Reranking service timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("Reranker endpoint not found, returning empty results")
                return []
            logger.error(f"Reranking HTTP error {e.response.status_code}: {str(e)}")
            raise RerankerError(f"Reranking service error: {e.response.status_code}")
        except RerankerError:
            raise
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            raise RerankerError(f"Unexpected error: {str(e)}")

    async def close(self):
        """Close HTTP client (only if we own it, otherwise managed by DI container)."""
        if self._owns_client:
            await self.client.aclose()
