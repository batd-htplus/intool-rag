"""HTTP-based reranker provider for calling remote model service."""

from typing import List, Optional
import httpx
from rag.config import config
from rag.logging import logger
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
        http_client: Optional[httpx.AsyncClient] = None
    ):
        self.model_name = model_name or config.RERANKER_MODEL
        self.base_url = base_url or config.MODEL_SERVICE_URL
        self.client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0, pool=10.0)
        )
        self._owns_client = http_client is None

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
        try:
            if not documents:
                return []

            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
            }

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

        except httpx.TimeoutException as e:
            logger.error(f"Reranking timeout: {str(e)}")
            raise RuntimeError("Reranking service timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            logger.error(f"Reranking HTTP error {e.response.status_code}: {str(e)}")
            raise RuntimeError(f"Reranking service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            raise

    async def close(self):
        """Close HTTP client (only if we own it, otherwise managed by DI container)."""
        if self._owns_client:
            await self.client.aclose()
