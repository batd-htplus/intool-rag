"""
Gemini Embedding Provider
=========================

Uses Google Gemini API for embeddings.

Design goals:
- Semantic consistency with Gemini semantic analyzer
- Stable embedding dimension
- Safe handling of empty / invalid inputs
- Only embed semantic nodes or queries (never raw PDF)

Used by:
- BUILD phase: embed semantic nodes → FAISS
- QUERY phase: embed user query → FAISS search
"""

from typing import List, Optional
from rag.logging import logger
from rag.config import config
from rag.providers.base import EmbeddingProvider

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiEmbeddingProvider(EmbeddingProvider):
    DIMENSION = 768
    MODEL_NAME = "gemini-embedding-001"

    def __init__(self):
        if not HAS_GEMINI:
            raise RuntimeError("google-genai not installed")

        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing")

        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        logger.info("[EMBED] Gemini embedding provider initialized")

    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.DIMENSION

        payload = text.strip()
        if instruction:
            payload = f"{instruction}\n{payload}"

        try:
            result = self.client.models.embed_content(
                model=self.MODEL_NAME,
                content=payload,
            )
            return list(result.embedding)

        except Exception as e:
            logger.error(f"[EMBED] Gemini embed_single failed: {e}")
            raise

    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        if not texts:
            return []

        embeddings: List[List[float]] = []

        for text in texts:
            if not text or not text.strip():
                embeddings.append([0.0] * self.DIMENSION)
                continue

            payload = text.strip()
            if instruction:
                payload = f"{instruction}\n{payload}"

            try:
                result = self.client.models.embed_content(
                    model=self.MODEL_NAME,
                    content=payload,
                )
                embeddings.append(list(result.embedding))

            except Exception as e:
                logger.error(f"[EMBED] Gemini embed_batch failed: {e}")
                raise

        return embeddings

    def dimension(self) -> int:
        return self.DIMENSION
