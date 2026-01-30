from typing import List, Optional
import asyncio
from rag.logging import logger
from rag.providers.base import EmbeddingProvider

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_HF = True
except ImportError:
    HAS_HF = False


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Local HuggingFace embedding provider (FALLBACK).

    Used when:
    - Gemini unavailable
    - Offline / air-gapped environments
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        if not HAS_HF:
            raise RuntimeError(
                "LangChain not installed: pip install langchain-community"
            )

        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        probe = self.embeddings.embed_query("dimension probe")
        self._dimension = len(probe)

        logger.info(f"[EMBED] ✓ HF ready (model={model_name}, dim={self._dimension})")

    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dimension

        clean_text = text.strip()

        try:
            return await asyncio.to_thread(
                self.embeddings.embed_query,
                clean_text,
            )
        except Exception as e:
            logger.error(f"[EMBED] HF embed_single failed: {e}")
            raise

    async def embed_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        if not texts:
            return []

        clean_texts = [
            t.strip() if t and t.strip() else ""
            for t in texts
        ]

        try:
            vectors = await asyncio.to_thread(
                self.embeddings.embed_documents,
                clean_texts,
            )

            return [
                v if v else [0.0] * self._dimension
                for v in vectors
            ]

        except Exception as e:
            logger.error(f"[EMBED] HF embed_batch failed: {e}")
            raise

    def dimension(self) -> int:
        return self._dimension
