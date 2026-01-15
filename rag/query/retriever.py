from typing import List, Dict, Any, Optional
import hashlib
from rag.vector_store.qdrant import vector_store
from rag.config import config
from rag.cache import get_query_cache
from rag.logging import logger
from rag.core.container import get_container
from rag.core.exceptions import EmbeddingError, RetrievalError
from rag.query.prompt import clean_text_multilang


# =====================================================
# Result Object
# =====================================================

class QueryResult:
    """Normalized retrieval result"""
    def __init__(self, text: str, score: float, metadata: dict):
        self.text = text
        self.score = score
        self.metadata = metadata


# =====================================================
# Cache Key
# =====================================================

def _build_cache_key(
    question: str,
    filters: Optional[Dict[str, Any]],
    top_k: int
) -> str:
    raw = f"{question}|{filters}|{top_k}|{config.HYBRID_SEARCH_ENABLED}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# =====================================================
# Retrieval
# =====================================================

async def retrieve(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    use_cache: bool = True
) -> List[QueryResult]:
    """
    Precision-first retrieval (noise-safe)
    """
    try:
        if not question or not question.strip():
            return []

        top_k = top_k or config.RETRIEVAL_TOP_K

        # Clean & normalize query (VERY IMPORTANT)
        question = clean_text_multilang(question)

        cache_key = _build_cache_key(question, filters, top_k)
        query_cache = get_query_cache()

        if use_cache:
            cached = query_cache.get(cache_key)
            if cached:
                return [
                    QueryResult(
                        text=r["text"],
                        score=r["score"],
                        metadata=r.get("metadata", {})
                    )
                    for r in cached
                ]

        # =================================================
        # Embedding
        # =================================================
        try:
            embedding_provider = get_container().get_embedding_provider()
            vector = await embedding_provider.embed_single(
                question,
                instruction=config.EMBEDDING_QUERY_INSTRUCTION
            )
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise EmbeddingError(str(e))

        # =================================================
        # Search
        # =================================================
        if config.HYBRID_SEARCH_ENABLED:
            from rag.query.hybrid_retriever import HybridRetriever
            hybrid = HybridRetriever()
            raw_results = await hybrid.search(
                query=question,
                vector_embedding=vector,
                filters=filters,
                top_k=top_k
            )
        else:
            vector_results = await vector_store.search(
                query_vector=vector,
                filters=filters,
                limit=top_k
            )
            # Convert VectorSearchResult to dict format
            raw_results = [
                {
                    "text": r.payload.get("text", ""),
                    "score": r.score,
                    "metadata": r.payload
                }
                for r in vector_results
                if r.payload.get("text")
            ]

        # =================================================
        # Normalize & Filter noise
        # =================================================
        results: List[QueryResult] = []

        for r in raw_results:
            text = r.get("text") if isinstance(r, dict) else getattr(r, "text", "")
            score = r.get("score") if isinstance(r, dict) else getattr(r, "score", 0.0)
            metadata = r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", {})

            if not text or score <= 0:
                continue

            # Optional hard threshold (noise killer)
            if score < config.RETRIEVAL_MIN_SCORE:
                continue

            results.append(QueryResult(text=text, score=score, metadata=metadata))

        # Sort by score (safety)
        results.sort(key=lambda x: x.score, reverse=True)

        # =================================================
        # Cache
        # =================================================
        if use_cache and results:
            query_cache.set(
                cache_key,
                [
                    {
                        "text": r.text,
                        "score": r.score,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            )

        return results

    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise RetrievalError(str(e))
