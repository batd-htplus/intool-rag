from typing import List, Dict, Any, Optional
from rag.vector_store.qdrant import vector_store
from rag.config import config
from rag.cache import get_query_cache
from rag.logging import logger
from rag.core.container import get_container
from rag.core.exceptions import EmbeddingError, RetrievalError
from rag.query.prompt import clean_text_multilang


def _extract_metadata_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata fields from payload, excluding the text field.
    
    This ensures consistent metadata structure between ingest and query.
    
    Args:
        payload: Full payload from Qdrant (includes text + metadata)
        
    Returns:
        Metadata dictionary (without text field)
    """
    if not isinstance(payload, dict):
        return {}
    
    # Copy all fields except 'text' (text is separate)
    metadata = {k: v for k, v in payload.items() if k != "text"}
    return metadata


def _is_structured_data_chunk(metadata: dict) -> bool:
    """
    Check if chunk contains structured data.
    
    Args:
        metadata: Chunk metadata
        
    Returns:
        True if chunk contains structured data
    """
    if not metadata:
        return False
    return (
        metadata.get("has_table", False) or
        metadata.get("doc_type") == "table" or
        metadata.get("has_list", False) or
        metadata.get("doc_type") == "list"
    )


def _apply_structured_data_boost(score: float, metadata: dict) -> float:
    """
    Apply boost to structured data chunks.
    
    Args:
        score: Original score
        metadata: Chunk metadata
        
    Returns:
        Boosted score
    """
    if not _is_structured_data_chunk(metadata):
        return score
    
    return score * config.TABLE_BOOST_MULTIPLIER


# Result Object
class QueryResult:
    """Normalized retrieval result"""
    def __init__(self, text: str, score: float, metadata: dict):
        self.text = text
        self.score = score
        self.metadata = metadata


_hybrid_retriever = None

def _get_hybrid_retriever():
    """Get singleton HybridRetriever instance"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from rag.query.hybrid_retriever import HybridRetriever
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


# Retrieval
async def retrieve(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    use_cache: bool = True
) -> List[QueryResult]:
    """
    Precision-first retrieval (noise-safe).
    
    Features:
    - Hybrid search (vector + keyword)
    - Structured data boost
    - Score filtering
    - Result caching
    
    Args:
        question: Query question
        filters: Optional filters
        top_k: Number of results to return
        use_cache: Whether to use query cache
        
    Returns:
        List of QueryResult objects
        
    Raises:
        EmbeddingError: If embedding fails
        RetrievalError: If retrieval fails
    """
    try:
        if not question or not question.strip():
            return []

        top_k = top_k or config.RETRIEVAL_TOP_K
        question = clean_text_multilang(question)
        query_cache = get_query_cache()

        if use_cache:
            cached = query_cache.get(question, filters)
            if cached:
                return [
                    QueryResult(
                        text=r["text"],
                        score=r["score"],
                        metadata=r.get("metadata", {})
                    )
                    for r in cached
                ]

        # Generate embedding
        try:
            embedding_provider = get_container().get_embedding_provider()
            vector = await embedding_provider.embed_single(
                question,
                instruction=config.EMBEDDING_QUERY_INSTRUCTION
            )
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise EmbeddingError(str(e))

        # Perform search
        if config.HYBRID_SEARCH_ENABLED:
            hybrid = _get_hybrid_retriever()
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
            raw_results = [
                {
                    "text": r.payload.get("text", ""),
                    "score": r.score,
                    "metadata": _extract_metadata_from_payload(r.payload)
                }
                for r in vector_results
                if r.payload.get("text")
            ]

        # Normalize & Filter noise
        results: List[QueryResult] = []

        for r in raw_results:
            text = r.get("text") if isinstance(r, dict) else getattr(r, "text", "")
            score = r.get("score") if isinstance(r, dict) else getattr(r, "score", 0.0)
            metadata = r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", {})

            if not text or score <= 0:
                continue

            if score < config.RETRIEVAL_MIN_SCORE:
                continue
            
            # Apply structured data boost (if not already applied in hybrid retriever)
            if not config.HYBRID_SEARCH_ENABLED:
                score = _apply_structured_data_boost(score, metadata)

            results.append(QueryResult(text=text, score=score, metadata=metadata))

        results.sort(key=lambda x: x.score, reverse=True)

        # Cache results
        if use_cache and results:
            query_cache.set(
                question,
                [
                    {
                        "text": r.text,
                        "score": r.score,
                        "metadata": r.metadata
                    }
                    for r in results
                ],
                filters
            )

        return results

    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise RetrievalError(str(e))
