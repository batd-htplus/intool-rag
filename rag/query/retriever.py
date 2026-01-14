from typing import List, Dict, Any, Optional, AsyncIterator
from rag.vector_store.qdrant import search
from rag.config import config
from rag.cache import get_query_cache
from rag.logging import logger
from rag.core.container import get_container
from rag.core.exceptions import EmbeddingError, RetrievalError

class QueryResult:
    """Query result object with text, score and metadata."""
    def __init__(self, text: str, score: float, metadata: dict):
        self.text = text
        self.score = score
        self.metadata = metadata


async def retrieve(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    use_cache: bool = True
) -> List['QueryResult']:
    """
    Retrieve relevant chunks from vector store with DI container.
    
    Uses hybrid search (vector + BM25) with optional caching and reranking.
    Benefits:
    - Uses shared HTTP client via DI container
    - Connection pooling across multiple queries
    - Cached embeddings when possible
    
    Args:
        question: Query question
        filters: Qdrant filters for metadata
        top_k: Number of results to return
        use_cache: Whether to use query result cache
    
    Returns:
        List of QueryResult objects
    """
    try:
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        query_cache = get_query_cache()
        if use_cache:
            cached_results = query_cache.get(question, filters)
            if cached_results:
                return [
                    QueryResult(
                        text=r.get("text", ""),
                        score=r.get("score", 0.0),
                        metadata=r.get("metadata", {})
                    )
                    for r in cached_results
                ]
        
        # Use DI container for embedding provider
        try:
            embedding_provider = get_container().get_embedding_provider()
            question_embedding = await embedding_provider.embed_single(
                question,
                instruction=config.EMBEDDING_QUERY_INSTRUCTION
            )
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise EmbeddingError(f"Failed to embed question: {str(e)}")
        
        if config.HYBRID_SEARCH_ENABLED:
            from rag.query.hybrid_retriever import HybridRetriever
            hybrid_retriever = HybridRetriever()
            results = await hybrid_retriever.search(
                query=question,
                vector_embedding=question_embedding,
                filters=filters,
                top_k=top_k
            )
            
            query_results = [
                QueryResult(
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {})
                )
                for r in results
            ]
        else:
            results = await search(
                vector=question_embedding,
                filters=filters,
                limit=top_k
            )
            query_results = results
        
        if use_cache and query_results:
            cache_data = [
                {
                    "text": r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in query_results
            ]
            query_cache.set(question, cache_data, filters)
        
        return query_results
    
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise RetrievalError(f"Failed to retrieve documents: {str(e)}")
