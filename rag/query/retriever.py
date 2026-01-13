from typing import List, Dict, Any, AsyncIterator
from rag.vector_store.qdrant import search
from rag.logging import logger

_embedding_model = None

def get_embedding_model():
    """Get embedding model instance (lazy loaded, singleton)"""
    global _embedding_model
    if _embedding_model is None:
        from rag.embedding.http_client import HTTPEmbedding
        _embedding_model = HTTPEmbedding()
    return _embedding_model

class QueryResult:
    """Query result object"""
    def __init__(self, text: str, score: float, metadata: dict):
        self.text = text
        self.score = score
        self.metadata = metadata

async def retrieve(
    question: str,
    filters: Dict[str, Any] = None,
    top_k: int = None
) -> List[QueryResult]:
    """Retrieve relevant chunks from vector store"""
    try:
        from rag.config import config
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        logger.info(f"Retrieving top {top_k} documents for: {question}")
        
        from rag.config import config
        embedding_model = get_embedding_model()
        question_embedding = embedding_model.embed_single(
            question,
            instruction=config.EMBEDDING_QUERY_INSTRUCTION
        )
        
        results = await search(
            vector=question_embedding,
            filters=filters,
            limit=top_k
        )
        
        logger.info(f"Retrieved {len(results)} results")
        
        return results
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise
