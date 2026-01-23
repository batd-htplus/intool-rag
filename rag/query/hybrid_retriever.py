from typing import List, Dict, Any, Optional
import re
from rag.vector_store.qdrant import vector_store
from rag.config import config
from rag.logging import logger
from rag.query.prompt import clean_text_multilang


# =====================================================
# Keyword Scoring (Noise Killer, not true BM25)
# =====================================================

class KeywordScorer:
    """
    Lightweight keyword relevance scorer.
    Designed to FILTER noise, not to rank heavily.
    Safe for Japanese / Vietnamese / English.
    """

    def __init__(self):
        self.min_len = 3

    def extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for scoring."""
        query = clean_text_multilang(query).lower()
        tokens = re.findall(r'\w+', query)
        return [t for t in tokens if len(t) >= self.min_len]

    def score(self, text: str, keywords: List[str]) -> float:
        """
        Score based on keyword presence density.
        Returns value in [0, 1]
        """
        if not text or not keywords:
            return 0.0

        text = clean_text_multilang(text).lower()
        hits = sum(1 for k in keywords if k in text)

        if hits == 0:
            return 0.0

        return hits / len(keywords)

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


def _apply_structured_data_boost(
    score: float,
    metadata: dict,
    base_multiplier: float = None
) -> float:
    """
    Apply boost to structured data chunks.
    
    Structured data chunks often have lower semantic similarity
    but higher factual value, so they need score boosting.
    
    Args:
        score: Original score
        metadata: Chunk metadata
        base_multiplier: Base boost multiplier (from config)
        
    Returns:
        Boosted score
    """
    if not _is_structured_data_chunk(metadata):
        return score
    
    multiplier = base_multiplier or config.TABLE_BOOST_MULTIPLIER
    return score * multiplier


# =====================================================
# Hybrid Retriever
# =====================================================

class HybridRetriever:
    """
    Precision-first hybrid retriever:
    - Vector search (primary)
    - Keyword score (noise filter / light boost)
    - Structured data boost (configurable)
    - Optional reranker (OFF by default)
    """

    def __init__(self):
        self.keyword_scorer = KeywordScorer()
        self.vector_weight = config.VECTOR_WEIGHT
        self.keyword_weight = config.BM25_WEIGHT
        self._reranker = None

    def _get_reranker(self):
        """Get reranker from DI container (optimized - reuse shared instance)"""
        if not config.RERANKER_ENABLED:
            return None

        if self._reranker is None:
            try:
                from rag.core.container import get_container
                container = get_container()
                self._reranker = container.get_reranker_provider()
            except Exception as e:
                logger.warning(f"Reranker disabled: {e}")
                self._reranker = False

        return self._reranker if self._reranker is not False else None

    async def search(
        self,
        query: str,
        vector_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with structured data boost.
        
        Args:
            query: Query text
            vector_embedding: Query vector embedding
            filters: Optional filters
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        if not query or not vector_embedding:
            return []

        candidate_k = max(top_k * 2, 10)

        vector_results = await vector_store.search(
            query_vector=vector_embedding,
            filters=filters,
            limit=candidate_k
        )

        if not vector_results:
            return []

        # Keyword scoring (noise reduction)
        keywords = self.keyword_scorer.extract_keywords(query)

        combined = []
        for r in vector_results:
            payload = r.payload if isinstance(r.payload, dict) else {}
            text = payload.get("text", "")
            vec_score = r.score
            
            meta = {k: v for k, v in payload.items() if k != "text"}

            kw_score = self.keyword_scorer.score(text, keywords)

            # Hard noise filter
            if kw_score == 0.0 and vec_score < config.RETRIEVAL_MIN_SCORE:
                continue

            # Calculate combined score
            combined_score = (
                self.vector_weight * vec_score +
                self.keyword_weight * kw_score
            )
            
            # Apply structured data boost
            combined_score = _apply_structured_data_boost(combined_score, meta)

            combined.append({
                "text": text,
                "score": combined_score,
                "vector_score": vec_score,
                "keyword_score": kw_score,
                "metadata": meta
            })

        if not combined:
            return []

        combined.sort(key=lambda x: x["score"], reverse=True)
        combined = combined[:top_k]

        # Optional reranker (LAST RESORT)
        reranker = self._get_reranker()
        if reranker and len(combined) > 1:
            try:
                texts = [c["text"] for c in combined]
                reranked = await reranker.rerank(
                    query=query,
                    documents=texts,
                    top_k=top_k
                )

                if reranked:
                    reordered = []
                    for idx, score in reranked:
                        if idx < len(combined):
                            item = combined[idx]
                            item["score"] = score
                            item["rerank_score"] = score
                            reordered.append(item)
                    combined = reordered
            except Exception as e:
                logger.warning(f"Rerank failed, fallback to hybrid: {e}")

        return combined
