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


# =====================================================
# Hybrid Retriever
# =====================================================

class HybridRetriever:
    """
    Precision-first hybrid retriever:
    - Vector search (primary)
    - Keyword score (noise filter / light boost)
    - Optional reranker (OFF by default)
    """

    def __init__(self):
        self.keyword_scorer = KeywordScorer()
        self.vector_weight = config.VECTOR_WEIGHT
        self.keyword_weight = config.BM25_WEIGHT
        self._reranker = None

    def _get_reranker(self):
        if not config.RERANKER_ENABLED:
            return None

        if self._reranker is None:
            try:
                from rag.providers.reranker_provider import HTTPRerankerProvider
                self._reranker = HTTPRerankerProvider()
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
        top_k = top_k or config.RETRIEVAL_TOP_K

        if not query or not vector_embedding:
            return []

        # =================================================
        # Step 1: Vector search (recall)
        # =================================================
        candidate_k = max(top_k * 2, 10)

        vector_results = await vector_store.search(
            query_vector=vector_embedding,
            filters=filters,
            limit=candidate_k
        )

        if not vector_results:
            return []

        # =================================================
        # Step 2: Keyword scoring (noise reduction)
        # =================================================
        keywords = self.keyword_scorer.extract_keywords(query)

        combined = []
        for r in vector_results:
            payload = r.payload if isinstance(r.payload, dict) else {}
            text = payload.get("text", "")
            vec_score = r.score
            meta = payload

            kw_score = self.keyword_scorer.score(text, keywords)

            # Hard noise filter
            if kw_score == 0.0 and vec_score < config.RETRIEVAL_MIN_SCORE:
                continue

            combined_score = (
                self.vector_weight * vec_score +
                self.keyword_weight * kw_score
            )

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

        # =================================================
        # Step 3: Optional reranker (LAST RESORT)
        # =================================================
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
