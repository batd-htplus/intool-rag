"""
Hybrid search combining vector search with BM25 keyword search.
Includes optional reranking with cross-encoder models.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from rag.vector_store.qdrant import search as vector_search
from rag.config import config
from rag.logging import logger


class BM25Retriever:
    """Simple BM25-like keyword retriever using Qdrant payload search"""
    
    def __init__(self):
        self.min_word_length = 3
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) >= self.min_word_length]
    
    async def search(
        self,
        query: str,
        documents_with_scores: List[Tuple[str, str, float, Dict]],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using BM25-like scoring.
        
        Args:
            query: Query text
            documents_with_scores: List of (id, text, vector_score, metadata)
            top_k: Number of results to return
        
        Returns:
            List of (index_in_documents, bm25_score) tuples
        """
        query_tokens = set(self._tokenize(query))
        
        if not query_tokens:
            # If query has no valid tokens, return as-is with low BM25 scores
            return [(i, 0.0) for i in range(min(top_k, len(documents_with_scores)))]
        
        scored_docs = []
        
        for idx, (doc_id, text, vec_score, metadata) in enumerate(documents_with_scores):
            doc_tokens = self._tokenize(text)
            
            if not doc_tokens:
                scored_docs.append((idx, 0.0))
                continue
            
            # Simple BM25-like scoring: proportion of query terms found
            matching_terms = sum(1 for token in query_tokens if token in doc_tokens)
            bm25_score = matching_terms / len(query_tokens) if query_tokens else 0.0
            
            scored_docs.append((idx, bm25_score))
        
        # Sort by BM25 score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


class HybridRetriever:
    """Hybrid retriever combining vector and keyword search"""
    
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.vector_weight = config.VECTOR_WEIGHT
        self.bm25_weight = config.BM25_WEIGHT
        self._reranker = None
    
    def _get_reranker(self):
        """Lazy load reranker"""
        if self._reranker is None and config.RERANKER_ENABLED:
            try:
                from rag.providers.reranker_provider import HTTPRerankerProvider
                self._reranker = HTTPRerankerProvider()
            except Exception as e:
                logger.warning(f"Failed to load reranker: {str(e)}")
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
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Query text
            vector_embedding: Query embedding
            filters: Qdrant filters
            top_k: Number of results to return
        
        Returns:
            List of result dicts with combined scores
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        # Step 1: Vector search (get more candidates for hybrid scoring)
        retrieval_limit = int(top_k * 2)
        
        vector_results = await vector_search(
            vector=vector_embedding,
            filters=filters,
            limit=retrieval_limit
        )
        
        if not vector_results:
            return []
        
        # Prepare documents for BM25
        docs_for_bm25 = [
            (str(i), r.text, r.score, r.metadata)
            for i, r in enumerate(vector_results)
        ]
        
        # Step 2: BM25 reranking
        bm25_scores = await self.bm25.search(query, docs_for_bm25, top_k=retrieval_limit)
        
        # Step 3: Combine scores
        combined_results = []
        
        for idx, bm25_score in bm25_scores:
            original_result = vector_results[idx]
            
            vector_score_norm = original_result.score
            bm25_score_norm = bm25_score
            
            combined_score = (
                self.vector_weight * vector_score_norm +
                self.bm25_weight * bm25_score_norm
            )
            
            combined_results.append({
                "index": idx,
                "text": original_result.text,
                "score": combined_score,
                "vector_score": vector_score_norm,
                "bm25_score": bm25_score_norm,
                "metadata": original_result.metadata
            })
        
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Step 4: Optional reranking with cross-encoder
        reranker = self._get_reranker()
        if reranker and len(combined_results) > 0:
            try:
                rerank_docs = [r["text"] for r in combined_results]
                reranked_indices = await reranker.rerank(
                    query=query,
                    documents=rerank_docs,
                    top_k=top_k
                )
                
                if reranked_indices:
                    reordered = []
                    for orig_idx, rerank_score in reranked_indices:
                        if orig_idx < len(combined_results):
                            result = combined_results[orig_idx].copy()
                            result["rerank_score"] = rerank_score
                            result["score"] = rerank_score
                            reordered.append(result)
                    
                    combined_results = reordered
            except Exception as e:
                pass
        
        return combined_results[:top_k]
