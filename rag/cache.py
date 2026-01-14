"""
Caching layer for embeddings and query results.
Supports hash-based embedding cache to avoid re-embedding identical texts.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from rag.config import config
from rag.logging import logger


class EmbeddingCache:
    """Simple file-based embedding cache using content hash"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.CACHE_EMBEDDINGS
    
    def _get_content_hash(self, text: str, model: str = "") -> str:
        """Generate hash of text content for caching"""
        combined = f"{model}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, text: str, model: str = "") -> Optional[List[float]]:
        """Get cached embedding if exists"""
        if not self.enabled:
            return None
        
        try:
            hash_key = self._get_content_hash(text, model)
            cache_file = self.cache_dir / f"{hash_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                return embedding
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
        
        return None
    
    def set(self, text: str, embedding: List[float], model: str = ""):
        """Cache an embedding"""
        if not self.enabled:
            return
        
        try:
            hash_key = self._get_content_hash(text, model)
            cache_file = self.cache_dir / f"{hash_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
    
    def batch_get(self, texts: List[str], model: str = "") -> Dict[int, List[float]]:
        """Get cached embeddings for a batch, returns dict mapping index to embedding"""
        cached = {}
        
        for idx, text in enumerate(texts):
            embedding = self.get(text, model)
            if embedding:
                cached[idx] = embedding
        
        return cached
    
    def batch_set(self, texts: List[str], embeddings: List[List[float]], model: str = ""):
        """Cache a batch of embeddings"""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model)
    
    def clear(self):
        """Clear all cached embeddings"""
        if not self.enabled:
            return
        
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Cache clear error: {str(e)}")


class QueryResultCache:
    """Simple query result cache"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or config.CACHE_DIR / "queries"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.CACHE_EMBEDDINGS
    
    def _get_query_hash(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate hash of query and filters"""
        combined = f"{query}:{json.dumps(filters or {}, sort_keys=True, default=str)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results"""
        if not self.enabled:
            return None
        
        try:
            hash_key = self._get_query_hash(query, filters)
            cache_file = self.cache_dir / f"{hash_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    results = json.load(f)
                return results
        except Exception as e:
            logger.warning(f"Query cache read error: {str(e)}")
        
        return None
    
    def set(self, query: str, results: List[Dict[str, Any]], filters: Optional[Dict] = None):
        """Cache query results"""
        if not self.enabled:
            return
        
        try:
            hash_key = self._get_query_hash(query, filters)
            cache_file = self.cache_dir / f"{hash_key}.json"
            
            # Keep only serializable fields
            serializable_results = []
            for r in results:
                safe_result = {k: v for k, v in r.items() if k not in ['metadata']}
                if 'metadata' in r and isinstance(r['metadata'], dict):
                    safe_result['metadata'] = r['metadata']
                serializable_results.append(safe_result)
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_results, f)
        except Exception as e:
            logger.warning(f"Query cache write error: {str(e)}")
    
    def clear(self):
        """Clear all cached query results"""
        if not self.enabled:
            return
        
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Query cache clear error: {str(e)}")


# Global cache instances
_embedding_cache = None
_query_cache = None


def get_embedding_cache() -> EmbeddingCache:
    """Get singleton embedding cache"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_query_cache() -> QueryResultCache:
    """Get singleton query cache"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryResultCache()
    return _query_cache
