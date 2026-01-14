from typing import List, Dict, Any, Optional
import uuid
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue,
    HnswConfigDiff
)
from rag.config import config
from rag.logging import logger
from rag.vector_store.base import VectorStore, VectorSearchResult, VectorStoreError, VectorStoreConnectionError, VectorStoreSearchError, VectorStoreUpsertError, VectorStoreDeleteError

class QdrantVectorStore(VectorStore):
    """Qdrant vector store with optimized HNSW configuration"""
    
    def __init__(self):
        """Initialize Qdrant vector store with optimized HNSW configuration."""
        self.url = config.QDRANT_URL
        self.collection = config.QDRANT_COLLECTION
        self.vector_size = config.VECTOR_DIMENSION
        
        try:
            self.client = QdrantClient(url=self.url)
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise VectorStoreConnectionError(f"Cannot connect to Qdrant at {self.url}: {str(e)}")
        
        try:
            collection_info = self.client.get_collection(self.collection)
            logger.info(f"Connected to Qdrant collection: {self.collection}")
        except Exception as get_e:
            error_str = str(get_e).lower()
            if "validation" in error_str or "parsingmodel" in error_str:
                logger.warning(f"Qdrant client version mismatch, using REST API")
                return
            elif "not found" not in error_str and "404" not in error_str:
                raise
            
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=config.QDRANT_HNSW_M,
                        ef_construct=config.QDRANT_HNSW_EF_CONSTRUCT,
                    ),
                )
                self._create_payload_indexes()
            except Exception as e:
                error_str = str(e)
                if "already exists" in error_str.lower() or "409" in error_str:
                    logger.info(f"Collection {self.collection} already exists")
                else:
                    raise VectorStoreConnectionError(f"Failed to create collection: {str(e)}")    
    def _create_payload_indexes(self):
        """Create indexes on common metadata fields for faster filtering"""
        for field in ["project", "doc_id", "language"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword"
                )
            except Exception:
                pass

    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Insert or update vectors with metadata (implements VectorStore interface)."""
        try:
            if len(vectors) != len(payloads) or len(vectors) != len(ids):
                raise ValueError("ids, vectors, and payloads must have the same length")
            
            points = []
            for i in range(len(vectors)):
                point_id = ids[i] if ids[i] else uuid.uuid4().hex
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vectors[i],
                        payload=payloads[i]
                    )
                )

            self.client.upsert(
                collection_name=self.collection,
                points=points
            )

        except Exception as e:
            logger.error(f"Upsert error: {str(e)}")
            raise VectorStoreUpsertError(f"Failed to upsert vectors: {str(e)}")

    async def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors (implements VectorStore interface)."""
        try:
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions) if len(conditions) > 1 else conditions[0]
            
            search_url = f"{self.url}/collections/{self.collection}/points/search"
            payload = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
                "score_threshold": score_threshold
            }
            if query_filter:
                filter_dict = {"must": []}
                if hasattr(query_filter, 'must'):
                    for condition in query_filter.must:
                        filter_dict["must"].append({
                            "key": condition.key,
                            "match": {"value": condition.match.value}
                        })
                else:
                    filter_dict["must"].append({
                        "key": query_filter.key,
                        "match": {"value": query_filter.match.value}
                    })
                payload["filter"] = filter_dict
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(search_url, json=payload)
                response.raise_for_status()
                data = response.json()
                points = data.get("result") or []
            
            if not isinstance(points, list):
                points = []
            
            
            results = []
            for r in points:
                if r is None:
                    continue
                score = r.get("score", 0.0)
                if score >= score_threshold:
                    results.append(
                        VectorSearchResult(
                            id=str(r.get("id", "")),
                            score=score,
                            payload=r.get("payload") or {}
                        )
                    )
            
            return results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise VectorStoreSearchError(f"Failed to search vectors: {str(e)}")

    async def delete(
        self,
        ids: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete vectors by ID or filters (implements VectorStore interface)."""
        try:
            if ids:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=ids
                )
            elif filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions) if len(conditions) > 1 else conditions[0]
                    result = self.client.delete(
                        collection_name=self.collection,
                        points_selector=query_filter
                    )
            else:
                raise ValueError("Either ids or filters must be provided")
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
            raise VectorStoreDeleteError(f"Failed to delete vectors: {str(e)}")
    
    async def clear(self) -> None:
        """Delete all vectors (reset collection)."""
        try:
            self.client.delete_collection(self.collection)
            self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=config.QDRANT_HNSW_M,
                    ef_construct=config.QDRANT_HNSW_EF_CONSTRUCT,
                ),
            )
            self._create_payload_indexes()
        except Exception as e:
            logger.error(f"Clear error: {str(e)}")
            raise VectorStoreError(f"Failed to clear collection: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            collection_info = self.client.get_collection(self.collection)
            return {
                "vector_count": collection_info.points_count,
                "dimension": collection_info.config.params.vectors.size,
                "collection_name": self.collection,
                "storage_size": getattr(collection_info, "vectors_count", 0) * self.vector_size * 4,  # Approximate
            }
        except Exception as e:
            logger.error(f"Get stats error: {str(e)}")
            raise VectorStoreError(f"Failed to get stats: {str(e)}")
    
    async def close(self) -> None:
        """Close connection to vector store."""
        pass

vector_store = QdrantVectorStore()

async def upsert(vectors, payloads, ids):
    """Upsert vectors to Qdrant (legacy wrapper - maintains backward compatibility)"""
    return await vector_store.upsert(ids, vectors, payloads)

async def search(vector, filters=None, limit=5):
    """Search Qdrant (legacy wrapper)"""
    results = await vector_store.search(vector, filters, limit)
    
    from rag.query.retriever import QueryResult
    return [
        QueryResult(
            text=(r.payload.get("text", "") or 
                  (r.payload.get("metadata", {}).get("text", "") if isinstance(r.payload.get("metadata"), dict) else "")),
            score=r.score,
            metadata=r.payload if isinstance(r.payload, dict) else {}
        )
        for r in results if r is not None
    ]

async def delete(doc_id):
    """Delete vectors"""
    return await vector_store.delete(doc_id)
