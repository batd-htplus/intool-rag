from typing import List, Dict, Any, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    HnswConfigDiff,
    SearchParams,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from rag.config import config
from rag.logging import logger
from rag.vector_store.base import (
    VectorStore,
    VectorSearchResult,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreSearchError,
    VectorStoreUpsertError,
    VectorStoreDeleteError,
)

# Qdrant Vector Store
class QdrantVectorStore(VectorStore):
    """Production-grade Qdrant vector store for RAG."""

    def __init__(self):
        self.url = config.QDRANT_URL
        self.collection = config.QDRANT_COLLECTION
        self.vector_size = config.VECTOR_DIMENSION
        self.api_key = config.QDRANT_API_KEY

        try:
            client_kwargs = {"url": self.url}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            
            self.client = QdrantClient(**client_kwargs)
        except Exception as e:
            raise VectorStoreConnectionError(
                f"Cannot connect to Qdrant at {self.url}: {e}"
            )

        self._ensure_collection()

    # Collection setup
    def _ensure_collection(self) -> None:
        """
        Ensure collection exists, create if it doesn't.
        Handles race conditions where collection might be created by another instance.
        """
        try:
            self.client.get_collection(self.collection)
            logger.info(f"[Qdrant] Using existing collection: {self.collection}")
            self._create_payload_indexes() 
            return
        except Exception as e:
            logger.debug(f"[Qdrant] Collection not found, will create: {e}")

        # Try to create collection
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
            logger.info(f"[Qdrant] Created collection: {self.collection}")
        except UnexpectedResponse as e:
            if e.status_code == 409:
                logger.info(f"[Qdrant] Collection already exists (race condition): {self.collection}")
                self._create_payload_indexes()  # Ensure indexes exist
                return
            logger.error(f"[Qdrant] Failed to create collection: {e}")
            raise VectorStoreConnectionError(f"Failed to create collection: {e}")
        except Exception as e:
            logger.error(f"[Qdrant] Failed to create collection: {e}")
            raise VectorStoreConnectionError(f"Failed to create collection: {e}")

    def _create_payload_indexes(self) -> None:
        for field in ("project", "doc_id", "language", "source", "doc_type"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass

    # Helpers
    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=conditions) if conditions else None

    # Upsert
    async def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """
        Optimized batch upsert with validation.
        
        Features:
        - Input validation
        - Batch processing optimization
        - Error handling with detailed logging
        """
        if not ids or not vectors or not payloads:
            raise VectorStoreUpsertError("Empty input lists")
        
        if not (len(ids) == len(vectors) == len(payloads)):
            raise VectorStoreUpsertError(
                f"Length mismatch: ids={len(ids)}, vectors={len(vectors)}, payloads={len(payloads)}"
            )

        # Validate vector dimensions
        expected_dim = self.vector_size
        for i, vector in enumerate(vectors):
            if not isinstance(vector, list):
                raise VectorStoreUpsertError(f"Invalid vector type at index {i}")
            if len(vector) != expected_dim:
                raise VectorStoreUpsertError(
                    f"Vector dimension mismatch at index {i}: expected {expected_dim}, got {len(vector)}"
                )

        try:
            # Build points efficiently
            points = []
            for i in range(len(vectors)):
                point_id = ids[i] if ids[i] else uuid.uuid4().hex
                
                payload = payloads[i]
                if not isinstance(payload, dict):
                    logger.warning(f"Invalid payload type at index {i}, skipping")
                    continue
                
                if "text" not in payload:
                    logger.warning(f"Missing 'text' field in payload at index {i}, skipping")
                    continue
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vectors[i],
                        payload=payload,
                    )
                )

            if not points:
                logger.warning("[Qdrant] No valid points to upsert")
                return

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
            
            logger.debug(f"[Qdrant] Upserted {len(points)} points successfully")
            
        except Exception as e:
            logger.error(f"[Qdrant] Upsert failed: {e}", exc_info=True)
            raise VectorStoreUpsertError(str(e))

    # Search
    async def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[VectorSearchResult]:
        try:
            query_filter = self._build_filter(filters)

            # Use optimized HNSW search parameters
            search_params = None
            if config.QDRANT_HNSW_EF_SEARCH:
                search_params = SearchParams(hnsw_ef=config.QDRANT_HNSW_EF_SEARCH)
            
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
                search_params=search_params,
            )

            return [
                VectorSearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    payload=hit.payload or {},
                )
                for hit in hits
            ]

        except Exception as e:
            logger.error(f"[Qdrant] Search failed: {e}")
            raise VectorStoreSearchError(str(e))

    # Delete
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if ids:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=ids,
                )
                return

            if filters:
                query_filter = self._build_filter(filters)
                if not query_filter:
                    raise ValueError("Invalid delete filter")

                self.client.delete(
                    collection_name=self.collection,
                    points_selector=query_filter,
                )
                return

            raise ValueError("Either ids or filters must be provided")

        except Exception as e:
            logger.error(f"[Qdrant] Delete failed: {e}")
            raise VectorStoreDeleteError(str(e))

    async def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection)
            self._ensure_collection()
        except Exception as e:
            raise VectorStoreError(f"Clear failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection)
            return {
                "collection": self.collection,
                "points": info.points_count,
                "dimension": info.config.params.vectors.size,
            }
        except Exception as e:
            raise VectorStoreError(str(e))

    async def close(self) -> None:
        pass

vector_store = QdrantVectorStore()
