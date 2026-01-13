from typing import List, Dict, Any
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
from rag.config import config
from rag.logging import logger

class QdrantVectorStore:
    """Qdrant vector store interface"""
    
    def __init__(self):
        self.url = config.QDRANT_URL
        self.collection = config.QDRANT_COLLECTION
        self.vector_size = config.VECTOR_DIMENSION
        
        self.client = QdrantClient(url=self.url)
        
        try:
            self.client.get_collection(self.collection)
            logger.info(f"Connected to Qdrant collection: {self.collection}")
        except:
            logger.info(f"Creating new Qdrant collection: {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    async def upsert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Insert or update vectors"""
        try:
            points = []
            for i in range(len(vectors)):
                provided_id = None
                if ids and i < len(ids) and isinstance(ids[i], str) and ids[i]:
                    provided_id = ids[i]
                point_id = provided_id or uuid.uuid4().hex
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

            logger.info(f"Upserted {len(points)} vectors")
        except Exception as e:
            logger.error(f"Upsert error: {str(e)}")
            raise

    async def search(
        self,
        vector: List[float],
        filters: Dict[str, Any] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
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
            
            from qdrant_client.models import NearestQuery
            
            query = NearestQuery(nearest=vector)
            
            results = self.client.query_points(
                collection_name=self.collection,
                query=query,
                query_filter=query_filter if query_filter else None,
                limit=limit
            )
            
            logger.info(f"Found {len(results.points)} search results")
            
            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload
                }
                for r in results.points
            ]
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def delete(self, doc_id: str) -> int:
        """Delete vectors by document ID"""
        try:
            result = self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted {result.deleted} vectors for doc {doc_id}")
            return result.deleted
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
            raise

vector_store = QdrantVectorStore()

async def upsert(vectors, payloads, ids):
    """Upsert vectors to Qdrant"""
    return await vector_store.upsert(vectors, payloads, ids)

async def search(vector, filters=None, limit=5):
    """Search Qdrant"""
    results = await vector_store.search(vector, filters, limit)
    
    from rag.query.retriever import QueryResult
    return [
        QueryResult(
            text=r["payload"].get("text", ""),
            score=r["score"],
            metadata=r["payload"]
        )
        for r in results
    ]

async def delete(doc_id):
    """Delete vectors"""
    return await vector_store.delete(doc_id)
