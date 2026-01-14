import uuid
import time
from typing import List, Dict, Any, Optional
from rag.ingest.loaders import load
from rag.ingest.semantic_chunker import chunk
from rag.ingest.semantic_chunker import SemanticChunker
from rag.vector_store.qdrant import upsert
from rag.config import config
from rag.cache import get_embedding_cache
from rag.core.container import get_container
from rag.core.exceptions import IngestError, ChunkingError, EmbeddingError
from rag.logging import logger


class AsyncIngestPipeline:
    """Async document ingestion pipeline with batching and caching.
    
    Benefits:
    - Uses DI container for shared embedding provider
    - Connection pooling via shared HTTP client
    - Batch embedding with caching
    - Semantic chunking with document structure awareness
    """
    
    def __init__(self):
        self.embedding_cache = get_embedding_cache()
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        self.semantic_chunker = SemanticChunker()
    
    def _get_embedding_provider(self):
        """Get embedding provider from DI container (shared instance)"""
        return get_container().get_embedding_provider()
    
    async def ingest_document(
        self,
        filepath: str,
        project: str,
        language: str = "en",
        doc_id: Optional[str] = None,
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Full pipeline with optimization:
        1. Load document
        2. Chunk text (semantic-aware)
        3. Embed chunks (with batching and caching)
        4. Upsert to vector DB
        
        Args:
            filepath: Path to document
            project: Project identifier
            language: Document language
            doc_id: Optional document ID
            async_mode: If True, enqueue as background task
        
        Returns:
            Dictionary with ingestion results
        """
        if async_mode:
            from rag.background_tasks import get_task_queue
            queue = get_task_queue()
            task_coro = self._ingest_document_impl(filepath, project, language, doc_id)
            await queue.enqueue(task_coro)
            
            return {
                "doc_id": doc_id or str(uuid.uuid4()),
                "status": "queued",
                "message": "Document queued for ingestion"
            }
        else:
            return await self._ingest_document_impl(filepath, project, language, doc_id)
    
    async def _ingest_document_impl(
        self,
        filepath: str,
        project: str,
        language: str = "en",
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal implementation of document ingestion"""
        try:
            if not doc_id:
                doc_id = str(uuid.uuid4())
            
            start_time = time.time()
            
            # Step 1: Load document
            documents = load(filepath)
            
            total_chunks = 0
            
            # Step 2-4: Process each document
            for doc in documents:
                chunk_start = time.time()
                if config.SEMANTIC_CHUNKING:
                    chunks_with_meta = self.semantic_chunker.chunk_with_metadata(
                        doc.text,
                        doc_type="markdown" if hasattr(doc, 'doc_type') else "plain_text"
                    )
                    chunks = [c[0] for c in chunks_with_meta]
                    chunk_metadata = {i: c[1] for i, c in enumerate(chunks_with_meta)}
                else:
                    chunks = chunk(doc.text)
                    chunk_metadata = {}
                
                if not chunks:
                    continue
                
                # Step 3: Batch embedding with caching
                embeddings = await self._embed_chunks_optimized(chunks)
                
                # Step 4: Prepare payloads
                texts = []
                payloads = []
                ids = []
                
                for i, chunk_text in enumerate(chunks):
                    texts.append(chunk_text)
                    
                    payload = {
                        "project": project,
                        "source": filepath.split('/')[-1],
                        "language": language,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk_text,
                        **doc.metadata
                    }
                    
                    if i in chunk_metadata:
                        payload["chunk_meta"] = chunk_metadata[i]
                    
                    payloads.append(payload)
                    ids.append(str(uuid.uuid4()))
                
                await upsert(embeddings, payloads, ids)
                
                total_chunks += len(chunks)
            
            total_time = time.time() - start_time
            
            return {
                "doc_id": doc_id,
                "chunks_created": total_chunks,
                "status": "completed",
                "duration_seconds": total_time
            }
        
        except Exception as e:
            logger.error(f"Ingestion error: {str(e)}")
            if isinstance(e, (ChunkingError, EmbeddingError)):
                raise
            raise IngestError(f"Document ingestion failed: {str(e)}")
    
    async def _embed_chunks_optimized(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed chunks with caching and batching.
        
        - Checks cache for already-embedded chunks
        - Batches remaining chunks for efficient embedding
        - Caches new embeddings
        """
        embeddings_result = [None] * len(chunks)
        
        # Step 1: Check cache for existing embeddings
        if config.CACHE_EMBEDDINGS:
            cached = self.embedding_cache.batch_get(chunks, config.EMBEDDING_MODEL)
            for idx, embedding in cached.items():
                embeddings_result[idx] = embedding
            
            uncached_indices = [i for i in range(len(chunks)) if embeddings_result[i] is None]
            uncached_chunks = [chunks[i] for i in uncached_indices]
        else:
            uncached_indices = list(range(len(chunks)))
            uncached_chunks = chunks
        
        if not uncached_chunks:
            return embeddings_result
        
        # Step 2: Embed uncached chunks in batches
        try:
            provider = self._get_embedding_provider()
            uncached_embeddings = await provider.embed_batch(
                uncached_chunks,
                instruction=config.EMBEDDING_PASSAGE_INSTRUCTION or None,
                batch_size=self.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to embed chunks: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Batch embedding failed: {str(e)}")
        
        # Step 3: Cache new embeddings and merge results
        for idx, embedding in zip(uncached_indices, uncached_embeddings):
            embeddings_result[idx] = embedding
            if config.CACHE_EMBEDDINGS:
                self.embedding_cache.set(chunks[idx], embedding, config.EMBEDDING_MODEL)
        
        return embeddings_result


pipeline = AsyncIngestPipeline()
