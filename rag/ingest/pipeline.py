import uuid
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from rag.ingest.loaders import load
from rag.ingest.semantic_chunker import SemanticChunker
from rag.ingest.normalizer import (
    normalize_text_for_storage,
    create_payload,
    validate_payload,
    _is_valid_text
)
from rag.vector_store.qdrant import vector_store
from rag.config import config
from rag.cache import get_embedding_cache
from rag.core.container import get_container
from rag.core.exceptions import IngestError, ChunkingError, EmbeddingError
from rag.logging import logger

class AsyncIngestPipeline:
    """
    Precision-first ingestion pipeline:
    - Clean text (JA / VI / EN)
    - Semantic chunking with hard limits
    - Batched embedding with cache
    - Minimal, stable metadata
    """

    def __init__(self):
        self.embedding_cache = get_embedding_cache()
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        self.progressive_batch = config.EMBEDDING_PROGRESSIVE_BATCH
        self.max_parallel = config.EMBEDDING_MAX_PARALLEL
        self.chunker = SemanticChunker()

    def _get_embedding_provider(self):
        return get_container().get_embedding_provider()

    async def ingest_document(
        self,
        filepath: str,
        project: str,
        language: str = "en",
        doc_id: Optional[str] = None,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Public API used by /ingest endpoint.
        - Keeps backward compatibility with async_mode parameter.
        - For now we always run synchronously to keep behavior simple & predictable.
        """
        return await self._ingest_impl(filepath, project, language, doc_id)

    async def _ingest_impl(
        self,
        filepath: str,
        project: str,
        language: str,
        doc_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Optimized ingestion implementation with data normalization.
        
        Features:
        - Text quality validation
        - Metadata standardization
        - Batch processing optimization
        - Payload validation
        """
        try:
            doc_id = doc_id or str(uuid.uuid4())
            start_time = time.time()

            # Load documents
            documents = load(filepath)
            if not documents:
                raise IngestError("No document content loaded")

            # Normalize source filename
            source_path = Path(filepath)
            source_filename = source_path.name

            total_chunks = 0
            chunk_counter = 0
            skipped_chunks = 0

            # Progressive processing: embed and upsert in small batches to avoid timeout
            progressive_payloads = []
            progressive_ids = []
            progressive_embeddings = []

            for doc in documents:
                raw_text = getattr(doc, "text", "")
                if not raw_text or not raw_text.strip():
                    continue

                # Normalize text early
                clean_text = normalize_text_for_storage(raw_text)
                if not clean_text:
                    logger.debug(f"Skipping document with invalid text quality")
                    continue

                # Chunk text
                try:
                    doc_metadata = getattr(doc, "metadata", {}) or {}
                    doc_type = "markdown" if doc_metadata.get("doc_type") == "markdown" else "plain_text"
                    
                    chunks_with_meta = self.chunker.chunk_with_metadata(
                        clean_text,
                        doc_type=doc_type
                    )
                except Exception as e:
                    raise ChunkingError(str(e))

                if not chunks_with_meta:
                    continue

                # Filter and validate chunks
                valid_chunks = []
                valid_chunk_meta = []

                for text, meta in chunks_with_meta:
                    text = text.strip()
                    if not text:
                        continue
                    
                    if len(text) > config.MAX_CHUNK_CHAR:
                        text = text[:config.MAX_CHUNK_CHAR]
                    
                    if not _is_valid_text(text, min_length=10):
                        skipped_chunks += 1
                        continue
                    
                    valid_chunks.append(text)
                    valid_chunk_meta.append(meta)

                if not valid_chunks:
                    continue

                # Progressive embedding: process in smaller batches to avoid timeout
                for batch_start in range(0, len(valid_chunks), self.progressive_batch):
                    batch_end = min(batch_start + self.progressive_batch, len(valid_chunks))
                    batch_chunks = valid_chunks[batch_start:batch_end]
                    batch_meta = valid_chunk_meta[batch_start:batch_end]

                    batch_embeddings = await self._embed_chunks(batch_chunks)

                    for i, text in enumerate(batch_chunks):
                        payload = create_payload(
                            text=text,
                            metadata=doc_metadata,
                            source_filename=source_filename,
                            project=project,
                            language=language,
                            doc_id=doc_id,
                            chunk_index=chunk_counter,
                            chunk_meta=batch_meta[i]
                        )
                        
                        if not payload:
                            skipped_chunks += 1
                            continue
                        
                        progressive_payloads.append(payload)
                        progressive_ids.append(str(uuid.uuid4()))
                        progressive_embeddings.append(batch_embeddings[i])
                        chunk_counter += 1

                    if len(progressive_payloads) >= self.progressive_batch:
                        await vector_store.upsert(progressive_ids, progressive_embeddings, progressive_payloads)
                        total_chunks += len(progressive_payloads)
                        logger.debug(f"Upserted {len(progressive_payloads)} chunks (total: {total_chunks})")
                        progressive_payloads = []
                        progressive_ids = []
                        progressive_embeddings = []

            if progressive_payloads:
                await vector_store.upsert(progressive_ids, progressive_embeddings, progressive_payloads)
                total_chunks += len(progressive_payloads)

            duration = round(time.time() - start_time, 3)
            
            logger.info(
                f"Ingestion completed: {total_chunks} chunks created, "
                f"{skipped_chunks} skipped, {duration}s"
            )

            return {
                "doc_id": doc_id,
                "chunks_created": total_chunks,
                "chunks_skipped": skipped_chunks,
                "status": "completed",
                "duration_seconds": duration,
            }

        except (ChunkingError, EmbeddingError):
            raise
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise IngestError(str(e))

    async def _embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Optimized embedding with cache and parallel batch processing."""
        if not chunks:
            return []
        
        results = [None] * len(chunks)
        instruction = config.EMBEDDING_PASSAGE_INSTRUCTION or ""

        # Batch cache lookup
        if config.CACHE_EMBEDDINGS:
            cached = self.embedding_cache.batch_get(
                chunks,
                config.EMBEDDING_MODEL,
                instruction=instruction
            )
            for idx, emb in cached.items():
                results[idx] = emb

        uncached_idx = [i for i, r in enumerate(results) if r is None]
        if not uncached_idx:
            return results

        uncached_chunks = [chunks[i] for i in uncached_idx]

        try:
            provider = self._get_embedding_provider()
            
            # Optimized parallel batch embedding with concurrency limit
            # Split into batches and process with controlled parallelism
            num_batches = (len(uncached_chunks) + self.batch_size - 1) // self.batch_size
            
            if num_batches > 1 and self.max_parallel > 1:
                # Multiple batches - process with controlled parallelism
                new_embeddings = []
                semaphore = asyncio.Semaphore(self.max_parallel)
                
                async def embed_with_semaphore(batch):
                    async with semaphore:
                        return await provider.embed_batch(
                            batch,
                            instruction=instruction or None,
                            batch_size=len(batch)
                        )
                
                # Create batches
                batches = [
                    uncached_chunks[i:i + self.batch_size]
                    for i in range(0, len(uncached_chunks), self.batch_size)
                ]
                
                # Execute with concurrency limit
                batch_results = await asyncio.gather(*[
                    embed_with_semaphore(batch) for batch in batches
                ])
                
                # Flatten results
                for batch_embeddings in batch_results:
                    new_embeddings.extend(batch_embeddings)
            else:
                # Single batch or parallel disabled - process sequentially
                new_embeddings = await provider.embed_batch(
                    uncached_chunks,
                    instruction=instruction or None,
                    batch_size=self.batch_size
                )
        except Exception as e:
            raise EmbeddingError(str(e))

        # Update results and cache
        for i, emb in zip(uncached_idx, new_embeddings):
            results[i] = emb
            if config.CACHE_EMBEDDINGS:
                self.embedding_cache.set(
                    chunks[i],
                    emb,
                    config.EMBEDDING_MODEL,
                    instruction=instruction
                )

        return results


pipeline = AsyncIngestPipeline()
