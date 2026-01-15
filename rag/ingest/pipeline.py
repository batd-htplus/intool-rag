import uuid
import time
from typing import List, Dict, Any, Optional

from rag.ingest.loaders import load
from rag.ingest.semantic_chunker import SemanticChunker
from rag.vector_store.qdrant import vector_store
from rag.config import config
from rag.cache import get_embedding_cache
from rag.core.container import get_container
from rag.core.exceptions import IngestError, ChunkingError, EmbeddingError
from rag.logging import logger
from rag.query.prompt import clean_text_multilang


# =====================================================
# Async Ingest Pipeline (Precision-first)
# =====================================================

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
        self.chunker = SemanticChunker()

    def _get_embedding_provider(self):
        return get_container().get_embedding_provider()

    # =================================================
    # Public API
    # =================================================

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

    # =================================================
    # Core Implementation
    # =================================================

    async def _ingest_impl(
        self,
        filepath: str,
        project: str,
        language: str,
        doc_id: Optional[str]
    ) -> Dict[str, Any]:
        try:
            doc_id = doc_id or str(uuid.uuid4())
            start_time = time.time()

            documents = load(filepath)
            if not documents:
                raise IngestError("No document content loaded")

            total_chunks = 0
            chunk_counter = 0

            for doc in documents:
                raw_text = getattr(doc, "text", "")
                if not raw_text.strip():
                    continue

                clean_text = clean_text_multilang(raw_text)
                if not clean_text:
                    continue

                try:
                    chunks_with_meta = self.chunker.chunk_with_metadata(
                        clean_text,
                        doc_type="markdown" if getattr(doc, "doc_type", "") == "markdown" else "plain_text"
                    )
                except Exception as e:
                    raise ChunkingError(str(e))

                if not chunks_with_meta:
                    continue

                chunks = []
                chunk_meta = []

                for text, meta in chunks_with_meta:
                    text = text.strip()
                    if not text:
                        continue
                    if len(text) > config.MAX_CHUNK_CHAR:
                        text = text[:config.MAX_CHUNK_CHAR]
                    chunks.append(text)
                    chunk_meta.append(meta)

                if not chunks:
                    continue

                embeddings = await self._embed_chunks(chunks)

                payloads = []
                ids = []

                for i, text in enumerate(chunks):
                    payloads.append({
                        "text": text,
                        "project": project,
                        "doc_id": doc_id,
                        "source": filepath.split("/")[-1],
                        "language": language,
                        "chunk_index": chunk_counter,
                        "chunk_meta": chunk_meta[i],
                    })
                    ids.append(str(uuid.uuid4()))
                    chunk_counter += 1

                await vector_store.upsert(ids, embeddings, payloads)
                total_chunks += len(chunks)

            return {
                "doc_id": doc_id,
                "chunks_created": total_chunks,
                "status": "completed",
                "duration_seconds": round(time.time() - start_time, 3),
            }

        except (ChunkingError, EmbeddingError):
            raise
        except Exception as e:
            raise IngestError(str(e))

    # =================================================
    # Embedding with Cache
    # =================================================

    async def _embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        results = [None] * len(chunks)
        instruction = config.EMBEDDING_PASSAGE_INSTRUCTION or ""

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
            new_embeddings = await provider.embed_batch(
                uncached_chunks,
                instruction=instruction or None,
                batch_size=self.batch_size
            )
        except Exception as e:
            raise EmbeddingError(str(e))

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
