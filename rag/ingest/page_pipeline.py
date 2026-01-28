"""
Page-Aware Ingestion Pipeline (Strict 2-Phase Architecture)
============================================================

PHASE 1: BUILD STRUCTURE (No Embeddings)
- Load PDF page by page
- Extract text/OCR per page
- Normalize text per page
- Analyze structure (chapters, sections, titles)
- Build PageIndex (structural metadata only)
- Create page-aware chunks
- Output: page_index.json, chunks.json
- GUARDRAIL: NO EMBEDDING CALLS IN THIS PHASE

PHASE 2: EMBEDDING & INDEXING (After Structure is Final)
- Read chunks.json ONLY
- Generate embeddings from chunk text
- Build FAISS vector index
- Output: faiss.index, faiss_meta.json
- GUARDRAIL: Only source of text is chunks.json
"""

import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from rag.ingest.page_loader import load_pages, RawPageData
from rag.ingest.page_normalizer import PageNormalizer
from rag.ingest.page_index_builder import build_page_index
from rag.ingest.page_chunker import PageAwareChunker, PageChunk
from rag.core.page_index import PageIndex, PageMetadata
from rag.storage.file_storage import FileStorageManager
from rag.storage.faiss_index import create_faiss_index, save_faiss_index
from rag.cache import get_embedding_cache
from rag.config import config
from rag.core.container import get_container
from rag.core.exceptions import IngestError
from rag.logging import logger


class PageAwareIngestPipeline:
    """
    Ingestion pipeline:
    
    BUILD STRUCTURE
    - Load, normalize, analyze, chunk document
    - Save page_index.json (structure only, NO text)
    - Save chunks.json (verbatim text with page refs)
    - NO embedding calls
    
   EMBEDDING & INDEXING
    - Load chunks.json (only source of text)
    - Generate embeddings
    - Save faiss.index (vectors only)
    - Save faiss_meta.json (ID mappings only)
    """
    
    def __init__(self):
        self.normalizer = PageNormalizer()
        self.chunker = PageAwareChunker(
            max_chunk_size=config.CHUNK_MAX_SIZE,
            min_chunk_size=config.CHUNK_MIN_SIZE,
        )
        self.embedding_cache = get_embedding_cache()
        self.batch_size = config.EMBEDDING_BATCH_SIZE
    
    def _get_embedding_provider(self):
        return get_container().get_embedding_provider()
    
    async def build_structure(
        self,
        filepath: str,
        doc_id: Optional[str] = None,
        storage_dir: str = "./data/rag",
        language: str = "en",
    ) -> Tuple[str, int, int]:
        """
        Load document, build structure, save artifacts.
        
        GUARDRAIL: This phase MUST NOT generate any embeddings.
        
        Steps:
        1. Load PDF page by page
        2. Normalize text per page
        3. Build PageIndex (analyze structure)
        4. Create page-aware chunks
        5. Save page_index.json (structure only, NO text content)
        6. Save chunks.json (verbatim text with page references)
        
        Args:
            filepath: Path to PDF file
            doc_id: Document ID (auto-generate if None)
            storage_dir: Directory to save artifacts
            language: Document language
            
        Returns:
            (doc_id, page_count, chunk_count)
        """
        logger.info(f"PHASE 1: BUILD STRUCTURE - Starting for {filepath}")
        logger.info("GUARDRAIL: No embeddings will be generated in this phase")
        
        doc_id = doc_id or str(uuid.uuid4())
        filepath_obj = Path(filepath)
        source_filename = filepath_obj.name
        
        try:
            raw_pages = load_pages(str(filepath))
            
            if not raw_pages:
                raise IngestError("No content extracted from document")
            
            normalized_pages = []
            
            for raw_page in raw_pages:
                normalized = self.normalizer.normalize_page(
                    raw_page.page,
                    raw_page.raw_content
                )
                
                if normalized:
                    normalized["raw_content"] = raw_page.raw_content
                    normalized["has_ocr"] = raw_page.has_ocr
                    normalized["extraction_confidence"] = raw_page.extraction_confidence
                    normalized_pages.append(normalized)
            
            if not normalized_pages:
                raise IngestError("No valid content after normalization")
            
            page_index = build_page_index(doc_id, source_filename, normalized_pages)
            
            all_chunks = []
            
            for page_entry in page_index.get_all_pages():
                page_metadata = page_entry.to_page_metadata()
                page_metadata.doc_id = doc_id
                page_metadata.source_filename = source_filename
                page_metadata.language = language
                page_metadata.processing_timestamp = datetime.now().isoformat()
                
                chunks = self.chunker.chunk_page(
                    page_entry.page,
                    page_entry.clean_text,
                    page_metadata,
                )
                
                all_chunks.extend(chunks)
            
            storage = FileStorageManager(storage_dir)
            storage.save_page_index(page_index, doc_id=doc_id)
            storage.save_chunks(all_chunks, doc_id=doc_id)
            
            page_count = page_index.get_page_count()
            chunk_count = len(all_chunks)
            
            logger.info(f"  Artifacts: {doc_id}_page_index.json, {doc_id}_chunks.json")
            
            return doc_id, page_count, chunk_count
        
        except Exception as e:
            raise IngestError(f"Structure building failed: {e}")
    
    async def embed_and_index(
        self,
        doc_id: str,
        storage_dir: str = "./data/rag",
    ) -> Dict[str, Any]:
        """
        EMBEDDING & INDEXING - Generate embeddings from chunks.json.
        
        GUARDRAIL: Text source is ONLY chunks.json. Nothing else.
        
        Steps:
        1. Load chunks.json (only source of text)
        2. Generate embeddings batch by batch
        3. Create FAISS index (vectors only)
        4. Save faiss.index
        5. Save faiss_meta.json (ID mappings only)
        
        Args:
            doc_id: Document ID (must have Phase 1 artifacts)
            storage_dir: Directory containing Phase 1 artifacts
            
        Returns:
            Indexing result summary
        """
        try:
            storage = FileStorageManager(storage_dir)
            chunks_data = storage.load_chunks_for_embedding(doc_id)
            
            if not chunks_data:
                raise IngestError(f"No chunks found for doc_id={doc_id}")
            
            embedding_provider = self._get_embedding_provider()
            texts_to_embed = [chunk["text"] for chunk in chunks_data]
            
            embeddings = []
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch_texts = texts_to_embed[i:i+self.batch_size]
                batch_embeds = await embedding_provider.embed_batch(batch_texts)
                embeddings.extend(batch_embeds)
                
            faiss_index = create_faiss_index(embeddings)

            faiss_path = Path(storage_dir) / f"{doc_id}_faiss.index"
            save_faiss_index(faiss_index, str(faiss_path))
            
            from rag.ingest.page_chunker import PageChunk
            from rag.core.page_index import PageMetadata
            
            mock_chunks = []
            for i, chunk_dict in enumerate(chunks_data):
                metadata = PageMetadata(
                    page=chunk_dict["page"],
                    clean_text="",
                )
                mock_chunk = PageChunk(
                    chunk_id=chunk_dict["chunk_id"],
                    page=chunk_dict["page"],
                    text=chunk_dict["text"],
                    metadata=metadata,
                )
                mock_chunks.append(mock_chunk)
            
            storage.save_faiss_metadata(mock_chunks, doc_id=doc_id)
            
            result = {
                "doc_id": doc_id,
                "total_chunks": len(chunks_data),
                "vectors_indexed": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            }
            
            return result
        
        except Exception as e:
            raise IngestError(f"Embedding/indexing failed: {e}")
    
    async def ingest_document(
        self,
        filepath: str,
        language: str = "en",
        doc_id: Optional[str] = None,
        storage_dir: str = "./data/rag",
    ) -> Dict[str, Any]:
        """
        Complete ingestion pipeline.
        
        BUILD STRUCTURE (no embeddings)
        EMBEDDING & INDEXING (from chunks.json only)
        
        Args:
            filepath: Path to PDF file
            language: Document language
            doc_id: Document ID (auto-generate if None)
            storage_dir: Directory to save files
            
        Returns:
            Ingestion result
        """
        start_time = time.time()
        
        try:
            doc_id, page_count, chunk_count = await self.build_structure(
                filepath, doc_id, storage_dir, language
            )
            
            index_result = await self.embed_and_index(
                doc_id, storage_dir
            )
            
            elapsed = time.time() - start_time
            
            result = {
                "success": True,
                "doc_id": doc_id,
                "source_file": Path(filepath).name,
                "storage_dir": storage_dir,
                "total_pages": page_count,
                "total_chunks": chunk_count,
                "vectors_indexed": index_result["vectors_indexed"],
                "processing_time_seconds": round(elapsed, 2),
            }
            
            logger.info(f"✓ Ingestion complete in {elapsed:.1f}s: {result}")
            
            return result
        
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


# Convenience function
async def ingest_pdf(
    filepath: str,
    language: str = "en",
    doc_id: Optional[str] = None,
    storage_dir: str = "./data/rag",
) -> Dict[str, Any]:
    """
    Ingest PDF using strict 2-phase pipeline.
    
    Args:
        filepath: Path to PDF
        language: Document language
        doc_id: Document ID
        storage_dir: Storage directory
        
    Returns:
        Ingestion result
    """
    pipeline = PageAwareIngestPipeline()
    return await pipeline.ingest_document(
        filepath, language, doc_id, storage_dir
    )
