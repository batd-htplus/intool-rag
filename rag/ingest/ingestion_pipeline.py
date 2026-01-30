
from typing import List, Dict, Any, Optional
import time
from rag.logging import logger
from rag.config import config
from rag.ingest.page_loader import load_pages
from rag.ingest.page_normalizer import PageNormalizer
from rag.ingest.semantic.tree_builder import SemanticTreeBuilder
from rag.ingest.node_aware_chunker import ChunksBuilder
from rag.ingest.schemas import PageIndex, ChunksIndex
from rag.storage.file_storage import FileStorageManager
from rag.llm.embeddings.factory import get_embedding_provider
from rag.storage.faiss_index import create_faiss_index, save_faiss_index

class IngestionPipeline:
    """
    Orchestrates the complete document ingestion process:
    1. Load Pages
    2. Normalize Text
    3. Build Semantic Tree (PageIndex)
    4. Create Chunks (ChunksIndex)
    5. Generate Embeddings
    6. Build and Save FAISS Index
    """

    def __init__(self):
        self.normalizer = PageNormalizer()
        self.tree_builder = SemanticTreeBuilder()
        self.chunks_builder = ChunksBuilder()
        self.storage = FileStorageManager(config.STORAGE_DIR)

    async def ingest_file(
        self,
        filepath: str,
        project: str,
        doc_id: str,
        source_filename: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        logger.info(f"Starting ingestion for {source_filename} (doc_id={doc_id})")

        # --- PHASE 1: Load and Normalize ---
        raw_pages = load_pages(filepath)
        pages_data = []
        for p in raw_pages:
            normalized = self.normalizer.normalize_page(
                page=p.page,
                raw_text=p.raw_content
            )
            if normalized:
                pages_data.append(normalized)
        
        if not pages_data:
            raise ValueError("No valid pages extracted")

        logger.info(f"Normalized {len(pages_data)} pages")

        # --- PHASE 2: Semantic Tree ---
        page_index = await self.tree_builder.build(
            doc_id=doc_id,
            source_filename=source_filename,
            pages_data=pages_data,
            language=language
        )
        
        self.storage.save_page_index(page_index, doc_id=doc_id)

        # --- PHASE 3: Chunking ---
        chunks_index = await self.chunks_builder.build_chunks(
            doc_id=doc_id,
            page_index=page_index,
            pages_data=pages_data
        )
        
        self.storage.save_chunks(chunks_index.chunks, doc_id=doc_id)

        # --- PHASE 4: Embeddings & Indexing ---
        embedding_provider = get_embedding_provider()
        
        chunks = chunks_index.chunks
        texts = [c.text for c in chunks]
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = await embedding_provider.embed_batch(texts)
        
        index = create_faiss_index(embeddings)
        
        index_filename = f"{doc_id}_faiss.index"
        index_path = config.STORAGE_DIR / index_filename
        save_faiss_index(index, str(index_path))
        
        self.storage.save_faiss_metadata(chunks, doc_id=doc_id)
        
        total_time = time.time() - start_time
        logger.info(f"Ingestion complete in {total_time:.2f}s")

        return {
            "success": True,
            "doc_id": doc_id,
            "page_count": len(pages_data),
            "chunk_count": len(chunks),
            "vectors_indexed": len(embeddings),
            "processing_time": total_time
        }
