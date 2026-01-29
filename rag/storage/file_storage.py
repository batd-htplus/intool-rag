"""
File-Based Storage for RAG Agent
================================

Agent reads 3 files:
1. page_index.json - Document structure (chapters, sections, titles)
2. chunks.json - Chunk content and metadata
3. faiss.index - Vector index (+ faiss_meta.json for mapping)

No database, pure file I/O. Portable and debuggable.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from rag.logging import logger
from rag.ingest.schemas import Chunk, PageIndex
class FileStorageManager:
    """Manage file-based storage for RAG data"""
    
    PAGE_INDEX_FILE = "page_index.json"
    CHUNKS_FILE = "chunks.json"
    FAISS_INDEX_FILE = "faiss.index"
    FAISS_META_FILE = "faiss_meta.json"
    
    def __init__(self, data_dir: str):
        """
        Initialize storage manager.
        
        Args:
            data_dir: Directory containing page_index.json, chunks.json, faiss.index
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_page_index(self, page_index: PageIndex, doc_id: Optional[str] = None) -> str:
        """
        Save PageIndex to JSON file.
        
        Filename: {doc_id or timestamp}.page_index.json
        
        Args:
            page_index: PageIndex object
            doc_id: Optional document ID for filename
            
        Returns:
            Path to saved file
        """
        filename = (doc_id or "index") + "_" + self.PAGE_INDEX_FILE
        filepath = self.data_dir / filename
        
        data = page_index.to_dict()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved PageIndex to {filepath}")
        
        return str(filepath)
    
    def load_page_index(self, doc_id: str) -> PageIndex:
        """
        Load PageIndex from file.
        
        Args:
            doc_id: Document ID
            
        Returns:
            PageIndex object
        """
        filepath = self.data_dir / (doc_id + "_" + self.PAGE_INDEX_FILE)
        
        if not filepath.exists():
            raise FileNotFoundError(f"PageIndex not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        page_index = PageIndex.from_dict(data)
        
        logger.info(f"Loaded PageIndex: {page_index.get_page_count()} pages")
        
        return page_index
    
    def save_chunks(
        self,
        chunks: List[Chunk],
        doc_id: Optional[str] = None
    ) -> str:
        """
        Save chunks to JSON file.
        
        Format:
        {
          "chunks": [
            {
              "chunk_id": "c_12_00",
              "page": 12,
              "text": "...",
              "metadata": {...}
            }
          ]
        }
        
        Args:
            chunks: List of Chunk objects
            doc_id: Optional document ID for filename
            
        Returns:
            Path to saved file
        """
        filename = (doc_id or "index") + "_" + self.CHUNKS_FILE
        filepath = self.data_dir / filename
        
        # Serialize
        chunks_data = [
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "metadata": asdict(chunk.metadata),
            }
            for chunk in chunks
        ]
        
        data = {
            "total": len(chunks),
            "chunks": chunks_data,
        }
        
        # Save
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {filepath}")
        
        return str(filepath)
    
    def load_chunks(self, doc_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Load chunks from file.
        
        Returns dict keyed by chunk_id for fast lookup.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict of {chunk_id: chunk_data}
        """
        filepath = self.data_dir / (doc_id + "_" + self.CHUNKS_FILE)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Chunks not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Index by chunk_id for fast lookup
        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in data.get("chunks", [])
        }
        
        logger.info(f"Loaded {len(chunks_by_id)} chunks")
        
        return chunks_by_id
    
    def load_chunks_for_embedding(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Load chunks in order for Phase 2 embedding.
        
        Returns chunks in order matching intended faiss_id (index position).
        Used by Phase 2 to generate embeddings from chunks.json only.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk dicts in order (index = faiss_id)
        """
        filepath = self.data_dir / (doc_id + "_" + self.CHUNKS_FILE)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Chunks not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        logger.info(f"Loaded {len(chunks)} chunks for embedding")
        
        return chunks
    
    def save_faiss_metadata(
        self,
        chunks: List[Chunk],
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Save FAISS metadata mapping (chunk_id → faiss_id).
        
        Format:
        {
          "mapping": {
            "c_12_00": 0,
            "c_12_01": 1,
            ...
          },
          "reverse_mapping": {
            "0": "c_12_00",
            ...
          }
        }
        
        Args:
            chunks: List of Chunk objects
            doc_id: Optional document ID
            
        Returns:
            Path to saved file
        """
        filename = (doc_id or "index") + "_" + self.FAISS_META_FILE
        filepath = self.data_dir / filename
        
        mapping = {
            chunk.chunk_id: i
            for i, chunk in enumerate(chunks)
        }
        
        reverse_mapping = {
            str(i): chunk.chunk_id
            for i, chunk in enumerate(chunks)
        }
        
        page_mapping = {
            str(i): chunk.page
            for i, chunk in enumerate(chunks)
        }
        
        data = {
            "total": len(chunks),
            "mapping": mapping,
            "reverse_mapping": reverse_mapping,
            "page_mapping": page_mapping,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved FAISS metadata for {len(chunks)} chunks")
        
        return str(filepath)
    
    def load_faiss_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Load FAISS metadata.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Metadata dict with mapping and reverse_mapping
        """
        filepath = self.data_dir / (doc_id + "_" + self.FAISS_META_FILE)
        
        if not filepath.exists():
            raise FileNotFoundError(f"FAISS metadata not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document data exists"""
        return (self.data_dir / (doc_id + "_" + self.PAGE_INDEX_FILE)).exists()
    
    def list_documents(self) -> List[str]:
        """List all document IDs in storage"""
        doc_ids = set()
        
        for file in self.data_dir.glob("*_page_index.json"):
            doc_id = file.stem.replace("_page_index", "")
            doc_ids.add(doc_id)
        
        return sorted(list(doc_ids))
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get info about a document"""
        try:
            page_index = self.load_page_index(doc_id)
            chunks = self.load_chunks(doc_id)
            
            return {
                "doc_id": doc_id,
                "page_count": page_index.get_page_count(),
                "chunk_count": len(chunks),
                "source_filename": page_index.source_filename,
            }
        except Exception as e:
            logger.warning(f"Failed to get document info: {e}")
            return None
    
    async def get_unique_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in storage.
        
        Scans for all *_page_index.json files and returns document info.
        
        Returns:
            List of document info dicts
        """
        documents = []
        
        try:
            for index_file in self.data_dir.glob("*_page_index.json"):
                doc_id = index_file.stem.replace("_page_index", "")
                
                info = self.get_document_info(doc_id)
                if info:
                    documents.append(info)
            
            logger.info(f"Found {len(documents)} documents in storage")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
