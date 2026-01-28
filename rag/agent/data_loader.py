"""
Data Loader: Load page_index.json, chunks.json, faiss_meta.json

Read-only access to 3 core files created during BUILD STRUCTURE phase.
Agent does NOT write files.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from rag.logging import logger


@dataclass
class PageIndexEntry:
    """Page structure info"""
    page_id: int
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class ChunkEntry:
    """Chunk metadata and content"""
    chunk_id: str
    page_id: int
    text: str
    offset: List[int]
    section: Optional[str] = None
    title: Optional[str] = None


@dataclass
class FAISSMeta:
    """FAISS vector metadata"""
    chunk_id: str
    page_id: int
    embedding_id: int


class AgentStorage:
    """Load and cache 3 core data files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._page_index: Optional[Dict[int, PageIndexEntry]] = None
        self._chunks: Optional[Dict[str, ChunkEntry]] = None
        self._faiss_meta: Optional[Dict[str, FAISSMeta]] = None
    
    def _validate_files(self) -> None:
        """Verify all required files exist"""
        missing = []
        for name in ["page_index.json", "chunks.json", "faiss_meta.json"]:
            if not (self.data_dir / name).exists():
                missing.append(name)
        
        if missing:
            raise FileNotFoundError(f"Missing: {', '.join(missing)}")
    
    def load_page_index(self) -> Dict[int, PageIndexEntry]:
        """Load page structure"""
        if self._page_index is not None:
            return self._page_index
        
        logger.info("Loading page_index.json")
        with open(self.data_dir / "page_index.json", "r") as f:
            data = json.load(f)
        
        self._page_index = {}
        for page_id_str, entry_data in data.items():
            self._page_index[int(page_id_str)] = PageIndexEntry(
                page_id=int(page_id_str),
                chapter=entry_data.get("chapter"),
                section=entry_data.get("section"),
                subsection=entry_data.get("subsection"),
                title=entry_data.get("title"),
                summary=entry_data.get("summary"),
            )
        
        logger.info(f"Loaded {len(self._page_index)} pages")
        return self._page_index
    
    def load_chunks(self) -> Dict[str, ChunkEntry]:
        """Load chunks"""
        if self._chunks is not None:
            return self._chunks
        
        logger.info("Loading chunks.json")
        with open(self.data_dir / "chunks.json", "r") as f:
            data = json.load(f)
        
        self._chunks = {
            chunk_id: ChunkEntry(
                chunk_id=chunk_id,
                page_id=chunk_data["page_id"],
                text=chunk_data["text"],
                offset=chunk_data["offset"],
                section=chunk_data.get("section"),
                title=chunk_data.get("title"),
            )
            for chunk_id, chunk_data in data.items()
        }
        
        logger.info(f"Loaded {len(self._chunks)} chunks")
        return self._chunks
    
    def load_faiss_meta(self) -> Dict[str, FAISSMeta]:
        """Load FAISS metadata"""
        if self._faiss_meta is not None:
            return self._faiss_meta
        
        logger.info("Loading faiss_meta.json")
        with open(self.data_dir / "faiss_meta.json", "r") as f:
            data = json.load(f)
        
        self._faiss_meta = {
            chunk_id: FAISSMeta(
                chunk_id=chunk_id,
                page_id=meta_data["page_id"],
                embedding_id=meta_data["embedding_id"],
            )
            for chunk_id, meta_data in data.items()
        }
        
        logger.info(f"Loaded {len(self._faiss_meta)} metadata entries")
        return self._faiss_meta
    
    def get_page_info(self, page_id: int) -> Optional[PageIndexEntry]:
        """Get page info"""
        return self.load_page_index().get(page_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkEntry]:
        """Get chunk content"""
        return self.load_chunks().get(chunk_id)
    
    def get_chunks_for_page(self, page_id: int) -> List[ChunkEntry]:
        """Get all chunks for page"""
        return [c for c in self.load_chunks().values() if c.page_id == page_id]
    
    def get_faiss_id_for_chunk(self, chunk_id: str) -> Optional[int]:
        """Get FAISS embedding ID"""
        entry = self.load_faiss_meta().get(chunk_id)
        return entry.embedding_id if entry else None
    
    def verify(self) -> bool:
        """Verify all files exist and are valid"""
        try:
            self._validate_files()
            self.load_page_index()
            self.load_chunks()
            self.load_faiss_meta()
            logger.info("✓ Storage verified")
            return True
        except Exception as e:
            logger.error(f"✗ Storage error: {e}")
            return False
