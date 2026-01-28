"""
Page-Aware Chunking
===================

Purpose:
- Split page text into semantic chunks
- NEVER break chunks across page boundaries
- Each chunk inherits page metadata
- Enable vector search + citation

Core principle:
- Chunk stays within 1 page
- Chunk ID encodes page number: "c_{page}_{chunk_index}"
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid
from rag.logging import logger
from rag.core.page_index import PageMetadata

@dataclass
class PageChunk:
    """Single chunk - atomic unit for embedding & retrieval"""
    chunk_id: str
    page: int
    text: str
    metadata: PageMetadata
    
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for embedding/storage"""
        return {
            "chunk_id": self.chunk_id,
            "page": self.page,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "metadata": asdict(self.metadata),
        }
    
    def to_embedding_payload(self) -> Dict[str, Any]:
        """Convert to payload for vector store"""
        return {
            "chunk_id": self.chunk_id,
            "page": self.page,
            "text": self.text,
            "chapter": self.metadata.chapter,
            "section": self.metadata.section,
            "subsection": self.metadata.subsection,
            "title": self.metadata.title,
            "doc_id": self.metadata.doc_id,
            "source_filename": self.metadata.source_filename,
        }


class PageAwareChunker:
    """
    Semantic chunking with page boundaries.
    
    Strategy:
    1. Split by paragraphs first
    2. Merge small paragraphs while respecting chunk size
    3. Each chunk stays in single page
    4. Preserve order
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 50,
        overlap: int = 0,
    ):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk (soft limit)
            min_chunk_size: Minimum characters per chunk
            overlap: Character overlap between chunks (not used for page-aware)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Merge paragraphs into chunks while respecting size limits.
        
        Strategy:
        - Start with first paragraph
        - Keep adding paragraphs until chunk would exceed max_chunk_size
        - If single paragraph > max_chunk_size, keep it as-is
        """
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if not current_chunk:
                current_chunk = para
            else:
                test_chunk = current_chunk + "\n\n" + para
                
                if len(test_chunk) <= self.max_chunk_size:
                    current_chunk = test_chunk
                else:
                    if len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = para
                    else:
                        if len(test_chunk) <= self.max_chunk_size * 1.2:
                            current_chunk = test_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = para
        
        # Don't forget last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        elif current_chunk and chunks:
            # Too small to be standalone, merge with previous
            chunks[-1] = chunks[-1] + "\n\n" + current_chunk
        elif current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_page(
        self,
        page: int,
        page_text: str,
        page_metadata: PageMetadata,
    ) -> List[PageChunk]:
        """
        Chunk single page.
        
        Args:
            page: Page number
            page_text: Clean text from page
            page_metadata: Page metadata
            
        Returns:
            List of PageChunk
        """
        if not page_text or not page_text.strip():
            return []
        
        paragraphs = self._split_by_paragraphs(page_text)
        if not paragraphs:
            return []
        
        chunk_texts = self._merge_paragraphs(paragraphs)
        if not chunk_texts:
            return []
        
        chunks = []
        char_pos = 0
        
        for chunk_idx, chunk_text in enumerate(chunk_texts):
            chunk_id = f"c_{page}_{chunk_idx:03d}"
            
            chunk = PageChunk(
                chunk_id=chunk_id,
                page=page,
                text=chunk_text,
                metadata=page_metadata,
                chunk_index=chunk_idx,
                char_start=char_pos,
                char_end=char_pos + len(chunk_text),
            )
            
            chunks.append(chunk)
            char_pos += len(chunk_text) + 2
        
        logger.debug(
            f"Page {page}: Created {len(chunks)} chunks "
            f"(text len: {len(page_text)}, avg chunk: {len(page_text)//len(chunks) if chunks else 0})"
        )
        
        return chunks
    
    def chunk_pages(
        self,
        pages_with_metadata: List[Dict[str, Any]],
    ) -> List[PageChunk]:
        """
        Chunk multiple pages.
        
        Expected input format:
        [
            {
                "page": 1,
                "text": "...",
                "metadata": PageMetadata(...)
            }
        ]
        
        Args:
            pages_with_metadata: List of page dicts with text and metadata
            
        Returns:
            List of all chunks from all pages
        """
        all_chunks = []
        
        for page_data in pages_with_metadata:
            page = page_data["page"]
            text = page_data["text"]
            metadata = page_data["metadata"]
            
            chunks = self.chunk_page(page, text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages_with_metadata)} pages")
        
        return all_chunks


# Convenience functions
_chunker = None

def get_page_chunker(
    max_chunk_size: int = 512,
    min_chunk_size: int = 50,
) -> PageAwareChunker:
    """Get singleton chunker"""
    global _chunker
    if _chunker is None or \
       _chunker.max_chunk_size != max_chunk_size or \
       _chunker.min_chunk_size != min_chunk_size:
        _chunker = PageAwareChunker(max_chunk_size, min_chunk_size)
    return _chunker


def chunk_page(
    page: int,
    text: str,
    metadata: PageMetadata,
) -> List[PageChunk]:
    """Chunk single page"""
    chunker = get_page_chunker()
    return chunker.chunk_page(page, text, metadata)


def chunk_pages(pages_with_metadata: List[Dict[str, Any]]) -> List[PageChunk]:
    """Chunk multiple pages"""
    chunker = get_page_chunker()
    return chunker.chunk_pages(pages_with_metadata)
