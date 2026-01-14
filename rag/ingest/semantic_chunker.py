"""
Advanced semantic-aware chunking strategies for different document types.
"""

from typing import List, Tuple, Optional
import re
from enum import Enum
from rag.config import config
from rag.logging import logger


class DocumentType(Enum):
    """Supported document types"""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"


class ChunkMetadata:
    """Metadata for a chunk"""
    def __init__(self, text: str, section: str = "", level: int = 0, source_type: str = ""):
        self.text = text
        self.section = section
        self.level = level  # Hierarchy level (h1, h2, h3, etc.)
        self.source_type = source_type


class SemanticChunker:
    """
    Advanced chunking strategy aware of document structure.
    Chunks based on semantic boundaries (sections, paragraphs) 
    rather than just character/token limits.
    """
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
        self.max_chunk_size = int(self.chunk_size * 1.5)  # Allow up to 150% of chunk_size
        
    def _is_cjk(self, text: str) -> bool:
        """Detect if text contains CJK characters"""
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff':
                return True
        return False
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 chars for English, 1 char per token for CJK)
        BGE-M3 optimal chunk size: ~512-768 tokens
        """
        if self._is_cjk(text):
            return len(text)
        return len(text) // 4
    
    def chunk_markdown(self, text: str) -> List[Tuple[str, dict]]:
        """
        Smart chunking for Markdown documents.
        Respects heading hierarchy and sections.
        Returns: List of (text, metadata_dict) tuples
        """
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_section = ""
        current_level = 0
        current_tokens = 0
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                
                if current_chunk and current_tokens > 50:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append((chunk_text, {
                            "section": current_section,
                            "level": current_level,
                            "doc_type": "markdown"
                        }))
                
                current_section = title
                current_level = level
                current_chunk = [line]
                current_tokens = self._estimate_tokens(line)
            
            else:
                line_tokens = self._estimate_tokens(line)
                
                if current_tokens + line_tokens > self.max_chunk_size and current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append((chunk_text, {
                            "section": current_section,
                            "level": current_level,
                            "doc_type": "markdown"
                        }))
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and len(current_chunk) > 1:
                        overlap_lines = current_chunk[-max(1, self.overlap // 50):]
                        current_chunk = overlap_lines + [line]
                    else:
                        current_chunk = [line]
                    
                    current_tokens = sum(self._estimate_tokens(l) for l in current_chunk)
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append((chunk_text, {
                    "section": current_section,
                    "level": current_level,
                    "doc_type": "markdown"
                }))
        
        return chunks
    
    def chunk_plain_text(self, text: str) -> List[Tuple[str, dict]]:
        """
        Smart chunking for plain text.
        Respects paragraph boundaries when possible.
        """
        chunks = []
        
        if self._is_cjk(text):
            # Character-based chunking for CJK
            i = 0
            while i < len(text):
                end = min(i + self.chunk_size, len(text))
                chunk_text = text[i:end]
                chunks.append((chunk_text, {"doc_type": "plain_text"}))
                i = end - self.overlap if end - self.overlap > i else end
        else:
            paragraphs = re.split(r'\n\s*\n+', text)
            
            current_chunk = []
            current_tokens = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_tokens = self._estimate_tokens(para)
                
                if para_tokens > self.max_chunk_size:
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunks.append((chunk_text, {"doc_type": "plain_text"}))
                        current_chunk = []
                        current_tokens = 0
                    
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sub_chunk = []
                    sub_tokens = 0
                    
                    for sent in sentences:
                        sent_tokens = self._estimate_tokens(sent)
                        
                        if sub_tokens + sent_tokens > self.max_chunk_size and sub_chunk:
                            chunk_text = ' '.join(sub_chunk)
                            chunks.append((chunk_text, {"doc_type": "plain_text"}))
                            sub_chunk = [sent]
                            sub_tokens = sent_tokens
                        else:
                            sub_chunk.append(sent)
                            sub_tokens += sent_tokens
                    
                    if sub_chunk:
                        chunk_text = ' '.join(sub_chunk)
                        chunks.append((chunk_text, {"doc_type": "plain_text"}))
                
                elif current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append((chunk_text, {"doc_type": "plain_text"}))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
            
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, {"doc_type": "plain_text"}))
        
        return chunks
    
    def chunk(self, text: str, doc_type: str = "plain_text") -> List[str]:
        """
        Main chunking method. Returns just the text chunks.
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Detect Markdown if not specified
        if doc_type == "plain_text" and re.search(r'^#+\s+', text, re.MULTILINE):
            doc_type = "markdown"
        
        if doc_type == "markdown":
            chunks_with_metadata = self.chunk_markdown(text)
        else:
            chunks_with_metadata = self.chunk_plain_text(text)
        
        return [chunk[0] for chunk in chunks_with_metadata]
    
    def chunk_with_metadata(self, text: str, doc_type: str = "plain_text") -> List[Tuple[str, dict]]:
        """
        Chunking that preserves metadata about each chunk.
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        if doc_type == "markdown":
            return self.chunk_markdown(text)
        else:
            return self.chunk_plain_text(text)

def chunk(text: str, doc_type: str = "plain_text") -> List[str]:
    """Legacy chunking function"""
    if config.SEMANTIC_CHUNKING:
        chunker = SemanticChunker()
        return chunker.chunk(text, doc_type)
    else:
        chunker = SemanticChunker()
        return chunker.chunk(text, doc_type)
