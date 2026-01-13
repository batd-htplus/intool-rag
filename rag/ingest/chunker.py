from typing import List
import re
from rag.config import config
from rag.logging import logger

class TextChunker:
    """Chunk text into overlapping segments"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not text or not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        
        # Heuristic: detect CJK characters. If present, use character-based chunking
        def has_cjk(s: str) -> bool:
            for ch in s:
                if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\u31f0' <= ch <= '\u31ff':
                    return True
            return False

        if has_cjk(text):
            i = 0
            L = len(text)
            while i < L:
                end = min(i + self.chunk_size, L)
                chunk_text = text[i:end]
                chunks.append(chunk_text)
                i = end - self.overlap if end - self.overlap > i else end
        else:
            # Latin-script: sentence-aware splitting using punctuation
            sentences = re.split(r'(?<=[\.!?])\s+', text.replace('\n', ' '))
            current = ''
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if not current:
                    current = sent
                else:
                    candidate = current + ' ' + sent
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        chunks.append(current)
                        current = sent
            if current:
                chunks.append(current)

        if self.overlap and self.overlap > 0 and len(chunks) > 1:
            merged = []
            for i, c in enumerate(chunks):
                if i == 0:
                    merged.append(c)
                else:
                    prev = merged[-1]
                    overlap_text = c[:self.overlap]
                    if overlap_text:
                        merged[-1] = prev + ' ' + overlap_text
                    merged.append(c)
            chunks = merged

        logger.info(f"Created {len(chunks)} chunks from text (size={self.chunk_size}, overlap={self.overlap})")
        return chunks

def chunk(text: str) -> List[str]:
    """Default chunking function"""
    chunker = TextChunker()
    return chunker.chunk(text)
