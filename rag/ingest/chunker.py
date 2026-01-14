"""
Legacy chunker module - wrapper for semantic_chunker for backward compatibility.
All actual chunking logic has been moved to semantic_chunker.py
"""

from typing import List
from rag.ingest.semantic_chunker import SemanticChunker

_chunker = None

def get_chunker() -> SemanticChunker:
    """Get default semantic chunker instance"""
    global _chunker
    if _chunker is None:
        _chunker = SemanticChunker()
    return _chunker

def chunk(text: str) -> List[str]:
    """Chunk text - wrapper for backward compatibility"""
    chunker = get_chunker()
    return chunker.chunk(text)
