"""
Production-grade semantic chunker for RAG.
Optimized for embedding quality, low noise, and stability.
"""

from typing import List, Tuple, Dict
import re
from enum import Enum
from rag.config import config
from rag.logging import logger


# =====================================================
# Document Types
# =====================================================

class DocumentType(str, Enum):
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"


# =====================================================
# Semantic Chunker
# =====================================================

class SemanticChunker:
    """
    RAG-optimized semantic chunker.
    - Stable
    - Predictable
    - Language-aware (EN / VI / JA)
    """

    def __init__(
        self,
        target_tokens: int = None,
        overlap_tokens: int = None
    ):
        self.target_tokens = target_tokens or config.CHUNK_SIZE
        self.overlap_tokens = overlap_tokens or config.CHUNK_OVERLAP
        self.max_tokens = int(self.target_tokens * 1.4)
        self.min_tokens = int(self.target_tokens * 0.4)

    # =================================================
    # Token Estimation
    # =================================================

    def _is_cjk(self, text: str) -> bool:
        return bool(re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text))

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._is_cjk(text):
            return len(text)
        return max(1, len(text) // 3.5)

    # =================================================
    # Markdown Chunking
    # =================================================

    def _chunk_markdown(self, text: str) -> List[Tuple[str, Dict]]:
        chunks = []

        lines = text.splitlines()
        buffer = []
        token_count = 0

        section_stack = []
        current_level = 0

        def flush():
            nonlocal buffer, token_count
            if not buffer:
                return
            content = "\n".join(buffer).strip()
            if content:
                chunks.append((
                    content,
                    {
                        "doc_type": "markdown",
                        "section": " / ".join(section_stack),
                        "level": current_level,
                        "tokens": token_count
                    }
                ))
            buffer = []
            token_count = 0

        for line in lines:
            line = line.rstrip()
            if not line:
                continue

            heading = re.match(r'^(#{1,6})\s+(.*)', line)
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2).strip()

                section_stack = section_stack[:level - 1]
                section_stack.append(title)
                current_level = level

                buffer.append(line)
                token_count = self._estimate_tokens(line)
                continue

            line_tokens = self._estimate_tokens(line)

            if token_count + line_tokens > self.max_tokens:
                flush()

            buffer.append(line)
            token_count += line_tokens

        flush()
        return chunks

    # =================================================
    # Plain Text Chunking
    # =================================================

    def _chunk_plain_text(self, text: str) -> List[Tuple[str, Dict]]:
        chunks = []
        
        if "[TABLE]" in text and "[/TABLE]" in text:
            table_blocks = self._extract_table_blocks(text)
            if table_blocks:
                current_text = text
                for table_block in table_blocks:
                    before_table = current_text[:current_text.find(table_block)]
                    if before_table.strip():
                        chunks.extend(self._chunk_plain_text_no_table(before_table))
                    
                    table_tokens = self._estimate_tokens(table_block)
                    chunks.append((
                        table_block,
                        {
                            "doc_type": "table",
                            "tokens": table_tokens,
                            "has_table": True
                        }
                    ))
                    
                    current_text = current_text[current_text.find(table_block) + len(table_block):]
                
                if current_text.strip():
                    chunks.extend(self._chunk_plain_text_no_table(current_text))
                
                return chunks
        
        return self._chunk_plain_text_no_table(text)
    
    def _chunk_plain_text_no_table(self, text: str) -> List[Tuple[str, Dict]]:
        """Chunk plain text without table detection (internal helper to avoid recursion)"""
        chunks = []
        paragraphs = re.split(r'\n\s*\n+', text)

        buffer = []
        token_count = 0

        def flush():
            nonlocal buffer, token_count
            if not buffer:
                return
            content = "\n\n".join(buffer).strip()
            if content:
                chunks.append((
                    content,
                    {
                        "doc_type": "plain_text",
                        "tokens": token_count
                    }
                ))
            buffer = []
            token_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._estimate_tokens(para)

            if para_tokens > self.max_tokens:
                flush()
                sentences = re.split(r'(?<=[.!?])\s+', para)

                sub_buf = []
                sub_tokens = 0

                for sent in sentences:
                    sent_tokens = self._estimate_tokens(sent)
                    if sub_tokens + sent_tokens > self.max_tokens:
                        chunks.append((
                            " ".join(sub_buf),
                            {
                                "doc_type": "plain_text",
                                "tokens": sub_tokens
                            }
                        ))
                        sub_buf = [sent]
                        sub_tokens = sent_tokens
                    else:
                        sub_buf.append(sent)
                        sub_tokens += sent_tokens

                if sub_buf:
                    chunks.append((
                        " ".join(sub_buf),
                        {
                            "doc_type": "plain_text",
                            "tokens": sub_tokens
                        }
                    ))
                continue

            if token_count + para_tokens > self.max_tokens:
                flush()

            buffer.append(para)
            token_count += para_tokens

        flush()
        return chunks

    # =================================================
    # Public APIs
    # =================================================

    def chunk_with_metadata(
        self,
        text: str,
        doc_type: str = DocumentType.PLAIN_TEXT
    ) -> List[Tuple[str, Dict]]:
        if not text or not text.strip():
            return []

        text = text.strip()

        # FIX 1: Preserve table content as single chunk
        # Detect [TABLE]...[/TABLE] blocks and keep them intact
        if "[TABLE]" in text and "[/TABLE]" in text:
            table_blocks = self._extract_table_blocks(text)
            if table_blocks:
                chunks = []
                current_text = text
                
                for table_block in table_blocks:
                    # Extract text before table
                    before_table = current_text[:current_text.find(table_block)]
                    if before_table.strip():
                        # Chunk text before table normally
                        if doc_type == DocumentType.MARKDOWN or (
                            doc_type == DocumentType.PLAIN_TEXT and re.search(r'^#+\s+', before_table, re.MULTILINE)
                        ):
                            chunks.extend(self._chunk_markdown(before_table))
                        else:
                            chunks.extend(self._chunk_plain_text(before_table))
                    
                    # Keep table as single chunk
                    table_tokens = self._estimate_tokens(table_block)
                    chunks.append((
                        table_block,
                        {
                            "doc_type": "table",
                            "tokens": table_tokens,
                            "has_table": True
                        }
                    ))
                    
                    # Update current_text to after table
                    current_text = current_text[current_text.find(table_block) + len(table_block):]
                
                # Chunk remaining text after last table
                if current_text.strip():
                    if doc_type == DocumentType.MARKDOWN or (
                        doc_type == DocumentType.PLAIN_TEXT and re.search(r'^#+\s+', current_text, re.MULTILINE)
                    ):
                        chunks.extend(self._chunk_markdown(current_text))
                    else:
                        chunks.extend(self._chunk_plain_text(current_text))
                
                return chunks

        if doc_type == DocumentType.MARKDOWN or (
            doc_type == DocumentType.PLAIN_TEXT and re.search(r'^#+\s+', text, re.MULTILINE)
        ):
            return self._chunk_markdown(text)

        return self._chunk_plain_text(text)
    
    def _extract_table_blocks(self, text: str) -> List[str]:
        """
        Extract all [TABLE]...[/TABLE] blocks from text.
        
        Returns:
            List of table block strings (including markers)
        """
        blocks = []
        start_marker = "[TABLE]"
        end_marker = "[/TABLE]"
        
        start_idx = 0
        while True:
            start_pos = text.find(start_marker, start_idx)
            if start_pos == -1:
                break
            
            end_pos = text.find(end_marker, start_pos)
            if end_pos == -1:
                blocks.append(text[start_pos:])
                break
            
            table_block = text[start_pos:end_pos + len(end_marker)]
            blocks.append(table_block)
            start_idx = end_pos + len(end_marker)
        
        return blocks

    def chunk(self, text: str, doc_type: str = DocumentType.PLAIN_TEXT) -> List[str]:
        return [c[0] for c in self.chunk_with_metadata(text, doc_type)]


# =====================================================
# Legacy wrapper
# =====================================================

def chunk(text: str, doc_type: str = "plain_text") -> List[str]:
    chunker = SemanticChunker()
    return chunker.chunk(text, doc_type)
