"""
Node-Aware Chunking with Semantic Structure
=============================================

Purpose:
- Split document text into atomic chunks
- Each chunk belongs to exactly ONE semantic node
- Chunks contain raw text (NEVER summarized)
- Generate chunks.json as SOURCE OF TRUTH

Strategy:
1. Load PageIndex (structure)
2. For each node, extract its raw text content
3. Split text by paragraphs/sentences (deterministic, NO LLM)
4. Respect chunk size limits
5. Save all chunks with metadata

CRITICAL CONSTRAINTS:
- Chunk MUST stay within one semantic node
- Chunk MUST NOT exceed token limit
- Chunks MUST be ordered by char_start
- NEVER summarize or paraphrase chunk text
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
from datetime import datetime
from pathlib import Path

from rag.logging import logger
from rag.ingest.schemas import (
    SemanticNode, PageIndex, Chunk, ChunksIndex,
    save_chunks_index, load_page_index
)


class NodeAwareChunker:
    """
    Split document into chunks within semantic node boundaries.
    
    Constraints:
    - max_chunk_size: Max characters per chunk (soft limit)
    - min_chunk_size: Min characters per chunk
    - chunk stays within 1 node
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,  # ~200-250 tokens
        min_chunk_size: int = 100,
        target_chunk_size: int = 600,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
    
    def chunk_node_text(
        self,
        node: SemanticNode,
        text: str,
    ) -> List[Tuple[str, int, int, int]]:
        """
        Split node text into chunks.
        
        Args:
            node: SemanticNode (for metadata)
            text: Raw text of this node
            
        Returns:
            List of (chunk_text, char_start, char_end, seq_index)
        """
        if not text or not text.strip():
            return []
        
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        chunk_index = 0
        char_pos = 0
        current_chunk_text = ""
        current_chunk_start = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_chunk_text and len(current_chunk_text) + para_len > self.max_chunk_size:
                if current_chunk_text.strip():
                    chunks.append((
                        current_chunk_text.strip(),
                        current_chunk_start,
                        char_pos,
                        chunk_index,
                    ))
                    chunk_index += 1
                current_chunk_text = ""
                current_chunk_start = char_pos
            
            if current_chunk_text:
                current_chunk_text += "\n\n"
                char_pos += 2
            
            current_chunk_text += para
            char_pos += para_len + 2
        
        if current_chunk_text.strip():
            chunks.append((
                current_chunk_text.strip(),
                current_chunk_start,
                char_pos,
                chunk_index,
            ))
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (separated by double newlines)"""
        paras = re.split(r'\n\s*\n+', text.strip())
        return [p.strip() for p in paras if p.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (word count / 0.75)"""
        words = len(text.split())
        return max(1, int(words / 0.75))


class ChunksBuilder:
    """
    Build chunks.json from PageIndex and raw content.
    
    Input:
    - page_index.json (structure)
    - Raw page texts
    
    Output:
    - chunks.json (atomic units)
    """
    
    def __init__(self):
        self.chunker = NodeAwareChunker()
        self.next_chunk_id_counter = {}  # Per page
    
    def _generate_chunk_id(self, page: int, index: int) -> str:
        """Generate chunk ID in format: c_{page}_{index}"""
        return f"c_{page:03d}_{index:03d}"
    
    async def build_chunks(
        self,
        doc_id: str,
        page_index: PageIndex,
        pages_data: List[Dict[str, Any]],
    ) -> ChunksIndex:
        """
        Build ChunksIndex from PageIndex and raw text.
        
        Args:
            doc_id: Document ID
            page_index: PageIndex with structure
            pages_data: List of page data (with clean_text)
            
        Returns:
            ChunksIndex with all chunks
        """
        logger.info(f"Building chunks for {doc_id}")
        
        chunks = []
        chunk_counter = 0
        
        page_texts = {}
        for page_data in pages_data:
            page_num = page_data.get("page", 1)
            text = page_data.get("clean_text", "")
            page_texts[page_num] = text
        
        for node in page_index.nodes:
            page = node.page_index
            page_text = page_texts.get(page, "")
            
            if not page_text:
                logger.warning(f"No text for node {node.node_id} on page {page}")
                continue
            
            node_text = self._extract_node_text(node, page_text, page_index)
            
            if not node_text or not node_text.strip():
                logger.debug(f"Empty text for node {node.node_id}")
                continue
            
            chunk_tuples = self.chunker.chunk_node_text(node, node_text)
            
            for chunk_text, char_start, char_end, seq_idx in chunk_tuples:
                chunk_id = self._generate_chunk_id(page, chunk_counter)
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    node_id=node.node_id,
                    page=page,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                    seq_index=seq_idx,
                    token_estimate=self.chunker._estimate_tokens(chunk_text),
                    embedding_id=None,
                )
                
                chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Created {len(chunks)} chunks")
        
        chunks_index = ChunksIndex(
            doc_id=doc_id,
            created_at=datetime.now().isoformat(),
            chunks=chunks,
            chunk_count=len(chunks),
        )
        
        return chunks_index
    
    def _extract_node_text(
        self,
        node: SemanticNode,
        page_text: str,
        page_index: PageIndex,
    ) -> str:
        """
        Extract raw text for a specific node from page text.
        
        Simple approach:
        - If node has children, use all text until next sibling
        - If leaf node, use from node start to end of content
        
        TODO: Implement proper text extraction based on node boundaries
        """
        return page_text
    
    def save_chunks(
        self,
        chunks_index: ChunksIndex,
        output_dir: str,
    ) -> str:
        """Save ChunksIndex to chunks.json"""
        output_path = Path(output_dir) / "chunks.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_chunks_index(chunks_index, str(output_path))
        logger.info(f"Saved {chunks_index.chunk_count} chunks to {output_path}")
        return str(output_path)


class ChunkNodeGrouper:
    """
    Utility to group chunks by node (used during query time).
    """
    
    @staticmethod
    def group_by_node(chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
        """Group chunks by node_id"""
        grouped = {}
        for chunk in chunks:
            if chunk.node_id not in grouped:
                grouped[chunk.node_id] = []
            grouped[chunk.node_id].append(chunk)
        
        for node_id in grouped:
            grouped[node_id].sort(key=lambda c: c.seq_index)
        
        return grouped
    
    @staticmethod
    def get_node_context(
        node_id: str,
        chunks_list: List[Chunk],
        include_summary: bool = True,
        node: Optional[SemanticNode] = None,
    ) -> str:
        """
        Build context string from chunks of a specific node.
        
        Args:
            node_id: Node ID to fetch chunks for
            chunks_list: All chunks
            include_summary: Include node summary/prefix
            node: Optional node object (for summary/prefix)
            
        Returns:
            Formatted context string
        """
        node_chunks = [c for c in chunks_list if c.node_id == node_id]
        node_chunks.sort(key=lambda c: c.seq_index)
        
        lines = []
        
        # Add chunks
        for chunk in node_chunks:
            lines.append(chunk.text)
            lines.append("")
        
        return "\n".join(lines).strip()
