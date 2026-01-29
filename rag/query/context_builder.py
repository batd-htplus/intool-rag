"""
Context Builder - Intelligent Context Assembly for LLM
======================================================

Purpose:
- Build well-structured context from grouped chunks
- Handle different query types (simple, comparison, complex)
- Implement smart context merging strategies
- Generate citation metadata

Context Building Strategies:
1. Simple Retrieval: One primary node → one context block
2. Comparison Queries: Multiple nodes with parallel structure
3. Complex Queries: Hierarchical context with parent/sibling context
4. Large Documents: Smart truncation with importance scoring

CRITICAL: Context must preserve:
- Node hierarchy (breadcrumbs)
- Chunk ordering (char_start based)
- Original text (NO summarization in context block)
- Citation information
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from rag.logging import logger
from rag.ingest.schemas import SemanticNode, Chunk, PageIndex


class QueryType(str, Enum):
    """Different query types require different context strategies"""
    SIMPLE = "simple"  # Single topic retrieval
    COMPARISON = "comparison"  # Compare two or more concepts
    HIERARCHICAL = "hierarchical"  # Need parent context
    DEFINITION = "definition"  # Explain a term
    ANALYTICAL = "analytical"  # Combine multiple ideas


@dataclass
class ContextBlock:
    """Single context block with metadata"""
    node_id: str
    node_title: str
    node_level: str
    block_type: str  # "primary", "parent", "sibling", "reference"
    text: str
    relevance_score: float = 1.0
    page: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_title": self.node_title,
            "node_level": self.node_level,
            "block_type": self.block_type,
            "text": self.text,
            "relevance_score": self.relevance_score,
            "page": self.page,
        }


class ContextBuilder:
    """
    Build optimized context for LLM answer generation.
    
    Handles:
    - Different query types
    - Smart hierarchy traversal
    - Token budgeting
    - Fallback strategies
    """
    
    def __init__(
        self,
        max_context_tokens: int = 3000,
        max_blocks: int = 5,
        enable_parent_context: bool = True,
        enable_sibling_context: bool = False,
    ):
        self.max_context_tokens = max_context_tokens
        self.max_blocks = max_blocks
        self.enable_parent_context = enable_parent_context
        self.enable_sibling_context = enable_sibling_context
    
    # ========================================================================
    # QUERY TYPE DETECTION
    # ========================================================================
    
    def detect_query_type(self, query: str) -> QueryType:
        """
        Detect query type from user question.
        
        Heuristics:
        - "compare", "difference", "vs" → COMPARISON
        - "define", "what is", "explain" → DEFINITION
        - "how does", "why" → ANALYTICAL
        - Otherwise → SIMPLE
        """
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["compare", "difference", "vs", "versus"]):
            return QueryType.COMPARISON
        if any(w in query_lower for w in ["define", "what is", "explain", "describe"]):
            return QueryType.DEFINITION
        if any(w in query_lower for w in ["how", "why", "analyze"]):
            return QueryType.ANALYTICAL
        
        return QueryType.SIMPLE
    
    # ========================================================================
    # CONTEXT ASSEMBLY STRATEGIES
    # ========================================================================
    
    def build_context_simple(
        self,
        primary_node_id: str,
        node_chunks: Dict[str, List[Chunk]],
        page_index: PageIndex,
    ) -> List[ContextBlock]:
        """
        Simple context: Primary node only.
        
        Suitable for: Direct factual queries
        """
        blocks = []
        
        primary_node = page_index.get_node(primary_node_id)
        if not primary_node:
            return blocks
        
        # Add primary block
        primary_text = self._assemble_node_text(
            primary_node,
            node_chunks.get(primary_node_id, []),
        )
        
        blocks.append(ContextBlock(
            node_id=primary_node_id,
            node_title=primary_node.title,
            node_level=primary_node.level.value,
            block_type="primary",
            text=primary_text,
            relevance_score=1.0,
            page=primary_node.page_index,
        ))
        
        return blocks
    
    def build_context_with_hierarchy(
        self,
        primary_node_id: str,
        node_chunks: Dict[str, List[Chunk]],
        page_index: PageIndex,
    ) -> List[ContextBlock]:
        """
        Hierarchical context: Primary + parent + siblings.
        
        Suitable for: Context-dependent queries
        """
        blocks = []
        
        primary_node = page_index.get_node(primary_node_id)
        if not primary_node:
            return blocks
        
        # Add parent context (breadcrumb)
        if self.enable_parent_context and primary_node.parent_id:
            parent = page_index.get_node(primary_node.parent_id)
            if parent:
                parent_text = self._assemble_node_text(parent, [])
                blocks.append(ContextBlock(
                    node_id=parent.node_id,
                    node_title=parent.title,
                    node_level=parent.level.value,
                    block_type="parent",
                    text=parent_text,
                    relevance_score=0.7,
                    page=parent.page_index,
                ))
        
        # Add primary block
        primary_text = self._assemble_node_text(
            primary_node,
            node_chunks.get(primary_node_id, []),
        )
        
        blocks.append(ContextBlock(
            node_id=primary_node_id,
            node_title=primary_node.title,
            node_level=primary_node.level.value,
            block_type="primary",
            text=primary_text,
            relevance_score=1.0,
            page=primary_node.page_index,
        ))
        
        # Add sibling context (related topics)
        if self.enable_sibling_context and primary_node.parent_id:
            parent = page_index.get_node(primary_node.parent_id)
            if parent:
                for sibling_id in parent.children:
                    if sibling_id != primary_node_id:
                        sibling = page_index.get_node(sibling_id)
                        if sibling and sibling_id in node_chunks:
                            sibling_text = self._assemble_node_text(
                                sibling,
                                node_chunks[sibling_id],
                            )
                            blocks.append(ContextBlock(
                                node_id=sibling_id,
                                node_title=sibling.title,
                                node_level=sibling.level.value,
                                block_type="sibling",
                                text=sibling_text,
                                relevance_score=0.6,
                                page=sibling.page_index,
                            ))
        
        return blocks
    
    def build_context_comparison(
        self,
        primary_node_ids: List[str],
        node_chunks: Dict[str, List[Chunk]],
        page_index: PageIndex,
    ) -> List[ContextBlock]:
        """
        Comparison context: Multiple nodes side-by-side.
        
        Suitable for: Compare/contrast queries
        """
        blocks = []
        
        for node_id in primary_node_ids:
            node = page_index.get_node(node_id)
            if not node:
                continue
            
            text = self._assemble_node_text(
                node,
                node_chunks.get(node_id, []),
            )
            
            blocks.append(ContextBlock(
                node_id=node_id,
                node_title=node.title,
                node_level=node.level.value,
                block_type="comparison",
                text=text,
                relevance_score=1.0,
                page=node.page_index,
            ))
        
        return blocks
    
    # ========================================================================
    # TEXT ASSEMBLY
    # ========================================================================
    
    def _assemble_node_text(
        self,
        node: SemanticNode,
        chunks: List[Chunk],
        include_prefix: bool = True,
    ) -> str:
        """
        Assemble formatted text for a node.
        
        Structure:
        [summary - concise description]
        [chunks - raw text in order]
        """
        lines = []
        
        if not include_prefix and node.title:
            lines.append(f"## {node.title}")
            lines.append("")
        
        # Add summary (semantic understanding)
        if node.summary:
            lines.append(node.summary)
            lines.append("")
        
        # Add chunks (raw text, ordered by position)
        if chunks:
            sorted_chunks = sorted(chunks, key=lambda c: c.seq_index)
            for chunk in sorted_chunks:
                lines.append(chunk.text)
                lines.append("")
        
        return "\n".join(lines).strip()
    
    # ========================================================================
    # SMART CONTEXT TRUNCATION
    # ========================================================================
    
    def truncate_context(
        self,
        blocks: List[ContextBlock],
        max_tokens: Optional[int] = None,
    ) -> List[ContextBlock]:
        """
        Truncate context blocks to fit token budget.
        
        Strategy:
        1. Keep primary block (always)
        2. Keep parent block (if exists)
        3. Add remaining blocks by relevance score
        4. Truncate individual blocks if needed
        """
        if not blocks:
            return blocks
        
        max_tokens = max_tokens or self.max_context_tokens
        
        # Separate block types
        primary_blocks = [b for b in blocks if b.block_type == "primary"]
        parent_blocks = [b for b in blocks if b.block_type == "parent"]
        other_blocks = [b for b in blocks if b.block_type not in ["primary", "parent"]]
        
        # Sort other blocks by relevance
        other_blocks.sort(key=lambda b: b.relevance_score, reverse=True)
        
        # Build result with prioritization
        result = []
        token_count = 0
        
        # Always include primary + parent
        for block in primary_blocks + parent_blocks:
            result.append(block)
            token_count += self._estimate_tokens(block.text)
        
        # Add other blocks until budget
        for block in other_blocks:
            block_tokens = self._estimate_tokens(block.text)
            if token_count + block_tokens > max_tokens:
                # Try truncating the block
                truncated_text = self._truncate_text(
                    block.text,
                    max_tokens - token_count,
                )
                if truncated_text:
                    block.text = truncated_text
                    result.append(block)
                break
            
            result.append(block)
            token_count += block_tokens
        
        logger.info(
            f"Truncated context to {len(result)} blocks, "
            f"~{token_count} tokens"
        )
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens (words / 0.75)"""
        words = len(text.split())
        return max(1, int(words / 0.75))
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit"""
        max_words = int(max_tokens * 0.75)
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
            return " ".join(words) + "..."
        return text
    
    # ========================================================================
    # CONTEXT FORMATTING
    # ========================================================================
    
    def format_context_for_llm(self, blocks: List[ContextBlock]) -> str:
        """
        Format context blocks into single string for LLM.
        
        Output format:
        [Context Block 1]
        
        [Context Block 2]
        ...
        """
        formatted = []
        
        for i, block in enumerate(blocks):
            # Add block separator
            if i > 0:
                formatted.append("\n" + "=" * 60 + "\n")
            
            # Add metadata comment
            formatted.append(f"[Context from {block.node_title} (node: {block.node_id})]")
            formatted.append("")
            
            # Add block text
            formatted.append(block.text)
        
        return "\n".join(formatted)
    
    # ========================================================================
    # CITATION BUILDER
    # ========================================================================
    
    def build_citations(self, blocks: List[ContextBlock]) -> List[Dict[str, Any]]:
        """
        Build citation information from context blocks.
        
        Returns:
        [
            {
                "node_id": "0007",
                "title": "3.2.1 Scaled Dot-Product Attention",
                "node_level": "subsection",
                "page": 4
            }
        ]
        """
        citations = []
        seen = set()
        
        for block in blocks:
            if block.node_id not in seen:
                citations.append({
                    "node_id": block.node_id,
                    "title": block.node_title,
                    "node_level": block.node_level,
                    "page": block.page,
                })
                seen.add(block.node_id)
        
        return citations
    
    # ========================================================================
    # HIGH-LEVEL API
    # ========================================================================
    
    def build_context_adaptive(
        self,
        query: str,
        primary_node_id: str,
        comparison_node_ids: Optional[List[str]] = None,
        node_chunks: Optional[Dict[str, List[Chunk]]] = None,
        page_index: Optional[PageIndex] = None,
    ) -> Dict[str, Any]:
        """
        Adaptive context building based on query type.
        
        Returns:
        {
            "query_type": "simple",
            "blocks": [ContextBlock, ...],
            "formatted_text": "...",
            "citations": [...]
        }
        """
        query_type = self.detect_query_type(query)
        
        node_chunks = node_chunks or {}
        
        # Build blocks based on query type
        if query_type == QueryType.COMPARISON and comparison_node_ids:
            blocks = self.build_context_comparison(
                comparison_node_ids,
                node_chunks,
                page_index,
            )
        elif query_type in [QueryType.ANALYTICAL, QueryType.HIERARCHICAL]:
            blocks = self.build_context_with_hierarchy(
                primary_node_id,
                node_chunks,
                page_index,
            )
        else:
            blocks = self.build_context_simple(
                primary_node_id,
                node_chunks,
                page_index,
            )
        
        # Truncate to budget
        blocks = self.truncate_context(blocks)
        
        # Format for LLM
        formatted_text = self.format_context_for_llm(blocks)
        
        # Build citations
        citations = self.build_citations(blocks)
        
        return {
            "query_type": query_type.value,
            "blocks": [b.to_dict() for b in blocks],
            "formatted_text": formatted_text,
            "citations": citations,
            "block_count": len(blocks),
            "estimated_tokens": self._estimate_tokens(formatted_text),
        }
