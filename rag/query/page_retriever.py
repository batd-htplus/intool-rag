"""
Page-Level Retrieval & Ranking
===============================

Purpose:
- Retrieve chunks using semantic search (FAISS)
- Group chunks by page
- Rank pages by relevance
- Select best pages for context assembly

Key difference from chunk-level:
- Chunk retrieval: Get top-K chunks globally
- Page ranking: Group chunks by page, then rank pages
- Final selection: Return best pages (not scattered chunks)
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from rag.logging import logger
from rag.config import config
from rag.llm.embedding_service import get_embedding_provider
from rag.cache import get_query_cache


@dataclass
class RetrievedChunk:
    """Retrieved chunk with similarity score"""
    chunk_id: str
    text: str
    score: float
    page: int
    metadata: Dict[str, Any]


@dataclass
class PageRanking:
    """Page with ranking score and associated chunks"""
    page: int
    score: float
    chunks: List[RetrievedChunk]
    metadata: Dict[str, Any]
    
    def get_context_text(self) -> str:
        """Get formatted context from chunks on this page"""
        lines = []
        
        hierarchy = []
        if self.metadata.get("chapter"):
            hierarchy.append(f"Chapter {self.metadata['chapter']}")
        if self.metadata.get("section"):
            hierarchy.append(f"Section {self.metadata['section']}")
        if self.metadata.get("title"):
            hierarchy.append(f"{self.metadata['title']}")
        
        if hierarchy:
            lines.append(f"[{' | '.join(hierarchy)}]")
            lines.append("")
        
        for chunk in self.chunks:
            lines.append(chunk.text)
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def to_citation(self) -> Dict[str, Any]:
        """Convert to citation format for LLM response"""
        return {
            "page": self.page,
            "chapter": self.metadata.get("chapter"),
            "section": self.metadata.get("section"),
            "subsection": self.metadata.get("subsection"),
            "title": self.metadata.get("title"),
            "source_file": self.metadata.get("source_filename"),
            "relevance_score": round(self.score, 3),
        }


class PageLevelRetriever:
    """Retrieve and rank at page level"""
    
    def __init__(self, top_chunks: int = 50, top_pages: int = 5):
        """
        Initialize retriever.
        
        Args:
            top_chunks: How many chunks to retrieve initially
            top_pages: How many pages to return for context assembly
        """
        self.top_chunks = top_chunks
        self.top_pages = top_pages
    
    async def retrieve_chunks(
        self,
        query: str,
        project: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-K chunks using semantic search.
        
        Args:
            query: User query
            project: Optional project filter
            
        Returns:
            List of RetrievedChunk objects
        """
        logger.info(f"Retrieving top-{self.top_chunks} chunks for query")
        
        embedding_provider = get_embedding_provider()
        query_embedding = await embedding_provider.embed_single(query)
        
        from rag.storage.faiss_index import search_faiss_by_vector
        from rag.storage.file_storage import FileStorageManager
        from rag.config import config
        
        storage = FileStorageManager(config.STORAGE_DIR)
        search_results = await search_faiss_by_vector(
            query_embedding,
            limit=self.top_chunks,
            project=project,
        )
        
        chunks = []
        for result in search_results:
            chunk = RetrievedChunk(
                chunk_id=result.get("chunk_id", "unknown"),
                text=result.get("text", ""),
                score=result.get("score", 0),
                page=result.get("page", 0),
                metadata={
                    "chapter": result.get("chapter"),
                    "section": result.get("section"),
                    "subsection": result.get("subsection"),
                    "title": result.get("title"),
                    "source_filename": result.get("source_filename"),
                    "doc_id": result.get("doc_id"),
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        
        return chunks
    
    def group_chunks_by_page(
        self,
        chunks: List[RetrievedChunk],
    ) -> Dict[int, List[RetrievedChunk]]:
        """
        Group chunks by page number.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Dict of {page: [chunks]}
        """
        grouped = {}
        for chunk in chunks:
            if chunk.page not in grouped:
                grouped[chunk.page] = []
            grouped[chunk.page].append(chunk)
        
        return grouped
    
    def rank_pages(
        self,
        chunks_by_page: Dict[int, List[RetrievedChunk]],
    ) -> List[PageRanking]:
        """
        Rank pages by relevance.
        
        Strategy:
        1. Score each page based on chunks' relevance scores
        2. Consider number of relevant chunks
        3. Sort by combined score
        
        Args:
            chunks_by_page: Chunks grouped by page
            
        Returns:
            Sorted list of PageRanking
        """
        rankings = []
        
        for page_num, page_chunks in chunks_by_page.items():
            # Calculate page score
            # - Average of chunk scores
            # - Boosted by number of relevant chunks
            
            avg_score = sum(c.score for c in page_chunks) / len(page_chunks)
            
            # Boost for multiple chunks (better coverage)
            chunk_boost = min(len(page_chunks) * 0.05, 0.15)
            
            combined_score = avg_score + chunk_boost
            
            # Get metadata from first chunk (should be same for all)
            metadata = page_chunks[0].metadata
            
            ranking = PageRanking(
                page=page_num,
                score=combined_score,
                chunks=page_chunks,
                metadata=metadata,
            )
            
            rankings.append(ranking)
        
        # Sort by score descending
        rankings.sort(key=lambda r: r.score, reverse=True)
        
        return rankings
    
    def select_top_pages(
        self,
        rankings: List[PageRanking],
        max_pages: Optional[int] = None,
    ) -> List[PageRanking]:
        """
        Select top N pages.
        
        Args:
            rankings: Ranked pages
            max_pages: Maximum pages to return
            
        Returns:
            Top N pages
        """
        max_pages = max_pages or self.top_pages
        
        selected = rankings[:max_pages]
        
        logger.info(f"Selected {len(selected)} pages from {len(rankings)} candidates")
        
        return selected
    
    async def retrieve_and_rank_pages(
        self,
        query: str,
        project: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> List[PageRanking]:
        """
        Complete retrieval and ranking pipeline.
        
        Args:
            query: User query
            project: Optional project filter
            max_pages: Maximum pages to return
            
        Returns:
            Ranked pages ready for context assembly
        """
        chunks = await self.retrieve_chunks(query, project)
        
        if not chunks:
            logger.warning("No chunks retrieved")
            return []
        
        chunks_by_page = self.group_chunks_by_page(chunks)
        
        rankings = self.rank_pages(chunks_by_page)
        
        selected = self.select_top_pages(rankings, max_pages)
        
        return selected


# Convenience functions
async def retrieve_and_rank_pages(
    query: str,
    project: Optional[str] = None,
    top_pages: int = 5,
) -> List[PageRanking]:
    """
    Retrieve and rank pages for a query.
    
    Args:
        query: User query
        project: Optional project filter
        top_pages: Number of pages to return
        
    Returns:
        List of ranked PageRanking objects
    """
    retriever = PageLevelRetriever(top_pages=top_pages)
    return await retriever.retrieve_and_rank_pages(query, project, top_pages)
