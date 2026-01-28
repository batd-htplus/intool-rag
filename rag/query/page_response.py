"""
Page-Aware Response Assembly & Citation
========================================

Purpose:
- Assemble context from selected pages
- Format for LLM consumption
- Generate traceable citations
- Ensure quality and relevance
"""

from typing import List, Dict, Any, Optional
from rag.query.page_retriever import PageRanking
from rag.logging import logger


class ResponseAssembler:
    """Assemble LLM response with page-aware context and citations"""
    
    def __init__(self, max_context_length: int = 8000):
        """
        Initialize assembler.
        
        Args:
            max_context_length: Maximum context length for LLM
        """
        self.max_context_length = max_context_length
    
    def assemble_context(
        self,
        ranked_pages: List[PageRanking],
        max_length: Optional[int] = None,
    ) -> str:
        """
        Assemble context from ranked pages.
        
        Format:
        [Page 12 | Chapter 2 | Section 2.1 | Query Preprocessing]
        ...context...
        
        [Page 13 | Chapter 2 | Section 2.1]
        ...context...
        
        Args:
            ranked_pages: Ranked pages from retriever
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not ranked_pages:
            return ""
        
        max_length = max_length or self.max_context_length
        
        context_parts = []
        total_length = 0
        
        for page_ranking in ranked_pages:
            hierarchy = []
            if page_ranking.metadata.get("chapter"):
                hierarchy.append(f"Chapter {page_ranking.metadata['chapter']}")
            if page_ranking.metadata.get("section"):
                hierarchy.append(f"Section {page_ranking.metadata['section']}")
            if page_ranking.metadata.get("title"):
                hierarchy.append(f"{page_ranking.metadata['title']}")
            
            page_header = f"[Page {page_ranking.page}"
            if hierarchy:
                page_header += f" | {' | '.join(hierarchy)}"
            page_header += "]"
            
            page_context = page_ranking.get_context_text()
            section = f"{page_header}\n{page_context}\n"
            
            section_length = len(section)
            
            if total_length + section_length > max_length and context_parts:
                logger.warning(
                    f"Context length limit ({max_length}) reached, "
                    f"truncating at {len(context_parts)} pages"
                )
                break
            
            context_parts.append(section)
            total_length += section_length
        
        context = "\n".join(context_parts).strip()
        
        logger.info(f"Assembled context: {len(context)} chars from {len(context_parts)} pages")
        
        return context
    
    def build_prompt(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build complete prompt for LLM.
        
        Args:
            question: User question
            context: Assembled context
            system_prompt: Optional custom system prompt
            
        Returns:
            Complete prompt
        """
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()
        
        if context:
            prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        else:
            prompt = f"""{system_prompt}

QUESTION:
{question}

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def _get_default_system_prompt() -> str:
        """Get default system prompt"""
        return """You are a helpful assistant that answers questions based on provided context.

Instructions:
- Answer only based on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise and factual
- Cite specific sections or pages when relevant
- If uncertain, explain the level of confidence"""


class CitationFormatter:
    """Format citations from page rankings"""
    
    def create_citation_from_page(
        self,
        page_ranking: PageRanking,
    ) -> Dict[str, Any]:
        """
        Create citation from page ranking.
        
        Args:
            page_ranking: PageRanking object
            
        Returns:
            Citation dict
        """
        return page_ranking.to_citation()
    
    def create_citations(
        self,
        ranked_pages: List[PageRanking],
    ) -> List[Dict[str, Any]]:
        """
        Create citations from ranked pages.
        
        Args:
            ranked_pages: List of PageRanking
            
        Returns:
            List of citation dicts
        """
        citations = []
        for page_ranking in ranked_pages:
            citation = self.create_citation_from_page(page_ranking)
            citations.append(citation)
        
        return citations
    
    def format_sources(
        self,
        ranked_pages: List[PageRanking],
    ) -> Dict[str, Any]:
        """
        Format all sources for response.
        
        Args:
            ranked_pages: List of PageRanking
            
        Returns:
            Formatted sources
        """
        sources = []
        
        for page_ranking in ranked_pages:
            source = {
                "page": page_ranking.page,
                "relevance_score": round(page_ranking.score, 3),
            }
            
            # Add structure info if available
            if page_ranking.metadata.get("chapter"):
                source["chapter"] = page_ranking.metadata["chapter"]
            if page_ranking.metadata.get("section"):
                source["section"] = page_ranking.metadata["section"]
            if page_ranking.metadata.get("title"):
                source["title"] = page_ranking.metadata["title"]
            
            source["source_file"] = page_ranking.metadata.get("source_filename", "unknown")
            
            sources.append(source)
        
        return {
            "primary_sources": sources[:3] if sources else [],
            "all_sources": sources,
            "total_sources": len(sources),
        }


class PageAwareResponse:
    """Complete RAG response with context and citations"""
    
    def __init__(
        self,
        answer: str,
        ranked_pages: List[PageRanking],
    ):
        self.answer = answer
        self.ranked_pages = ranked_pages
        self.citation_formatter = CitationFormatter()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API response"""
        sources = self.citation_formatter.format_sources(self.ranked_pages)
        
        return {
            "answer": self.answer,
            "sources": sources,
            "confidence": self._estimate_confidence(),
        }
    
    def _estimate_confidence(self) -> str:
        """Estimate confidence level"""
        if not self.ranked_pages:
            return "low"
        
        avg_score = sum(p.score for p in self.ranked_pages) / len(self.ranked_pages)
        
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"


# Convenience functions
def assemble_page_context(
    ranked_pages: List[PageRanking],
    max_length: int = 8000,
) -> str:
    """Assemble context from ranked pages"""
    assembler = ResponseAssembler(max_context_length=max_length)
    return assembler.assemble_context(ranked_pages, max_length)


def build_rag_prompt(
    question: str,
    ranked_pages: List[PageRanking],
    max_context_length: int = 8000,
) -> str:
    """Build complete RAG prompt"""
    assembler = ResponseAssembler(max_context_length=max_context_length)
    context = assembler.assemble_context(ranked_pages, max_context_length)
    return assembler.build_prompt(question, context)


def create_page_aware_response(
    answer: str,
    ranked_pages: List[PageRanking],
) -> Dict[str, Any]:
    """Create page-aware response with citations"""
    response = PageAwareResponse(answer, ranked_pages)
    return response.to_dict()
