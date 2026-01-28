"""
LLM-Based Content Summarizer
==============================

Uses LLM to create deterministic summaries of document content.

Output: JSON with structure analysis + summary
- chapter, section, title (extracted)
- summary (LLM-generated)
- key_topics (LLM-extracted)
- content_type (analysis)

IMPORTANT: No text modification - summary references original only
"""

from typing import Dict, Any, Optional
import json
from rag.logging import logger
from rag.core.container import get_container
from rag.query.prompt_templates import (
    get_summarization_prompt,
    get_structure_analysis_prompt,
)


class LLMContentSummarizer:
    """Summarize page content using LLM"""
    
    def __init__(self):
        self.llm_provider = None
    
    def _get_llm_provider(self):
        """Get LLM provider from DI container"""
        if not self.llm_provider:
            self.llm_provider = get_container().get_llm_provider()
        return self.llm_provider
    
    async def analyze_page_structure(
        self,
        page: int,
        page_content: str,
    ) -> Dict[str, Any]:
        """
        Analyze page structure using LLM.
        
        Extracts:
        - chapter, section, title
        - main topics
        
        Args:
            page: Page number
            page_content: Clean page text
            
        Returns:
            Dict with structure info
        """
        if not page_content or len(page_content) < 50:
            logger.debug(f"Page {page}: Too short for analysis")
            return {
                "page": page,
                "chapter": None,
                "section": None,
                "title": None,
                "topics": [],
            }
        
        try:
            llm = self._get_llm_provider()
            template = get_structure_analysis_prompt()
            
            # Limit content length for LLM (first 2000 chars)
            content_excerpt = page_content[:2000]
            
            prompt = template.format(page_content=content_excerpt)
            
            # Get LLM analysis
            response = await llm.generate(
                prompt,
                temperature=0.3,  # Low temp for consistency
                max_tokens=300,
            )
            
            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # If not valid JSON, extract fields manually
                logger.warning(f"Page {page}: LLM response not JSON, parsing manually")
                result = self._parse_manual_structure(response)
            
            result["page"] = page
            return result
        
        except Exception as e:
            logger.warning(f"Page {page}: Structure analysis failed: {e}")
            return {
                "page": page,
                "chapter": None,
                "section": None,
                "title": None,
                "topics": [],
            }
    
    async def summarize_content(
        self,
        page: int,
        page_content: str,
        max_length: int = 150,
    ) -> Dict[str, Any]:
        """
        Summarize page content.
        
        Args:
            page: Page number
            page_content: Clean page text
            max_length: Maximum summary length in words
            
        Returns:
            Dict with summary and metadata
        """
        if not page_content or len(page_content) < 50:
            return {
                "page": page,
                "summary": page_content[:100],
                "summary_length": len(page_content),
            }
        
        try:
            llm = self._get_llm_provider()
            template = get_summarization_prompt()
            
            # Limit content length (first 3000 chars)
            content_excerpt = page_content[:3000]
            
            prompt = template.format(content=content_excerpt)
            
            summary = await llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=150,
            )
            
            return {
                "page": page,
                "summary": summary.strip(),
                "summary_length": len(summary.split()),
                "original_length": len(page_content.split()),
            }
        
        except Exception as e:
            logger.warning(f"Page {page}: Summarization failed: {e}")
            # Fallback to simple truncation
            words = page_content.split()[:max_length]
            return {
                "page": page,
                "summary": " ".join(words) + "...",
                "summary_length": len(words),
            }
    
    async def analyze_full_page(
        self,
        page: int,
        page_content: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive page analysis: structure + summary.
        
        Args:
            page: Page number
            page_content: Clean page text
            
        Returns:
            Dict with both structure and summary
        """
        logger.debug(f"Page {page}: Starting comprehensive analysis")
        
        # Run in parallel if possible
        structure = await self.analyze_page_structure(page, page_content)
        summary = await self.summarize_content(page, page_content)
        
        # Merge results
        result = {
            **structure,
            **summary,
            "analysis_complete": True,
        }
        
        return result
    
    def _parse_manual_structure(self, response: str) -> Dict[str, Any]:
        """
        Manually parse structure if JSON parsing fails.
        
        Args:
            response: LLM response text
            
        Returns:
            Dict with extracted fields
        """
        # Simple extraction - improve as needed
        return {
            "chapter": None,
            "section": None,
            "title": None,
            "topics": [],
            "raw_response": response,
        }


# Convenience function
async def analyze_page_content(
    page: int,
    page_content: str,
) -> Dict[str, Any]:
    """
    Analyze page content using LLM.
    
    Args:
        page: Page number
        page_content: Clean page text
        
    Returns:
        Analysis result dict
    """
    summarizer = LLMContentSummarizer()
    return await summarizer.analyze_full_page(page, page_content)
